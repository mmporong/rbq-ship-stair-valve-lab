"""Evaluate an exported RBQ stair policy in native MuJoCo on CPU.

The same bundle can be evaluated in MuJoCo 3.5 and 4.0. The exported mjlab
fixtures check observations, actor outputs and position commands before rollout.
"""

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Collection, Iterator
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from statistics import median_low

import mujoco
import numpy as np

ACTOR_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "command",
    "phase",
    "joint_pos",
    "joint_vel",
    "actions",
    "height_scan",
)
DOWN = np.array([0.0, 0.0, -1.0])
BAD_DYNAMICS = tuple(
    int(getattr(mujoco.mjtWarning, f"mjWARN_BAD{name}"))
    for name in ("QPOS", "QVEL", "QACC", "CTRL")
)
MIN_SUPPORT_FORCE_N = 1.0


def supporting_contacts(
    model: mujoco.MjModel, data: mujoco.MjData, feet: Collection[int]
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield foot and surface ids for loaded contacts with upward-facing normals."""
    wrench = np.zeros(6)
    for contact_id, contact in enumerate(data.contact):
        if contact.efc_address < 0:
            continue
        surface, foot = map(int, contact.geom)
        upward = contact.frame[2]
        if surface in feet:
            surface, foot = foot, surface
            upward = -upward
        if foot in feet and surface not in feet and upward > 0.5:
            mujoco.mj_contactForce(model, data, contact_id, wrench)
            if wrench[0] * upward >= MIN_SUPPORT_FORCE_N:
                yield foot, surface, contact.pos


class CpuStairs:
    def __init__(self, bundle: Path, *, model: mujoco.MjModel | None = None):
        self.meta = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        if model is None:
            model = mujoco.MjModel.from_xml_path(str(bundle / "scene.xml"))
        self.model = model
        if [model.joint(i).name for i in range(model.njnt)] != self.meta["model_joint_order"]:
            raise ValueError("Joint order changed when compiling the exported scene")
        with np.load(bundle / "actor.npz", allow_pickle=False) as actor:
            count = sum(key.startswith("weight_") for key in actor.files)
            self.layers = [(actor[f"weight_{i}"], actor[f"bias_{i}"]) for i in range(count)]
            self.mean = actor["obs_mean"]
            self.scale = actor["obs_std"] + actor["obs_eps"]
            self.ray_offsets = actor["ray_offsets"].astype(np.float64)
        obs_dims = [3, 3, 3, 2, *([len(self.meta["joint_names"])] * 2)]
        obs_dims += [len(self.meta["action_joints"]), len(self.ray_offsets)]
        if tuple(self.meta["actor_terms"]) != ACTOR_TERMS or self.meta["actor_term_dims"] != [
            [d] for d in obs_dims
        ]:
            raise ValueError("Incompatible actor observation layout")
        self.obs_dims = obs_dims
        self.base = model.body(self.meta["scan_body"]).id
        self.geom_groups = np.zeros(6, dtype=np.uint8)
        self.geom_groups[self.meta["scan_geom_groups"]] = 1
        self.ray_exclude = self.base if self.meta["scan_excludes_body"] else -1
        self.ray_range = self.meta["scan_max_distance_m"]
        self.ray_scale = self.meta["scan_scale"]
        self.box_min, self.box_max = self.scan_boxes()
        gyro = model.sensor("robot/imu_ang_vel")
        self.gyro = slice(int(gyro.adr[0]), int(gyro.adr[0] + gyro.dim[0]))
        joints = [model.joint(name) for name in self.meta["joint_names"]]
        self.pos_ids = [int(j.qposadr[0]) for j in joints]
        self.vel_ids = [int(j.dofadr[0]) for j in joints]
        self.defaults = np.array(self.meta["default_joint_pos"], dtype=np.float32)
        self.default_velocity = np.array(self.meta["default_joint_vel"], dtype=np.float32)
        self.offset = np.array(self.meta["action_offset"], dtype=np.float32)
        if [model.actuator(i).name for i in range(model.nu)] != self.meta["actuator_names"]:
            raise ValueError("Actuator order changed when compiling the exported scene")
        self.actuator_ids = [model.actuator(name).id for name in self.meta["action_joints"]]
        target_joints = [model.joint(name).id for name in self.meta["action_joints"]]
        if not np.array_equal(model.actuator_trnid[self.actuator_ids, 0], target_joints):
            raise ValueError("Action target joints do not match the actuator transmissions")
        self.control_dt = self.meta["control_timestep_s"]
        self.fall_cos = math.cos(self.meta["fall_angle_rad"])
        contact_sensors = [model.sensor(name) for name in self.meta["illegal_contact_sensors"]]
        self.contact_force_ids = np.concatenate(
            [np.arange(s.adr[0], s.adr[0] + s.dim[0]) for s in contact_sensors]
        ).reshape(-1, 3)
        self.contact_threshold_sq = self.meta["illegal_contact_force_n"] ** 2

    def actor(self, observation: np.ndarray) -> np.ndarray:
        x = (observation - self.mean) / self.scale
        for weight, bias in self.layers[:-1]:
            x = x @ weight.T + bias
            x = np.maximum(x, 0) + np.expm1(np.minimum(x, 0))
        weight, bias = self.layers[-1]
        return x @ weight.T + bias

    def applied_action(self, action: np.ndarray) -> np.ndarray:
        limit = self.meta["clip_actions"]
        return np.clip(action, -limit, limit) if limit is not None else action

    def position_targets(self, applied_action: np.ndarray) -> np.ndarray:
        return applied_action * self.meta["action_scale"] + self.offset

    def scan_boxes(self) -> tuple[np.ndarray, np.ndarray]:
        model = self.model
        visible = (self.geom_groups[model.geom_group] != 0) & (
            model.geom_bodyid != self.ray_exclude
        )
        body_ids = model.geom_bodyid[visible]
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        if (
            not visible.any()
            or np.any(model.geom_type[visible] != int(mujoco.mjtGeom.mjGEOM_BOX))
            or np.any(model.body_weldid[body_ids] != 0)
            or np.any(data.geom_xmat[visible] != np.eye(3).reshape(1, 9))
            or np.any(model.geom_rgba[visible, 3] != 1)
            or np.any(model.geom_matid[visible] != -1)
            or np.any(self.ray_offsets[:, 2] != 0)
        ):
            raise ValueError(
                "Expected a horizontal scan grid over opaque world-fixed axis-aligned boxes"
            )
        center = data.geom_xpos[visible]
        half_size = model.geom_size[visible]
        return center - half_size, center + half_size

    def ray_origins(self, data: mujoco.MjData, rotation: np.ndarray) -> np.ndarray:
        forward = rotation[:2, 0]
        norm = np.linalg.norm(forward)
        if norm < 0.1:
            forward = rotation[[1, 0], 1] * [1.0, -1.0]
            norm = max(1e-6, np.linalg.norm(forward))
        x, y = forward / norm
        yaw = np.array([[x, -y, 0.0], [y, x, 0.0], [0.0, 0.0, 1.0]])
        return self.ray_offsets @ yaw.T + data.xpos[self.base]

    def height_scan(self, origins: np.ndarray) -> np.ndarray:
        # Only boxes intersecting the projected scan footprint can hit a vertical ray.
        xy_min = origins[:, :2].min(axis=0)
        xy_max = origins[:, :2].max(axis=0)
        candidates = np.all(
            (self.box_min[:, :2] <= xy_max) & (self.box_max[:, :2] >= xy_min), axis=1
        )
        low, high = self.box_min[candidates], self.box_max[candidates]
        xy = origins[:, None, :2]
        in_footprint = np.all((xy >= low[:, :2]) & (xy <= high[:, :2]), axis=2)
        top = origins[:, 2, None] - high[:, 2]
        bottom = origins[:, 2, None] - low[:, 2]
        # A ray starting inside a box hits its lower face.
        distance = np.where(top >= 0, top, bottom)
        return (
            np.where(in_footprint & (distance >= 0), distance, np.inf).min(
                axis=1, initial=self.ray_range
            )
            * self.ray_scale
        )

    def native_height_scan(self, data: mujoco.MjData, origins: np.ndarray) -> np.ndarray:
        distances = np.fromiter(
            (
                mujoco.mj_ray(
                    self.model, data, p, DOWN, self.geom_groups, True, self.ray_exclude, None
                )
                for p in origins
            ),
            dtype=np.float64,
            count=len(origins),
        )
        distances[(distances < 0) | (distances > self.ray_range)] = self.ray_range
        return distances * self.ray_scale

    def command(self, command: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        heading = math.atan2(rotation[1, 0], rotation[0, 0])
        error = (self.meta["heading_target_rad"] - heading + math.pi) % (2 * math.pi) - math.pi
        return np.array(
            [
                *command[:2],
                np.clip(error * self.meta["heading_gain"], *self.meta["yaw_rate_limits"]),
            ],
            dtype=np.float32,
        )

    def observe(
        self, data: mujoco.MjData, command: np.ndarray, previous: np.ndarray, step: int
    ) -> np.ndarray:
        rotation = data.xmat[self.base].reshape(3, 3)
        period = np.float32(self.meta["period_s"])
        time_s = np.float32(step) * np.float32(self.control_dt)
        angle = (time_s % period) / period * np.float32(2 * math.pi)
        phase = (np.sin(angle), np.cos(angle)) if np.linalg.norm(command) >= 0.1 else (0.0, 0.0)
        return np.concatenate(
            (
                data.sensordata[self.gyro],
                -rotation[2],
                command,
                phase,
                data.qpos[self.pos_ids] - self.defaults,
                data.qvel[self.vel_ids] - self.default_velocity,
                previous,
                self.height_scan(self.ray_origins(data, rotation)),
            ),
            dtype=np.float32,
        )

    def verify_fixtures(self, bundle: Path) -> dict:
        data = mujoco.MjData(self.model)
        with np.load(bundle / "fixtures.npz", allow_pickle=False) as reference:
            steps, all_pos, all_vel, commands, previous = (
                reference[key] for key in ("step", "qpos", "qvel", "command", "previous_action")
            )
            obs_ref = reference["obs"]
            action_ref = reference["action"]
            ctrl_ref = reference["ctrl"][:, self.actuator_ids]
        observed = []
        ray_error = 0.0
        for step, qpos, qvel, command, prior in zip(steps, all_pos, all_vel, commands, previous):
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            mujoco.mj_forward(self.model, data)
            rotation = data.xmat[self.base].reshape(3, 3)
            if step:
                predicted = self.command(command, rotation)
                np.testing.assert_allclose(predicted, command, atol=1e-5, rtol=0)
            origins = self.ray_origins(data, rotation)
            scan = self.height_scan(origins)
            native = self.native_height_scan(data, origins)
            np.testing.assert_allclose(scan, native, atol=1e-10, rtol=0)
            ray_error = max(ray_error, float(np.max(np.abs(scan - native))))
            observed.append(self.observe(data, command, prior, step))
        observed = np.array(observed)
        actor_replay = self.actor(obs_ref)
        action = self.actor(observed)
        ctrl = self.position_targets(self.applied_action(action))
        obs_errors = np.split(np.abs(observed - obs_ref), np.cumsum(self.obs_dims)[:-1], axis=1)
        np.testing.assert_allclose(observed, obs_ref, atol=2e-4, rtol=0)
        np.testing.assert_allclose(actor_replay, action_ref, atol=2e-5, rtol=0)
        np.testing.assert_allclose(action, action_ref, atol=5e-4, rtol=0)
        np.testing.assert_allclose(ctrl, ctrl_ref, atol=2e-4, rtol=0)
        return {
            "fixtures": len(observed),
            "steps": sorted(set(steps.tolist())),
            "max_height_scan_error_vs_native_ray": ray_error,
            "max_observation_error_by_term": {
                name: float(error.max()) for name, error in zip(ACTOR_TERMS, obs_errors)
            },
            "max_actor_error_same_input": float(np.abs(actor_replay - action_ref).max()),
            "max_action_error": float(np.abs(action - action_ref).max()),
            "max_position_target_error_rad": float(np.abs(ctrl - ctrl_ref).max()),
        }

    def control_step(
        self,
        data: mujoco.MjData,
        command: np.ndarray,
        previous: np.ndarray,
        step: int,
        substeps: int,
        *,
        observer: Callable[[int, mujoco.MjData, np.ndarray], None] | None = None,
        physics_observer: Callable[[mujoco.MjData], None] | None = None,
    ) -> tuple[np.ndarray, str | None]:
        action = self.applied_action(self.actor(self.observe(data, command, previous, step)))
        data.ctrl[self.actuator_ids] = self.position_targets(action)
        if observer is not None:
            observer(step, data, command)
        illegal_contact = False
        for _ in range(substeps):
            mujoco.mj_step(self.model, data)
            if physics_observer is not None:
                physics_observer(data)
            forces = data.sensordata[self.contact_force_ids]
            illegal_contact |= bool(
                np.any(np.sum(forces * forces, axis=1) > self.contact_threshold_sq)
            )
        if any(data.warning[i].number for i in BAD_DYNAMICS):
            return action, "invalid_dynamics"
        if data.xmat[self.base, 8] < self.fall_cos:
            return action, "fell_over"
        if illegal_contact:
            return action, "illegal_contact"
        mujoco.mj_forward(self.model, data)
        return action, None

    def rollout(
        self,
        start: dict,
        steps: int,
        substeps: int,
        *,
        observer: Callable[[int, mujoco.MjData, np.ndarray], None] | None = None,
        physics_observer: Callable[[mujoco.MjData], None] | None = None,
    ) -> dict:
        model = self.model
        data = mujoco.MjData(model)
        data.qpos[:] = start["qpos"]
        data.qvel[:] = start["qvel"]
        mujoco.mj_forward(model, data)
        command = start["command"]
        previous = np.zeros(len(self.actuator_ids), dtype=np.float32)
        origin = start["origin"]
        termination = None
        speed_sum = error_sum = 0.0
        for step in range(steps):
            previous, termination = self.control_step(
                data,
                command,
                previous,
                step,
                substeps,
                observer=observer,
                physics_observer=physics_observer,
            )
            if termination is not None:
                break
            rotation = data.xmat[self.base].reshape(3, 3)
            velocity = (rotation.T @ data.qvel[:3])[:2]
            speed_sum += np.linalg.norm(velocity)
            error_sum += np.linalg.norm(velocity - command[:2])
            command = self.command(command, rotation)
        offset = data.xpos[self.base] - origin
        return {
            "survived": termination is None,
            "termination": termination,
            "control_steps": step + 1,
            "distance_m": float(np.linalg.norm(offset[:2])),
            "dz_m": float(offset[2]),
            "command_mps": float(np.linalg.norm(command[:2])),
            "speed_mps": float(speed_sum / steps) if termination is None else None,
            "tracking_error_mps": float(error_sum / steps) if termination is None else None,
        }


def evaluation_substeps(cpu: CpuStairs, timestep: float, speculative: bool) -> int:
    substeps = round(cpu.control_dt / timestep)
    if substeps < 1 or not math.isclose(substeps * timestep, cpu.control_dt, abs_tol=1e-10):
        raise ValueError("Physics timestep must divide the training policy's control timestep")
    cpu.model.opt.timestep = timestep
    if speculative:
        cpu.model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SPECULATIVE)
    return substeps


def evaluate_level(
    level: int,
    *,
    bundle: Path,
    timestep: float,
    speculative: bool,
    steps: int,
    terrain_filter: list[str] | None,
) -> tuple[int, list[dict]]:
    cpu = CpuStairs(bundle)
    substeps = evaluation_substeps(cpu, timestep, speculative)
    level_index = cpu.meta["levels"].index(level)
    with np.load(bundle / "starts.npz", allow_pickle=False) as starts:
        starts = {key: starts[key][level_index] for key in starts.files}
    rows = []
    for world, column in enumerate(cpu.meta["world_columns"]):
        terrain = cpu.meta["terrain_columns"][column]
        if terrain_filter is None or terrain in terrain_filter:
            row = cpu.rollout({key: value[world] for key, value in starts.items()}, steps, substeps)
            rows.append({"world": world, "terrain": terrain, "column": column, **row})
    return level, rows


def summarize(rows: list[dict], promote_m: float, demote_m: float) -> dict:
    alive = [row for row in rows if row["survived"]]

    def median(key: str, digits: int) -> float | None:
        return round(median_low(row[key] for row in alive), digits) if alive else None

    return {
        "envs": len(rows),
        "survived": round(len(alive) / len(rows), 3),
        "distance_median_m": median("distance_m", 2),
        "promote_share": round(sum(r["distance_m"] > promote_m for r in alive) / len(alive), 3)
        if alive
        else None,
        "demote_share": round(
            sum(r["distance_m"] < demote_m and r["command_mps"] > 0.1 for r in alive) / len(alive),
            3,
        )
        if alive
        else None,
        "dz_median_m": median("dz_m", 3),
        "command_median_mps": median("command_mps", 2),
        "speed_median_mps": median("speed_mps", 2),
        "tracking_error_median_mps": median("tracking_error_mps", 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--levels", type=int, nargs="+")
    parser.add_argument("--terrain", nargs="+")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--timestep", type=float)
    parser.add_argument("--speculative", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    cpu = CpuStairs(args.bundle)
    meta = cpu.meta
    levels = args.levels if args.levels is not None else meta["levels"]
    terrains = tuple(dict.fromkeys(meta["terrain_columns"]))
    steps = args.steps if args.steps is not None else meta["steps"]
    timestep = args.timestep if args.timestep is not None else meta["physics_timestep_s"]
    if not set(levels) <= set(meta["levels"]) or not set(args.terrain or ()) <= set(terrains):
        parser.error("Requested levels and terrain types must exist in the bundle")
    if not 0 < steps <= meta["steps"] or args.workers < 1 or not 0 < timestep <= cpu.control_dt:
        parser.error("Check steps, workers and timestep against the exported evaluation settings")
    fixtures = cpu.verify_fixtures(args.bundle)
    print(json.dumps(fixtures), flush=True)
    result = {
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "platform": sys.platform,
        "bundle": meta,
        "bundle_sha256": {
            name: hashlib.sha256((args.bundle / name).read_bytes()).hexdigest()
            for name in ("scene.xml", "actor.npz", "starts.npz", "fixtures.npz")
        },
        "parity": fixtures,
        "steps": steps,
        "physics_timestep_s": timestep,
        "speculative": args.speculative,
        "levels": {},
        "trials": {},
    }
    if not args.verify_only:
        evaluation_substeps(cpu, timestep, args.speculative)
        evaluate = partial(
            evaluate_level,
            bundle=args.bundle,
            timestep=timestep,
            speculative=args.speculative,
            steps=steps,
            terrain_filter=args.terrain,
        )
        if args.workers == 1:
            outcomes = list(map(evaluate, levels))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                outcomes = list(pool.map(evaluate, levels))
        for level, rows in outcomes:
            result["levels"][str(level)] = stats = {
                terrain: summarize(
                    [row for row in rows if row["terrain"] == terrain],
                    meta["promote_m"],
                    meta["demote_m"],
                )
                for terrain in terrains
                if args.terrain is None or terrain in args.terrain
            }
            result["trials"][str(level)] = rows
            print(level, json.dumps(stats), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
