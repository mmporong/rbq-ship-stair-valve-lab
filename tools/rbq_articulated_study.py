"""Paired CPU rollouts with a free base, articulated gripper and passive valve."""

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
from pathlib import Path

import mujoco
import numpy as np
from rbq_articulated_control import ArticulatedActor, ReferenceControl, digest, prepare_reference
from render_stair_mission import build_model
from triage_rbq_host_ab import canonical_sha256, validate_cases

from go2_wall.robot import LEGS, ROOT
from go2_wall.stair_loads import SURFACE_NAMES

ACTOR_MODES = ("no_intervention", "phase_25", "edge_slow_rule")
MODES = (*ACTOR_MODES, "joint_reference_pd", "contact_gated_reference", "teacher_open_loop")


class Contacts:
    """Upward loads from actual constraint wrenches and non-foot terrain collisions."""

    def __init__(self, model):
        self.model = model
        self.feet = {model.geom(name).id: i for i, name in enumerate(LEGS)}
        self.surfaces = {model.geom(name).id: i for i, name in enumerate(SURFACE_NAMES)}
        self.feet_load = np.zeros((4, len(self.surfaces)))
        self.wrench = np.empty(6)
        self.collision_n = 0.0

    def measure(self, data):
        loads = self.feet_load
        loads.fill(0)
        self.collision_n = 0.0
        for i, contact in enumerate(data.contact):
            first, second = int(contact.geom1), int(contact.geom2)
            if first in self.surfaces:
                foot, surface, sign = second, self.surfaces[first], 1.0
            elif second in self.surfaces:
                foot, surface, sign = first, self.surfaces[second], -1.0
            else:
                continue
            mujoco.mj_contactForce(self.model, data, i, self.wrench)
            if foot in self.feet:
                fz = contact.frame.reshape(3, 3)[:, 2] @ self.wrench[:3]
                loads[self.feet[foot], surface] += sign * fz
            else:
                self.collision_n = max(self.collision_n, float(np.linalg.norm(self.wrench[:3])))
        return loads


@cache
def prepared(scene_path, path):
    with np.load(path, allow_pickle=False) as source:
        reference = dict(source)
    if str(reference["scene_sha256"]) != digest(scene_path):
        raise ValueError("Reference and articulated scene differ")
    model, _ = build_model(json.loads(Path(scene_path).read_bytes()))
    return model, reference


def run_case(scene, reference_path, bundle, case, mode, state_dir=None):
    model, ref = prepared(scene, reference_path)
    dt, stride = float(ref["dt"]), int(ref["stride"])
    data = mujoco.MjData(model)
    data.qpos[:] = ref["qpos"][0]
    yaw = math.radians(case["yaw_deg"])
    data.qpos[1] += case["lateral_m"]
    quaternion = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    mujoco.mju_mulQuat(data.qpos[3:7], quaternion, ref["qpos"][0, 3:7])
    leg_positions = model.jnt_qposadr[model.actuator_trnid[:12, 0]]
    data.qpos[leg_positions] += np.random.default_rng(case["joint_seed"]).uniform(
        -case["joint_noise_rad"], case["joint_noise_rad"], 12
    )
    mujoco.mj_forward(model, data)
    controller = ReferenceControl(model, ref, mode, case["lateral_m"], case["yaw_deg"])
    actor = ArticulatedActor(model, bundle, mode) if mode in ACTOR_MODES else None
    ranges = model.actuator_ctrlrange
    contact = Contacts(model)
    loads = contact.measure(data)
    total = len(ref["command"]) * stride
    record = {}
    if state_dir:
        record = {
            field: np.empty((total, width), dtype=np.float64)
            for field, width in (
                ("qpos", model.nq),
                ("qvel", model.nv),
                ("ctrl", model.nu),
                ("requested", model.nu),
            )
        }
    active_steps = longest = at_original_horizon = highest = 0
    highest_at_original_horizon = 0
    stop = None
    collision = False
    max_collision_n = 0.0
    maximum_angle = 0.0
    target = np.asarray([5, 5, 3, 3])
    legs = np.arange(4)
    base = model.body("base").id
    hinge = model.joint("valve_hinge").qposadr[0]
    for t in range(total):
        if t % stride == 0:
            k = t // stride
            requested = controller.torque(k, data, loads.max(axis=1) >= 1.0)
            if actor:
                requested = actor.apply(k, data, requested)
            if not np.isfinite(requested).all():
                raise ValueError("Nonfinite torque request")
            data.ctrl[:] = np.clip(requested, ranges[:, 0], ranges[:, 1])
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        loads = contact.measure(data)
        active_steps = active_steps + 1 if (loads[legs, target] >= 1.0).all() else 0
        longest = max(longest, active_steps)
        highest = max(highest, int(np.max(np.where(loads[:, 1:11] >= 1.0, np.arange(1, 11), 0))))
        if t + 1 <= 19960:
            at_original_horizon = longest
            highest_at_original_horizon = highest
        max_collision_n = max(max_collision_n, contact.collision_n)
        collision |= contact.collision_n > 10.0
        maximum_angle = max(maximum_angle, float(data.qpos[hinge]))
        if record:
            for name in ("qpos", "qvel", "ctrl"):
                record[name][t] = getattr(data, name)
            record["requested"][t] = requested
        if collision or data.xmat[base, 8] < math.cos(math.radians(70)) or data.qpos[2] < 0.12:
            stop = "terrain_body_contact" if collision else "fall"
            break
        if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
            raise ValueError(f"Dynamics warning in {case['case_id']} at step {t + 1}")
    steps = t + 1
    state_hash = None
    if record:
        dest = Path(state_dir) / f"{case['case_id']}--{mode}.npz"
        # Uncompressed locally so replay workers can stream states without recomputing dynamics.
        with dest.open("xb") as f:
            np.savez(
                f,
                **{name: record[name][:steps] for name in ("qpos", "qvel", "ctrl")},
                requested_robot_nm=record["requested"][:steps, :18],
                requested_finger_force_n=record["requested"][:steps, 18],
                time_s=(np.arange(steps) + 1) * dt,
                scene_sha256=ref["scene_sha256"],
                mode=mode,
                case_id=case["case_id"],
            )
        state_hash = digest(dest)
    return {
        "case_id": case["case_id"],
        "candidate_id": mode,
        "termination": stop or "horizon",
        "simulated_s": round(steps * dt, 6),
        "body_collision": bool(collision),
        "max_nonfoot_terrain_force_n": max_collision_n,
        "highest_supported_tread": highest,
        "maximum_work_stance_s": longest * dt,
        "work_stance_reached": bool(longest >= 500),
        "original_horizon_work_stance": bool(at_original_horizon >= 500),
        "original_horizon_highest_tread": highest_at_original_horizon,
        "maximum_valve_angle_deg": math.degrees(maximum_angle),
        "final_valve_angle_deg": math.degrees(data.qpos[hinge]),
        "state_sha256": state_hash,
        "ppo_leg_policy_evaluated": bool(actor),
        "leg_source": "PPO" if actor else "expert demonstration",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    prepare = sub.add_parser("prepare")
    for name in ("source", "walk", "destination"):
        prepare.add_argument(name, type=Path)
    run = sub.add_parser("run")
    for name in ("scene", "reference", "cases", "out"):
        run.add_argument(name, type=Path)
    run.add_argument("--bundle", type=Path, default=ROOT / ".omx/t1-v8-cpu-9100")
    run.add_argument("--modes", nargs="+", choices=MODES, required=True)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--state-dir", type=Path)
    run.add_argument("--expected-sha256")
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare_reference(args.source, args.walk, args.destination)
        return
    cases = validate_cases(json.loads(args.cases.read_bytes()))
    if args.expected_sha256 and canonical_sha256(cases) != args.expected_sha256:
        parser.error("case commitment mismatch")
    if args.out.exists() or len(set(args.modes)) != len(args.modes):
        parser.error("result already exists or duplicate mode")
    if args.state_dir:
        args.state_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run_case, args.scene, args.reference, args.bundle, c, mode, args.state_dir)
            for c in cases
            for mode in args.modes
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                row["case_id"],
                row["candidate_id"],
                row["termination"],
                row["simulated_s"],
                row["maximum_work_stance_s"],
                round(row["final_valve_angle_deg"], 3),
                flush=True,
            )
    report = {
        "case_set_sha256": canonical_sha256(cases),
        "reference_sha256": digest(args.reference),
        "scene_sha256": digest(args.scene),
        "control_source_sha256": digest(Path(__file__).with_name("rbq_articulated_control.py")),
        "simulation_source_sha256": digest(Path(__file__)),
        "model": "articulated 50 degree, passive 10 Nm valve",
        "clock_s": {"physics": 0.001, "torque": 0.005, "ppo": 0.02, "horizon": 127.75},
        "trials": sorted(rows, key=lambda r: (r["case_id"], r["candidate_id"])),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
