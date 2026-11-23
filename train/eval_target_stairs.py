"""Replay a frozen RBQ policy up and down one T0 ship stair in a continuous simulation."""

import argparse
import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
from eval_stairs_cpu import MIN_SUPPORT_FORCE_N, CpuStairs, evaluation_substeps, supporting_contacts

from go2_wall.stairs import add_ship_stair, going_of, stair_parameters

FOOT_NAMES = ("FL", "FR", "RL", "RR")
START_X_M = -1.1
FLOOR_MU = 0.6
RETURN_TOLERANCE_M = 0.15
TURN_TOLERANCE_RAD = math.radians(10)
LATERAL_SPEED_LIMIT_MPS = 0.15
RETURN_HOLD_S = 0.5
RETURN_UPRIGHT_COS = math.cos(math.radians(10))
RETURN_MAX_LINEAR_MPS = 0.05
RETURN_MAX_ANGULAR_RADPS = 0.15


def mission_model(bundle: Path, slope: int, rise: float, treads: int) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(bundle / "scene.xml"))
    terrain = spec.body("terrain")
    for geom in list(terrain.geoms):
        spec.delete(geom)
    terrain.add_geom(
        name="approach_floor",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[8.0, 2.0, 0.25],
        pos=[0.0, 0.0, -0.25],
        friction=[FLOOR_MU, 0.02, 0.01],
        priority=2,
    )
    add_ship_stair(terrain, slope, rise, treads, FLOOR_MU)
    # The actor scans the stepping surfaces. Keep stringers in physics as group-3 geometry.
    for name in ("stringerL", "stringerR"):
        spec.geom(name).group = 3
    return spec.compile()


def stable_return(data: mujoco.MjData, base: int, supported: set[int], feet: set[int]) -> bool:
    return (
        supported == feet
        and np.linalg.norm(data.qpos[:2] - [START_X_M, 0.0]) <= RETURN_TOLERANCE_M
        and data.xmat[base, 8] >= RETURN_UPRIGHT_COS
        and np.linalg.norm(data.qvel[:3]) <= RETURN_MAX_LINEAR_MPS
        and np.linalg.norm(data.qvel[3:6]) <= RETURN_MAX_ANGULAR_RADPS
    )


def inside_route(
    points: np.ndarray,
    stair_half_width: float,
    landing_xy: np.ndarray,
    landing_half_size: np.ndarray,
    *,
    turning: bool = False,
) -> bool:
    on_landing = np.all(np.abs(points - landing_xy) <= landing_half_size, axis=1)
    if turning:
        return bool(on_landing.all())
    x, y = points.T
    landing_start = landing_xy[0] - landing_half_size[0]
    on_stair = (x >= 0) & (x < landing_start) & (np.abs(y) <= stair_half_width)
    return bool(np.all((x < 0) | on_stair | on_landing))


def route_command(
    cpu: CpuStairs, data: mujoco.MjData, phase: str, speed: float
) -> tuple[np.ndarray, float]:
    if phase == "settle":
        return np.zeros(3, dtype=np.float32), 0.0
    rotation = data.xmat[cpu.base].reshape(3, 3)
    heading = math.atan2(rotation[1, 0], rotation[0, 0])
    target = 0.0 if phase == "ascend" else math.pi
    error = (target - heading + math.pi) % (2 * math.pi) - math.pi
    yaw_rate = np.clip(error * cpu.meta["heading_gain"], *cpu.meta["yaw_rate_limits"])
    if phase == "turn":
        # A smaller angular command is treated as standing by the policy's phase input.
        if abs(error) > TURN_TOLERANCE_RAD and abs(yaw_rate) < 0.1:
            yaw_rate = math.copysign(0.1, error)
        return np.array([0.0, 0.0, yaw_rate], dtype=np.float32), error
    velocity_x = speed if phase == "ascend" else -speed
    velocity_y = np.clip(-data.xpos[cpu.base, 1], -LATERAL_SPEED_LIMIT_MPS, LATERAL_SPEED_LIMIT_MPS)
    c, s = math.cos(heading), math.sin(heading)
    command = np.array(
        [c * velocity_x + s * velocity_y, -s * velocity_x + c * velocity_y, yaw_rate]
    )
    return command.astype(np.float32), error


def run_route(
    cpu: CpuStairs,
    speed: float,
    treads: int,
    start_height: float,
    *,
    command_for: Callable[[CpuStairs, mujoco.MjData, str, float], tuple[np.ndarray, float]] = (
        route_command
    ),
) -> dict:
    model = cpu.model
    substeps = evaluation_substeps(cpu, cpu.meta["physics_timestep_s"], False)
    floor = model.geom("approach_floor").id
    landing = model.geom("platform").id
    tread_numbers = {model.geom(f"tread{k}").id: k for k in range(1, treads + 1)}
    tread_sections = {k: 3 * (k - 1) // treads for k in range(1, treads + 1)}
    feet = {model.geom(f"robot/{leg}_foot_collision").id for leg in FOOT_NAMES}
    sites = [model.site(f"robot/{leg}").id for leg in FOOT_NAMES]
    data = mujoco.MjData(model)
    data.qpos[:7] = [START_X_M, 0, start_height, 1, 0, 0, 0]
    data.qpos[cpu.pos_ids] = cpu.defaults
    mujoco.mj_forward(model, data)
    landing_xy = data.geom_xpos[landing, :2].copy()
    landing_half_size = model.geom_size[landing, :2]
    landing_top = data.geom_xpos[landing, 2] + model.geom_size[landing, 2]
    first_tread = model.geom("tread1").id
    stair_half_width = model.geom_size[first_tread, 1]
    first_tread_end = data.geom_xpos[first_tread, 0] + model.geom_size[first_tread, 0]
    walking_surfaces = {floor, landing, *tread_numbers}
    previous = np.zeros(len(cpu.actuator_ids), dtype=np.float32)
    peak_ratio = np.zeros(model.nu)
    low_limits, high_limits = model.actuator_forcerange.T
    if not (
        np.all(model.actuator_forcelimited)
        and np.all(low_limits < 0)
        and np.all(high_limits > 0)
        and np.all(model.actuator_gear[:, 0] == 1)
    ):
        raise ValueError("Expected bounded, unit-ratio joint actuators for torque measurement")
    nonfoot = {int(model.sensor(name).objid[0]) for name in cpu.meta["illegal_contact_sensors"]}
    terrain_body = model.body("terrain").id
    first_illegal_contacts = []
    phase = "ascend"
    phase_started = 0
    events = {"ascend_s": 0.0}
    tread_contacts = {"ascend": set(), "descend": set()}
    support: dict[int, set[int]] = {}
    route_violation = None
    stable_steps = 0
    required_stable_steps = math.ceil(RETURN_HOLD_S / model.opt.timestep)

    def measure(state: mujoco.MjData) -> None:
        nonlocal stable_steps, route_violation
        torque = state.actuator_force
        ratio = torque / np.where(torque < 0, low_limits, high_limits)
        np.maximum(peak_ratio, ratio, out=peak_ratio)
        support.clear()
        for foot, surface, point in supporting_contacts(model, state, feet):
            support.setdefault(surface, set()).add(foot)
            if (
                route_violation is None
                and phase in tread_contacts
                and (
                    surface not in walking_surfaces
                    or (surface == floor and point[0] > first_tread_end)
                )
            ):
                route_violation = {
                    "reason": "off_route_support",
                    "time_s": state.time,
                    "phase": phase,
                    "foot": model.geom(foot).name,
                    "surface": model.geom(surface).name,
                    "contact_xyz_m": point.tolist(),
                }
        if phase in tread_contacts or phase == "turn":
            points = np.vstack((state.xpos[cpu.base, :2], state.site_xpos[sites, :2]))
            turning = phase == "turn"
            if route_violation is None and (
                not inside_route(
                    points, stair_half_width, landing_xy, landing_half_size, turning=turning
                )
                or (turning and any(g != landing for g in support))
            ):
                route_violation = {
                    "reason": "left_landing_during_turn" if turning else "left_stair_lane",
                    "time_s": state.time,
                    "phase": phase,
                    "body_and_feet_xy_m": points.tolist(),
                }
        if phase in tread_contacts:
            tread_contacts[phase].update(tread_numbers[g] for g in support if g in tread_numbers)
        elif phase == "settle":
            stable = stable_return(state, cpu.base, support.get(floor, set()), feet)
            stable_steps = stable_steps + 1 if stable else 0

        forces = state.sensordata[cpu.contact_force_ids]
        if first_illegal_contacts or not np.any(
            np.sum(forces * forces, axis=1) > cpu.contact_threshold_sq
        ):
            return
        wrench = np.zeros(6)
        for contact_id, contact in enumerate(state.contact):
            if contact.efc_address < 0:
                continue
            robot, surface = map(int, contact.geom)
            if surface in nonfoot:
                robot, surface = surface, robot
            if robot not in nonfoot or model.geom_bodyid[surface] != terrain_body:
                continue
            mujoco.mj_contactForce(model, state, contact_id, wrench)
            force = float(np.linalg.norm(wrench[:3]))
            if force**2 > cpu.contact_threshold_sq:
                first_illegal_contacts.append(
                    {
                        "time_s": state.time,
                        "robot_geom": model.geom(robot).name,
                        "terrain_geom": model.geom(surface).name,
                        "force_n": force,
                        "contact_xyz_m": contact.pos.tolist(),
                        "body_xyz_m": state.xpos[cpu.base].tolist(),
                        "body_pitch_up_deg": math.degrees(
                            math.asin(np.clip(state.xmat[cpu.base, 6], -1.0, 1.0))
                        ),
                    }
                )

    def feet_over_landing() -> bool:
        foot_xyz = data.site_xpos[sites]
        return (
            len(support.get(landing, ())) >= 2
            and np.all(np.abs(foot_xyz[:, :2] - landing_xy) <= landing_half_size)
            and np.all(foot_xyz[:, 2] >= landing_top)
        )

    max_height = start_height
    max_forward = START_X_M
    termination = None
    for step in range(4 * cpu.meta["steps"]):
        if phase == "ascend" and feet_over_landing():
            phase = "turn"
            phase_started = step
            events["on_top_s"] = step * cpu.control_dt
        elif phase == "descend" and len(support.get(floor, ())) >= 2:
            at_start = (
                np.linalg.norm(data.xpos[cpu.base, :2] - [START_X_M, 0]) <= RETURN_TOLERANCE_M
            )
            if at_start:
                phase = "settle"
                phase_started = step
                events["reached_start_s"] = step * cpu.control_dt
        command, heading_error = command_for(cpu, data, phase, speed)
        if phase == "turn" and abs(heading_error) <= TURN_TOLERANCE_RAD and feet_over_landing():
            phase = "descend"
            phase_started = step
            events["turned_s"] = step * cpu.control_dt
            command, _ = command_for(cpu, data, phase, speed)
        if step - phase_started >= cpu.meta["steps"]:
            termination = "phase_timeout"
            break
        previous, termination = cpu.control_step(
            data, command, previous, step, substeps, physics_observer=measure
        )
        max_height = max(max_height, float(data.qpos[2]))
        max_forward = max(max_forward, float(data.qpos[0]))
        if termination is not None:
            break
        if phase == "settle" and stable_steps >= required_stable_steps:
            events["returned_s"] = data.time
            break
    ratio = float(peak_ratio.max())
    if termination is None:
        if route_violation is not None:
            termination = route_violation["reason"]
        elif "returned_s" not in events:
            termination = "route_timeout"
        elif any(
            {tread_sections[k] for k in recorded} != set(tread_sections.values())
            for recorded in tread_contacts.values()
        ):
            termination = "missing_tread_support"
        elif ratio > 1.0 + 1e-7:
            termination = "torque_limit"
    return {
        "speed_mps": speed,
        "passed_round_trip": termination is None,
        "last_phase": phase,
        "termination": termination,
        "events": events,
        "ascent_treads": sorted(tread_contacts["ascend"]),
        "descent_treads": sorted(tread_contacts["descend"]),
        "maximum_body_z_m": max_height,
        "maximum_forward_x_m": max_forward,
        "final_body_xyz_m": data.qpos[:3].tolist(),
        "peak_torque_ratio": ratio,
        "first_illegal_contacts": first_illegal_contacts,
        "first_route_violation": route_violation,
        "stable_return_s": stable_steps * model.opt.timestep,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--slope", type=int, default=50, choices=stair_parameters()["slopes_deg"])
    parser.add_argument("--rise", type=float, default=0.19)
    parser.add_argument("--treads", type=int, default=10)
    parser.add_argument("--speeds", type=float, nargs="+", default=(0.35, 0.5, 0.6))
    args = parser.parse_args()
    if not (args.rise > 0 and args.treads > 0 and all(0 < speed <= 0.8 for speed in args.speeds)):
        parser.error("Use a positive rise and tread count, and speeds from 0 to 0.8 m/s")
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    model = mission_model(args.bundle, args.slope, args.rise, args.treads)
    cpu = CpuStairs(args.bundle, model=model)
    with np.load(args.bundle / "starts.npz", allow_pickle=False) as starts:
        start_height = float(starts["qpos"][0, 0, 2] - starts["origin"][0, 0, 2])
    result = {
        "mujoco": mujoco.__version__,
        "checkpoint": cpu.meta["checkpoint"],
        "checkpoint_sha256": cpu.meta["checkpoint_sha256"],
        "actor_sha256": hashlib.sha256((args.bundle / "actor.npz").read_bytes()).hexdigest(),
        "slope_deg": args.slope,
        "rise_m": args.rise,
        "going_m": going_of(args.slope, args.rise),
        "treads": args.treads,
        "control_timestep_s": cpu.control_dt,
        "physics_timestep_s": cpu.meta["physics_timestep_s"],
        "phase_timeout_s": cpu.meta["steps"] * cpu.control_dt,
        "route_criteria": {
            "tread_sections_each_direction": min(3, args.treads),
            "minimum_foot_support_n": MIN_SUPPORT_FORCE_N,
            "return_tolerance_m": RETURN_TOLERANCE_M,
            "return_hold_s": RETURN_HOLD_S,
            "return_max_linear_mps": RETURN_MAX_LINEAR_MPS,
            "return_max_angular_radps": RETURN_MAX_ANGULAR_RADPS,
            "return_max_tilt_from_vertical_deg": math.degrees(math.acos(RETURN_UPRIGHT_COS)),
        },
        "start_xyz_m": [START_X_M, 0.0, start_height],
        "stair_geometry": stair_parameters(),
        "trials": [],
    }
    for speed in args.speeds:
        trial = run_route(cpu, speed, args.treads, start_height)
        result["trials"].append(trial)
        print(json.dumps(trial, allow_nan=False), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
