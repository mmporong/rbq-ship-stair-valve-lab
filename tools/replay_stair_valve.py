"""Find a fixed stair stance and replay a valve turn against a physical hinge load."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from replay_stair_transfer import LOAD_THRESHOLD_N, expected_surfaces, support_loads

from go2_wall import stair_valve, stairs, transition, valve
from go2_wall.lever import HOLD_LIMITS
from go2_wall.pose_search import clear_of_collisions
from go2_wall.robot import ARM_JOINTS, LEGS, load_parameters
from go2_wall.statics import joint_names, solve_static, torque_limits

HOLD_S = 5.0
INITIAL_HOLD_S = 1.5
END_SETTLE_S = 5.75
PHYSICS_STEP_S = 0.005
TARGET_DEG = 90.0
ARM_OVERTRAVEL_DEG = 1.4
ANGLE_TOLERANCE_DEG = 0.25
MOTION_MIN_SPEED_RADPS = 0.02


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


def anchor(plan: dict, geometry: dict) -> tuple:
    feet = {}
    rise, radius = geometry["rise_m"], geometry["foot_radius_m"]
    for index, move in enumerate(plan["moves"]):
        feet[move["leg"]] = np.asarray(move["target_xyz_m"])
        if len(feet) != len(LEGS):
            continue
        treads = tuple(round((feet[leg][2] - radius) / rise) for leg in LEGS)
        if treads == (5, 5, 3, 3):
            path = move["path"]
            base = path.get("swing_bases", [path["swing_base"]])[-1]
            return base, {leg: (feet[leg], "floor") for leg in LEGS}, index
    raise ValueError("The stair route has no stance on treads 5, 5, 3, 3")


def arc_point(centre, start_deg, sense, length, angle_deg):
    angle = np.radians(start_deg + sense * angle_deg)
    rad = np.array([np.cos(angle), 0.0, np.sin(angle)])
    direction = sense * np.array([-rad[2], 0.0, rad[0]])
    return centre + length * rad, direction


def find(geometry: dict, stair_route: dict, max_candidates: int) -> dict:
    stairs.ARM = geometry["arm"]
    stairs.use_stow()
    slope, rise = geometry["slope_deg"], geometry["rise_m"]
    model, data = stairs.build_stairs(slope, rise)
    limits = torque_limits(geometry["arm"])
    joints = [model.joint(name).qposadr[0] for name in ARM_JOINTS]
    body, stance, index = anchor(stair_route, geometry)
    feet = {leg: value[0] for leg, value in stance.items()}
    contacts = transition.contacts_for(stance)
    front = np.mean([feet["FL"], feet["FR"]], axis=0)
    settings = load_parameters()
    length = settings["valve_handle_length_m"]["value"]
    resistance = settings["ball_valve_operating_torque_nm"]["value"]["half_inch_reduced_bore"]
    force_n = resistance / length
    base_offsets = itertools.product((0.0, -0.045, 0.045), (0.0, -0.03), (0.0, -0.04, 0.04))
    cases = (
        (0.2, 0.6, 270, -1),
        (0.3, 0.6, 270, -1),
        (0.2, 0.8, 270, -1),
        (0.3, 0.6, 0, -1),
        (0.2, 0.6, 180, -1),
        (0.3, 0.8, 270, -1),
    )
    candidates = []
    preload = dict.fromkeys(LEGS, 10.0)
    sampling = np.linspace(0.0, TARGET_DEG + ARM_OVERTRAVEL_DEG, 37)
    for offset in base_offsets:
        base = [*(np.asarray(body[:3]) + offset), *body[3:]]
        if not transition.place(model, data, base, feet, geometry["arm_stow_rad"], 0.01):
            continue
        stationary = data.qpos.copy()
        for dx, dz, start, sense in cases:
            centre = front + [dx, 0, dz]
            for seed in valve.ARM_SEEDS:
                q = seed
                pose_rows, ratios = [], []
                for angle in sampling:
                    point, direction = arc_point(centre, start, sense, length, angle)
                    data.qpos[:] = stationary
                    if valve.arm_ik(model, data, point, q) > 0.001:
                        break
                    if not clear_of_collisions(model, data):
                        break
                    solution = solve_static(
                        model,
                        data,
                        contacts,
                        limits,
                        tip_force=-force_n * direction,
                        min_normal=preload,
                    )
                    if solution is None or solution.ratio > 1.0:
                        break
                    q = data.qpos[joints].copy()
                    pose_rows.append(q.tolist())
                    ratios.append(solution.ratio)
                else:
                    candidates.append(
                        {
                            "base_pose": base,
                            "feet_xyz_m": {k: v.tolist() for k, v in feet.items()},
                            "centre_m": centre.tolist(),
                            "start_deg": start,
                            "sense": sense,
                            "handle_length_m": length,
                            "resistance_nm": resistance,
                            "sample_angles_deg": sampling.tolist(),
                            "arm_joint_targets_rad": pose_rows,
                            "peak_static_ratio": max(ratios),
                        }
                    )
                    print(len(candidates), centre.tolist(), start, sense, max(ratios), flush=True)
                    break
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break
    return {
        "arm": geometry["arm"],
        "slope_deg": slope,
        "rise_m": rise,
        "foot_radius_m": geometry["foot_radius_m"],
        "anchor_move_index": index,
        "candidates": candidates,
    }


def make_track(
    study: dict, saved: dict, motion_s: float, settle_s: float, hold_torque_nm: float = 0.0
) -> dict:
    stairs.ARM = study["arm"]
    stairs.use_stow()
    model, data = stairs.build_stairs(study["slope_deg"], study["rise_m"])
    limits = torque_limits(study["arm"])
    feet = {leg: np.asarray(saved["feet_xyz_m"][leg]) for leg in LEGS}
    stance = {leg: (feet[leg], "floor") for leg in LEGS}
    contacts = transition.contacts_for(stance)
    arm_addr = [model.joint(name).qposadr[0] for name in ARM_JOINTS]
    joint_addr = [model.joint(name).qposadr[0] for name in joint_names()]
    samples = np.asarray(saved["arm_joint_targets_rad"])
    first_arm = dict(zip(ARM_JOINTS, samples[0], strict=True))
    if not transition.place(model, data, saved["base_pose"], feet, first_arm):
        raise ValueError("Saved valve stance has no initial pose")
    end = INITIAL_HOLD_S + motion_s
    duration = end + settle_s + HOLD_S
    times = np.arange(round(duration / transition.SAMPLE_S) + 1) * transition.SAMPLE_S
    base_q = data.qpos.copy()
    angle = saved["sample_angles_deg"]
    centre = np.asarray(saved["centre_m"])
    length = saved["handle_length_m"]
    force_n = saved["resistance_nm"] / length
    _, hold_tangent = arc_point(centre, saved["start_deg"], saved["sense"], length, TARGET_DEG)
    hold_ramp_start = end + max(0.0, settle_s - 1.5)
    qpos, targets, torques, external = [], [], [], []
    for time in times:
        progress = transition.smooth((time - INITIAL_HOLD_S) / motion_s)
        desired = angle[-1] * progress
        _, tangent = arc_point(
            centre, saved["start_deg"], saved["sense"], saved["handle_length_m"], desired
        )
        ramp = transition.smooth((time - INITIAL_HOLD_S) / 0.45)
        release = 1.0 - transition.smooth((time - end) / 0.45)
        hold_ramp = transition.smooth((time - hold_ramp_start) / 0.5)
        reaction = -force_n * ramp * release * tangent
        reaction -= hold_ramp * hold_tangent * hold_torque_nm / length
        data.qpos[:] = base_q
        data.qpos[arm_addr] = [np.interp(desired, angle, column) for column in samples.T]
        solution = solve_static(model, data, contacts, limits, tip_force=reaction, smooth=True)
        if solution is None:
            raise ValueError(f"Loaded valve pose is infeasible at {time:.2f}s")
        qpos.append(data.qpos.copy())
        targets.append(data.qpos[joint_addr].copy())
        torques.append(solution.torque)
        external.append(reaction)
    return {
        "time": times,
        "qpos": np.asarray(qpos),
        "target": np.asarray(targets),
        "torque": np.asarray(torques),
        "tip": np.asarray(external),
        "stance": np.ones((len(times), 4), dtype=bool),
        "wall": np.zeros((len(times), 4), dtype=bool),
        "cap": np.full((len(times), 4), np.inf),
        "feet": np.tile(np.asarray([feet[leg] for leg in LEGS]), (len(times), 1, 1)),
    }


def evaluate(
    log: dict, metrics: dict, motion_s: float, settle_s: float, resistance_nm: float
) -> dict:
    t = log["time"]
    angle = np.degrees(log["extra"][:, 0])
    speed = log["extra"][:, 1]
    active = t >= INITIAL_HOLD_S
    hold_start = INITIAL_HOLD_S + motion_s + settle_s
    holding = t >= hold_start
    moving = active & (angle > 5.0) & (angle < 85.0) & (speed > MOTION_MIN_SPEED_RADPS)
    start = np.flatnonzero(active)[0]
    loads = log["force"][active]
    upward = loads[..., 2]
    friction = (
        np.linalg.norm(loads[..., :2], axis=2)
        / np.maximum(upward, LOAD_THRESHOLD_N)
        / stair_valve.FLOOR_MU
    )
    sustained_torque = -log["extra"][moving, 2]
    hold_angle = float(angle[holding].min()) if np.any(holding) else 0.0
    hold_max = float(angle[holding].max()) if np.any(holding) else 0.0
    per_tread_load = log["extra"][active, len(stair_valve.OBSERVATION_FIELDS) :]
    body = np.linalg.norm(log["base"][active, :3] - log["base"][start, :3], axis=1)
    slip = np.linalg.norm(log["feet"][active] - log["feet"][start], axis=2)
    dt = t[1] - t[0]
    result = {
        "start_deg": float(angle[start]),
        "max_angle_deg": float(angle.max()),
        "final_angle_deg": float(angle[-1]),
        "min_hold_angle_deg": hold_angle,
        "max_hold_angle_deg": hold_max,
        "hold_s": float(dt * holding.sum()),
        "max_base_shift_mm": 1e3 * float(body.max()),
        "max_foot_slip_mm": 1e3 * float(slip.max()),
        "max_friction_use": float(friction.max()),
        "min_foot_normal_force_n": float(upward.min()),
        "min_expected_tread_force_n": float(per_tread_load.min()),
        "loaded_motion_samples": int(moving.sum()),
        "loaded_motion_angle_range_deg": (
            [float(angle[moving].min()), float(angle[moving].max())] if np.any(moving) else []
        ),
        "min_motion_resistance_nm": float(sustained_torque.min()) if np.any(moving) else 0.0,
        "max_motion_resistance_nm": float(sustained_torque.max()) if np.any(moving) else 0.0,
        "max_grip_gap_m": float(log["extra"][active, 3].max()),
        "max_abs_pin_torque_nm": float(np.abs(log["extra"][active, 4]).max()),
    }
    result["verified"] = bool(
        abs(result["start_deg"]) <= ANGLE_TOLERANCE_DEG
        and result["max_angle_deg"] >= TARGET_DEG
        and hold_angle >= TARGET_DEG - ANGLE_TOLERANCE_DEG
        and hold_max <= TARGET_DEG + ANGLE_TOLERANCE_DEG
        and result["hold_s"] >= HOLD_S - dt - 1e-9
        and body.max() <= HOLD_LIMITS["base_m"]
        and slip.max() <= HOLD_LIMITS["slip_m"]
        and friction.max() <= HOLD_LIMITS["friction"]
        and result["min_expected_tread_force_n"] >= LOAD_THRESHOLD_N
        and result["min_motion_resistance_nm"] >= resistance_nm - 0.02
        and metrics["max_ratio"] <= 1.0
        and metrics["saturated_steps"] == 0
        and not metrics["fell"]
        and max(metrics["unexpected_contacts_n"].values(), default=0.0) <= 10.0
    )
    return result


def run(
    study: dict,
    candidate: int,
    motion_s: float,
    settle_s: float,
    gain: float,
    timestep: float,
    hold_torque_nm: float = 0.0,
) -> dict:
    saved = study["candidates"][candidate]
    track = make_track(study, saved, motion_s, settle_s, hold_torque_nm)
    stance = {leg: (saved["feet_xyz_m"][leg], "floor") for leg in LEGS}
    treads = expected_surfaces(stance, study["rise_m"], study["foot_radius_m"])

    model, data = stair_valve.build_scene(
        study["slope_deg"],
        study["rise_m"],
        saved["centre_m"],
        saved["start_deg"],
        saved["sense"],
        saved["handle_length_m"],
        saved["resistance_nm"],
    )
    hinge = model.joint(stair_valve.VALVE_HINGE)
    assert model.nu == len(joint_names())
    assert not np.any(model.actuator_trnid[:, 0] == hinge.id)
    loss = float(model.dof_frictionloss[hinge.dofadr[0]])
    assert loss == saved["resistance_nm"]

    def watch(model, data):
        feet_on_treads = support_loads(model, data)[np.arange(len(LEGS)), treads]
        return np.r_[stair_valve.observe(model, data), feet_on_treads]

    metrics, log = transition.execute(
        track,
        None,
        physics={"timestep": timestep},
        builder=lambda: (model, data),
        watch=watch,
        gains_scale=gain,
        limits_nm=torque_limits(study["arm"]),
        log_every_step=True,
    )
    result = evaluate(log, metrics, motion_s, settle_s, loss)
    stride = 5
    return {
        "arm": study["arm"],
        "slope_deg": study["slope_deg"],
        "rise_m": study["rise_m"],
        "candidate": candidate,
        "motion_s": motion_s,
        "settle_s": settle_s,
        "hold_feedforward_nm": hold_torque_nm,
        "observation_step_s": float(log["time"][1] - log["time"][0]),
        "hinge_frictionloss_nm": loss,
        "foot_treads": treads,
        "foot_order": LEGS,
        "grip_model": "ideal_pin",
        "controller_metrics": metrics,
        "valve": result,
        "final_qpos": log["qpos"][-1].tolist(),
        "samples": {
            "time_s": log["time"][::stride].tolist(),
            "angle_deg": np.degrees(log["extra"][::stride, 0]).tolist(),
            "angular_velocity_rad_s": log["extra"][::stride, 1].tolist(),
            "friction_torque_nm": log["extra"][::stride, 2].tolist(),
            "pin_torque_nm": log["extra"][::stride, 4].tolist(),
        },
    }


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="mode", required=True)
    plan = modes.add_parser("find")
    for name in ("geometry", "stair_route", "out"):
        plan.add_argument(name, type=Path)
    plan.add_argument("--candidates", type=int, default=6)
    replay = modes.add_parser("replay")
    for name in ("study", "out"):
        replay.add_argument(name, type=Path)
    replay.add_argument("--candidate", type=int, default=0)
    replay.add_argument("--motion-seconds", type=float, default=8.75)
    replay.add_argument("--settle-seconds", type=float, default=END_SETTLE_S)
    replay.add_argument("--gain", type=float, default=2.4)
    replay.add_argument("--timestep", type=float, default=PHYSICS_STEP_S)
    replay.add_argument("--hold-feedforward-nm", type=float, default=3.0)
    args = cli.parse_args()
    if args.out.exists():
        cli.error(f"Output already exists: {args.out}")
    if args.mode == "find":
        if args.candidates < 1:
            cli.error("Request at least one candidate")
        result = find(read(args.geometry), read(args.stair_route), args.candidates)
        print(f"Fixed-body candidates: {len(result['candidates'])}", flush=True)
    else:
        positive = (args.motion_seconds, args.gain, args.timestep)
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            cli.error("Motion duration, gain and timestep must be finite and positive")
        if not np.isfinite(args.settle_seconds) or args.settle_seconds < 0:
            cli.error("Settle time must be finite and non-negative")
        if not np.isfinite(args.hold_feedforward_nm) or args.hold_feedforward_nm < 0:
            cli.error("Hold feedforward must be finite and non-negative")
        study = read(args.study)
        if not 0 <= args.candidate < len(study["candidates"]):
            cli.error("Candidate index is outside the saved study")
        result = run(
            study,
            args.candidate,
            args.motion_seconds,
            args.settle_seconds,
            args.gain,
            args.timestep,
            args.hold_feedforward_nm,
        )
        print(result["valve"], flush=True)
    save(args.out, result)


if __name__ == "__main__":
    main()
