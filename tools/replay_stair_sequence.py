"""Replay a saved contact sequence from the floor or a specified starting stance."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from probe_stair_transfer import START_HOLD_S, transfer_segments
from replay_stair_entry import contact_window, entry_stances
from replay_stair_transfer import (
    LOAD_THRESHOLD_N,
    SURFACES,
    TIMESTEP_S,
    expected_surfaces,
    stance_from,
    support_loads,
)

from go2_wall import stairs, transition
from go2_wall.robot import LEGS
from go2_wall.statics import torque_limits

DWELL_S = 0.3
MAX_FOOT_XY_ERROR_M = 0.035
NONFOOT_CONTACT_LIMIT_N = 10.0
RETURN_HOLD_S = 0.5
RETURN_XY_LIMIT_M = 0.15
RETURN_TILT_LIMIT_DEG = 10.0
RETURN_LINEAR_SPEED_LIMIT_MPS = 0.05
RETURN_ANGULAR_SPEED_LIMIT_RADPS = 0.15
TURN_TOLERANCE_DEG = 10.0
SWING_CURVES = {"standard": transition.swing_point, "over_tread": stairs.swing_over_tread}


def segments_and_milestones(
    geometry: dict, entry: dict | None, moves: list, *, initial: dict | None = None
) -> tuple[list, list]:
    if initial is None:
        before, left, stance = entry_stances(geometry)
        left_path = geometry["moves"]["floor_front_left_to_1"]["sampled_connection"]
        pair = next(replay for replay in entry["replays"] if replay["verified_pair"])
        plan = transfer_segments(before, left, "FL", left_path)
        milestones = [(sum(segment["seconds"] for segment in plan), left, "FL")]
        plan.extend(transfer_segments(left, stance, "FR", pair["connection"])[1:])
        milestones.append((sum(segment["seconds"] for segment in plan), stance, "FR"))
    else:
        stance = stance_from(initial["feet_xyz_m"])
        plan = [
            {"kind": "hold", "base": initial["base"], "stance": stance, "seconds": START_HOLD_S}
        ]
        milestones = []

    origin = tuple(plan[-1]["base"])
    for move in moves:
        leg = move["leg"]
        path = move["path"]
        target = np.asarray(move["target_xyz_m"], dtype=float)
        durations = {"shift": move.get("shift_s", transition.SHIFT_S), "swing": move["swing_s"]}
        curve = move.get("swing_curve", "standard")
        if (
            leg not in LEGS
            or tuple(path["start_base"]) != origin
            or target.shape != (3,)
            or not np.all(np.isfinite(target))
            or any(
                not np.isfinite(seconds) or seconds < transition.SAMPLE_S
                for seconds in durations.values()
            )
            or curve not in SWING_CURVES
        ):
            raise ValueError(f"Invalid saved transfer for {leg}")
        if any(
            np.linalg.norm(target - point) < 2 * geometry["foot_radius_m"] - 1e-9
            for other, (point, _) in stance.items()
            if other != leg
        ):
            raise ValueError(f"Saved target for {leg} overlaps a supporting foot")
        after = {**stance, leg: (target, "floor")}
        extra = transfer_segments(stance, after, leg, path, swing_curve=SWING_CURVES[curve])[1:]
        for segment in extra:
            if segment["kind"] in durations:
                segment["seconds"] = durations[segment["kind"]]
        plan.extend(extra)
        milestones.append((sum(segment["seconds"] for segment in plan), after, leg))
        stance, origin = after, tuple(extra[-1]["base"])
    return plan, milestones


def measure_milestone(log: dict, geometry: dict, end_s: float, stance: dict, leg: str) -> dict:
    treads = tuple(expected_surfaces(stance, geometry["rise_m"], geometry["foot_radius_m"]))
    dwell = contact_window(log, end_s - DWELL_S, end_s, treads)
    frames = np.flatnonzero((log["time"] >= end_s - DWELL_S) & (log["time"] < end_s))
    last = frames[-1]
    foot = log["feet"][last][LEGS.index(leg)]
    error = float(np.linalg.norm(foot[:2] - stance[leg][0][:2]))
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, log["qpos"][last, 3:7])
    return {
        "motion_end_s": end_s,
        "moved_leg": leg,
        "expected_foot_xyz_m": stance[leg][0].tolist(),
        "actual_foot_xyz_m": foot.tolist(),
        "terminal_foot_xy_error_m": error,
        "base_xyz_m": log["base"][last, :3].tolist(),
        "base_heading_deg": float(np.degrees(np.arctan2(rotation[3], rotation[0]))),
        "support": dwell,
    }


def measure_floor_return(log: dict, duration: float) -> dict:
    start = duration - RETURN_HOLD_S
    times = log["time"]
    frames = (times >= start) & (times <= duration)
    samples = times[frames]
    if len(times) < 2 or len(samples) == 0:
        return {"time_s": [start, duration], "sample_count": 0, "verified": False}

    poses = log["qpos"][frames]
    velocities = log["qvel"][frames]
    start_xy_error = np.linalg.norm(poses[:, :2] - log["qpos"][0, :2], axis=1)
    rotation = np.empty(9)
    up_z = []
    for qpos in poses:
        mujoco.mju_quat2Mat(rotation, qpos[3:7])
        up_z.append(rotation[8])
    result = {
        "time_s": [start, duration],
        "sample_count": len(samples),
        "max_start_xy_error_m": float(start_xy_error.max()),
        "min_up_z": float(min(up_z)),
        "max_linear_speed_mps": float(np.linalg.norm(velocities[:, :3], axis=1).max()),
        "max_angular_speed_radps": float(np.linalg.norm(velocities[:, 3:6], axis=1).max()),
        "min_floor_force_n": float(log["extra"][frames, :, 0].min()),
    }
    interval = times[1] - times[0]
    result["verified"] = bool(
        start >= 0.0
        and samples[0] <= start + interval + 1e-9
        and samples[-1] >= duration - interval - 1e-9
        and result["max_start_xy_error_m"] <= RETURN_XY_LIMIT_M
        and result["min_up_z"] >= np.cos(np.radians(RETURN_TILT_LIMIT_DEG))
        and result["max_linear_speed_mps"] <= RETURN_LINEAR_SPEED_LIMIT_MPS
        and result["max_angular_speed_radps"] <= RETURN_ANGULAR_SPEED_LIMIT_RADPS
        and result["min_floor_force_n"] >= LOAD_THRESHOLD_N
    )
    return result


def round_trip_summary(result: dict, log: dict) -> dict:
    metrics = result["controller_metrics"]
    stable = measure_floor_return(log, metrics["seconds"])
    platform = SURFACES.index("platform")
    landing = [
        m for m in result["milestones"]
        if tuple(m["support"]["treads"]) == (platform,) * len(LEGS)
        and m["support"]["all_expected_feet_ge_1n"]
    ]
    heading = landing[0]["base_heading_deg"] if landing else 0.0
    turn = next(
        (
            m for m in landing
            if abs((m["base_heading_deg"] - heading) % 360 - 180) <= TURN_TOLERANCE_DEG
        ),
        None,
    )
    return {
        "stable_floor_return": stable,
        "first_all_platform_s": landing[0]["motion_end_s"] if landing else None,
        "first_turn_on_platform_s": turn["motion_end_s"] if turn else None,
        "physical_round_trip_verified": bool(
            result["verified_sequence"]
            and metrics["passed"]
            and stable["verified"]
            and tuple(result["start_support"]["treads"]) == (0,) * len(LEGS)
            and turn is not None
        ),
    }


def replay(geometry: dict, entry: dict | None, sequence: dict, *, round_trip: bool = False) -> dict:
    arm, slope, rise = geometry["arm"], geometry["slope_deg"], geometry["rise_m"]
    stairs.ARM = arm
    stairs.use_stow()
    limits = torque_limits(arm)
    plan, ends = segments_and_milestones(
        geometry, entry, sequence["moves"], initial=sequence.get("initial")
    )

    def builder():
        return stairs.build_stairs(slope, rise)

    model, data = builder()
    np.testing.assert_array_equal(model.actuator_ctrlrange[:, 1], limits)
    track = transition.sample_segments(model, data, plan, limits, geometry["arm_stow_rad"])
    metrics, log = transition.execute(
        track,
        None,
        physics={"timestep": TIMESTEP_S},
        builder=builder,
        watch=support_loads,
        limits_nm=limits,
        gains_scale=sequence["gain_scale"],
        swing_from_plan=sequence["swing_from_plan"],
    )
    return evaluate_sequence(geometry, plan, ends, model, metrics, log, round_trip=round_trip)


def evaluate_sequence(geometry, plan, ends, model, metrics, log, *, round_trip=False) -> dict:
    """Judge the physical trajectory; its controller may also include a stationary arm task."""
    arm, slope, rise = geometry["arm"], geometry["slope_deg"], geometry["rise_m"]
    first_surfaces = expected_surfaces(plan[0]["stance"], rise, geometry["foot_radius_m"])
    start = contact_window(log, 0.2, START_HOLD_S, tuple(first_surfaces))
    milestones = [measure_milestone(log, geometry, *end) for end in ends]
    stair = geometry["stair_geometry"]
    positions = np.concatenate((log["feet"], log["base"][:, None, :3]), axis=1)
    platform = model.geom("platform")
    run_m = platform.pos[0] - platform.size[0]
    on_stair_x = (positions[..., 0] >= 0.0) & (positions[..., 0] < run_m)
    inside_width = not np.any(on_stair_x & (np.abs(positions[..., 1]) > stair["clear_width_m"] / 2))
    over_platform = positions[positions[..., 0] >= run_m]
    inside_platform = np.all(np.abs(over_platform[:, :2] - platform.pos[:2]) <= platform.size[:2])
    floor_beyond_first = np.any(
        (log["extra"][:, :, 0] >= 1.0) & (log["feet"][..., 0] > stair["tread_depth_m"])
    )
    verified = (
        start["all_expected_feet_ge_1n"]
        and inside_width
        and inside_platform
        and not floor_beyond_first
        and not metrics["fell"]
        and metrics["saturated_steps"] == 0
        and metrics["max_ratio"] <= 1.0
        and all(
            force <= NONFOOT_CONTACT_LIMIT_N for force in metrics["unexpected_contacts_n"].values()
        )
        and all(
            m["support"]["all_expected_feet_ge_1n"]
            and m["terminal_foot_xy_error_m"] <= MAX_FOOT_XY_ERROR_M
            for m in milestones
        )
    )
    result = {
        "arm": arm,
        "slope_deg": slope,
        "rise_m": rise,
        "foot_order": LEGS,
        "verified_sequence": bool(verified),
        "physics_timestep_s": float(metrics.get("physics", {}).get("timestep", model.opt.timestep)),
        "contact_record_interval_s": float(log["time"][1] - log["time"][0]),
        "maximum_base_x_m": float(log["base"][:, 0].max()),
        "maximum_base_z_m": float(log["base"][:, 2].max()),
        "start_support": start,
        "milestones": milestones,
        "inside_stair_width_on_samples": bool(inside_width),
        "inside_platform_on_samples": bool(inside_platform),
        "floor_support_beyond_first_tread": bool(floor_beyond_first),
        "final_qpos": log["qpos"][-1].tolist(),
        "controller_metrics": metrics,
    }
    if round_trip:
        result.update(round_trip_summary(result, log))
        arm_joint2 = geometry["arm_stow_rad"]["arm_joint2"]
        result["controller_settings"] = {"arm_joint2_stow_rad": arm_joint2}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("poses", type=Path)
    parser.add_argument("front_pair", type=Path, nargs="?")
    for name in ("sequence", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--round-trip", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    geometry = json.loads(args.poses.read_text(encoding="utf-8"))
    sequence = json.loads(args.sequence.read_text(encoding="utf-8"))
    entry = (
        None if args.front_pair is None else json.loads(args.front_pair.read_text(encoding="utf-8"))
    )
    if (entry is None) == (sequence.get("initial") is None):
        parser.error("Provide either front_pair or an initial pose in sequence")
    result = replay(geometry, entry, sequence, round_trip=args.round_trip)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(f"Sequence verified: {result['verified_sequence']}", flush=True)
    if args.round_trip:
        print(f"Physical round trip verified: {result['physical_round_trip_verified']}", flush=True)


if __name__ == "__main__":
    main()
