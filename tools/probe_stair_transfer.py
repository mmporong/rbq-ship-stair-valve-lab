"""Check RBQ-10+RB1 support poses and individual foot shifts on the T1 target stair."""

import argparse
import json
import multiprocessing
from functools import partial
from pathlib import Path

import mujoco
import numpy as np

from go2_wall import stairs, transition
from go2_wall.robot import LEGS
from go2_wall.statics import torque_limits

SLOPE_DEG = 50.0
RISE_M = 0.19
ARM = "rb1"
APPROACH_X_M = (-0.15, -0.60)  # Front and rear foot-centre x on the floor.
ENTRY_REAR_X_M = -0.45  # Deck support point for the partial entrance stance.
PITCH_OFFSETS_DEG = range(-50, 11, 5)
START_TRIES = 3
SWING_TRIES = 12
START_HOLD_S = 0.4
END_HOLD_S = 0.6


def pose_cases(radius: float) -> tuple[dict, list]:
    depth = stairs.stair_parameters()["tread_depth_m"]
    front_x, rear_x = APPROACH_X_M

    def stance(rear: int, front: int) -> dict:
        return stairs.stance_on(rear, front, SLOPE_DEG, RISE_M, depth, radius)

    floor = {
        leg: (
            np.array([front_x if leg.startswith("F") else rear_x, stairs.foot_y() * side, radius]),
            "floor",
        )
        for leg, side in zip(LEGS, (1, -1, 1, -1), strict=True)
    }
    front_one = stance(0, 1)
    for leg in ("RL", "RR"):
        front_one[leg][0][0] = ENTRY_REAR_X_M
    stances = {"floor": floor, "front_on_1": front_one, "rear_on_1_front_on_3": stance(1, 3)}
    moves = [
        ("floor_front_left_to_1", "floor", "FL", stance(0, 1)["FL"]),
        ("front_left_1_to_2", "front_on_1", "FL", stance(0, 2)["FL"]),
        ("rear_left_floor_to_1", "front_on_1", "RL", stance(1, 1)["RL"]),
        ("rear_left_1_to_2", "rear_on_1_front_on_3", "RL", stance(2, 3)["RL"]),
    ]
    return stances, moves


def scored_poses(
    pool, before: dict, after: dict, leg: str | None, *, swing_curve=transition.swing_point
) -> list:
    jobs = [(pitch, before, after, leg, SLOPE_DEG) for pitch in PITCH_OFFSETS_DEG]
    score = partial(stairs._quick, swing_curve=swing_curve)
    return sorted(row for group in pool.map(score, jobs) for row in group)


def transfer_segments(
    before: dict, after: dict, leg: str, path: dict, *, swing_curve=transition.swing_point
) -> list[dict]:
    start = tuple(path["start_base"])
    swing = tuple(path["swing_base"])
    swing_bases = path.get("swing_bases")
    settled = swing
    if swing_bases is not None:
        if (
            len(swing_bases) < 2
            or tuple(swing_bases[0]) != swing
            or any(len(base) != len(swing) or not np.all(np.isfinite(base)) for base in swing_bases)
        ):
            raise ValueError("Swing bases must start at the unloaded body pose")
        settled = tuple(swing_bases[-1])
    return [
        {"kind": "hold", "base": start, "stance": before, "seconds": START_HOLD_S},
        {
            "kind": "shift",
            "from": start,
            "to": swing,
            "stance": before,
            "seconds": transition.SHIFT_S,
        },
        {
            "kind": "unload",
            "base": swing,
            "stance": before,
            "leg": leg,
            "seconds": transition.HANDOFF_S,
        },
        {
            "kind": "swing",
            "base": swing,
            "before": before,
            "after": after,
            "leg": leg,
            "swing_curve": swing_curve,
            "swing_bases": swing_bases,
            "seconds": transition.SWING_S,
        },
        {
            "kind": "load",
            "base": settled,
            "stance": after,
            "leg": leg,
            "seconds": transition.HANDOFF_S,
        },
        {"kind": "hold", "base": settled, "stance": after, "seconds": END_HOLD_S},
    ]


def continuous_path(job, *, swing_curve=transition.swing_point) -> dict | None:
    start, swing, before, after, leg = job
    cost = stairs._path_job(job, swing_curve=swing_curve)
    if cost is None:
        return None
    path = {"start_base": start, "swing_base": swing, "path_cost": cost}
    model, data = stairs._STATE["pair"]
    try:
        transition.sample_segments(
            model,
            data,
            transfer_segments(before, after, leg, path, swing_curve=swing_curve),
            stairs._STATE["limits"],
            transition.tuck_arm(),
        )
    except transition.InfeasibleSegment:
        return None
    return path


def path_probe(
    pool, starts: list, swings: list, before: dict, after: dict, leg: str
) -> dict | None:
    for _, start in starts[:START_TRIES]:
        # Prefer feasible swing poses near this support posture for the bounded connection search.
        ranked = sorted(
            swings,
            key=lambda item: (
                np.linalg.norm(np.subtract(start[:3], item[1][:3])),
                abs(start[3] - item[1][3]),
            ),
        )
        paths = [(start, swing, before, after, leg) for _, swing in ranked[:SWING_TRIES]]
        for path in pool.map(continuous_path, paths):
            if path is not None:
                return path
    return None


def describe_feet(stance: dict) -> dict[str, list[float]]:
    return {leg: point.tolist() for leg, (point, _) in stance.items()}


def describe_poses(scored: list) -> dict:
    return {
        "robust_poses": len(scored),
        "base_x_past_first_edge_poses": sum(base[0] >= 0.0 for _, base in scored),
        "minimum_torque_ratio": scored[0][0] if scored else None,
        "minimum_ratio_base": scored[0][1] if scored else None,
        "maximum_forward_base": max(
            (base for _, base in scored), key=lambda base: base[0], default=None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--floor-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.out.exists():
        parser.error("Use a new output path and at least one worker")

    stairs.ARM = ARM
    stairs.use_stow()
    model, _ = stairs.build_stairs(SLOPE_DEG, RISE_M)
    limits = torque_limits(ARM)
    np.testing.assert_array_equal(model.actuator_ctrlrange[:, 1], limits)
    radius = float(model.geom_size[model.geom("FL").id, 0])
    stances, moves = pose_cases(radius)
    if args.floor_only:
        stances = {"floor": stances["floor"]}
        moves = [move for move in moves if move[1] == "floor"]
    result = {
        "mujoco": mujoco.__version__,
        "arm": ARM,
        "robot_mass_kg": float(model.body_subtreemass[model.body("base").id]),
        "slope_deg": SLOPE_DEG,
        "rise_m": RISE_M,
        "stair_geometry": stairs.stair_parameters(),
        "arm_stow_rad": stairs.tr.tuck_arm(),
        "torque_limits_nm": limits.tolist(),
        "foot_radius_m": radius,
        "body_clearance_m": stairs.tr.CLEARANCE_M,
        "friction_share": stairs.tr.FRICTION_MARGIN,
        "horizontal_push_body_weight_share": stairs.tr.DISTURBANCE_FRACTION,
        "pose_grid": {key: list(map(float, values)) for key, values in stairs.GRID.items()},
        "pitch_offsets_deg": list(PITCH_OFFSETS_DEG),
        "path_start_tries": START_TRIES,
        "path_swing_tries_per_start": SWING_TRIES,
        "floor_start_pitch_deg": 0.0,
        "stances": {},
        "moves": {},
    }
    samples = {}
    with multiprocessing.get_context("spawn").Pool(
        args.workers, initializer=stairs._init, initargs=(SLOPE_DEG, RISE_M, ARM)
    ) as pool:
        for name, before in stances.items():
            scored = samples[name] = scored_poses(pool, before, before, None)
            result["stances"][name] = {
                "foot_centres_xyz_m": describe_feet(before),
                **describe_poses(scored),
            }
            print(name, len(scored), flush=True)
        for name, start_name, leg, target in moves:
            before = stances[start_name]
            after = {**before, leg: target}
            scored = scored_poses(pool, before, after, leg)
            starts = samples[start_name]
            if start_name == "floor":
                starts = [row for row in starts if row[1][3] == 0.0]
            path = path_probe(pool, starts, scored, before, after, leg) if scored else None
            result["moves"][name] = {
                "start_stance": start_name,
                "leg": leg,
                "target_foot_centres_xyz_m": describe_feet(after),
                "sampled_connection": path,
                **describe_poses(scored),
            }
            print(name, len(scored), "connected" if path else "no sampled connection", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
