"""Join two front-foot placements into one CPU contact replay on the target stair."""

import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np
from probe_stair_transfer import continuous_path, scored_poses, transfer_segments
from replay_stair_transfer import TIMESTEP_S, stance_from, support_loads

from go2_wall import stairs, transition
from go2_wall.robot import LEGS
from go2_wall.statics import torque_limits

RIGHT_SWING_LIMIT = 36
PHYSICAL_REPLAY_LIMIT = 6


def find_right_foot_paths(left, both, base, arm, slope, rise, workers):
    with multiprocessing.get_context("spawn").Pool(
        workers, initializer=stairs._init, initargs=(slope, rise, arm)
    ) as pool:
        scored = scored_poses(pool, left, both, "FR")
        scored.sort(
            key=lambda row: (
                np.linalg.norm(np.subtract(row[1][:3], base[:3])),
                abs(row[1][3] - base[3]),
            )
        )
        candidates = [(base, pose, left, both, "FR") for _, pose in scored[:RIGHT_SWING_LIMIT]]
        connected = [path for path in pool.map(continuous_path, candidates) if path is not None]
    return len(scored), connected


def contact_window(log: dict, low: float, high: float, treads: tuple[int, ...]) -> dict:
    window = (log["time"] >= low) & (log["time"] < high)
    loads = log["extra"][window][:, np.arange(len(LEGS)), treads]
    return {
        "time_s": [low, high],
        "treads": treads,
        "all_expected_feet_ge_1n": bool(loads.size and np.all(loads >= 1.0)),
        "last_load_n": loads[-1].tolist() if loads.size else [],
    }


def entry_stances(geometry: dict) -> tuple[dict, dict, dict]:
    """Floor and two successive first-tread stances of the entrance plan."""
    before = stance_from(geometry["stances"]["floor"]["foot_centres_xyz_m"])
    move = geometry["moves"]["floor_front_left_to_1"]
    left = stance_from(move["target_foot_centres_xyz_m"])
    right = stairs.stance_on(
        0,
        1,
        geometry["slope_deg"],
        geometry["rise_m"],
        geometry["stair_geometry"]["tread_depth_m"],
        geometry["foot_radius_m"],
    )["FR"]
    return before, left, {**left, "FR": right}


def replay_pair(geometry: dict, workers: int) -> dict:
    arm, slope, rise = geometry["arm"], geometry["slope_deg"], geometry["rise_m"]
    stairs.ARM = arm
    stairs.use_stow()
    limits = torque_limits(arm)
    before, left, both = entry_stances(geometry)
    left_path = geometry["moves"]["floor_front_left_to_1"]["sampled_connection"]

    def builder():
        return stairs.build_stairs(slope, rise)

    count, paths = find_right_foot_paths(
        left, both, tuple(left_path["swing_base"]), arm, slope, rise, workers
    )
    result = {
        "arm": arm,
        "slope_deg": slope,
        "rise_m": rise,
        "foot_order": LEGS,
        "right_front_swing_poses": count,
        "sampled_right_front_connections": len(paths),
        "replays": [],
    }
    print(f"Right front: {count} support poses, {len(paths)} sampled connections", flush=True)
    first = transfer_segments(before, left, "FL", left_path)
    middle_s = sum(segment["seconds"] for segment in first)
    for path in paths[:PHYSICAL_REPLAY_LIMIT]:
        plan = first + transfer_segments(left, both, "FR", path)[1:]
        model, data = builder()
        try:
            track = transition.sample_segments(model, data, plan, limits, geometry["arm_stow_rad"])
        except transition.InfeasibleSegment as error:
            result["replays"].append({"connection": path, "sample_error": str(error)})
            continue
        metrics, log = transition.execute(
            track,
            None,
            physics={"timestep": TIMESTEP_S},
            builder=builder,
            watch=support_loads,
            limits_nm=limits,
        )
        end_s = track["time"][-1]
        dwell = [
            contact_window(log, 0.2, 0.4, (0, 0, 0, 0)),
            contact_window(log, middle_s - 0.3, middle_s, (1, 0, 0, 0)),
            contact_window(log, end_s - 0.3, end_s + 1e-9, (1, 1, 0, 0)),
        ]
        verified = (
            not metrics["fell"]
            and metrics["saturated_steps"] == 0
            and metrics["max_ratio"] <= 1.0
            and all(force <= 10.0 for force in metrics["unexpected_contacts_n"].values())
            and all(row["all_expected_feet_ge_1n"] for row in dwell)
        )
        result["replays"].append(
            {
                "connection": path,
                "dwell": dwell,
                "verified_pair": verified,
                "ending_qpos": log["qpos"][-1].tolist(),
                "metrics": metrics,
            }
        )
        print(f"Pair verified: {verified}, torque ratio {metrics['max_ratio']:.3f}", flush=True)
        if verified:
            break
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("poses", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.out.exists() or args.workers < 1:
        parser.error("Use a new output path and at least one worker")
    geometry = json.loads(args.poses.read_text(encoding="utf-8"))
    if geometry["moves"]["floor_front_left_to_1"]["sampled_connection"] is None:
        parser.error("The left-front transfer needs a sampled connection")
    result = replay_pair(geometry, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
