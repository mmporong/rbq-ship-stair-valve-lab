"""N1 walking-only control, with the same saved route at two integration rates."""

# ruff: noqa: E402 -- use the tools from this checkout for the archived replay.

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from replay_stair_sequence import evaluate_sequence
from replay_stair_transfer import support_loads
from stair_mission_route import route

from go2_wall import stairs, transition
from go2_wall.statics import torque_limits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dt", type=float)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    geo, seq, _, plan, ends, _ = route(50.0, 0.19)
    stairs.ARM = "rb1"
    stairs.use_stow()
    path = ROOT / "evidence/stair-mission-20261121/full/walk50.npz"
    with np.load(path, allow_pickle=False) as data:
        track = {
            k: data[k]
            for k in ("time", "qpos", "target", "torque", "tip", "stance", "wall", "cap", "feet")
        }
    model, data = stairs.build_stairs(50.0, 0.19)
    print("CPU walking control", args.dt, len(track["time"]), model.nq, flush=True)
    metrics, log = transition.execute(
        track,
        None,
        physics={"timestep": args.dt},
        builder=lambda: (model, data),
        watch=support_loads,
        limits_nm=torque_limits("rb1"),
        gains_scale=seq["gain_scale"],
        swing_from_plan=seq["swing_from_plan"],
        actuation_period_s=0.005,
    )
    result = evaluate_sequence(geo, plan, ends, model, metrics, log, round_trip=True)
    if result["physics_timestep_s"] != args.dt:
        raise ValueError("Walking result has the wrong physics timestep")
    result["motor_period_s"] = 0.005
    result["physics_steps"] = transition.physics_step_count(track["time"][-1], args.dt)
    result["source_scope"] = "RBQ-10+RB1 N1 stair without a valve; no manipulation or bypass"
    result["fixed_walk_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with (out / "result.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    with (out / "support.npz").open("xb") as f:
        np.savez_compressed(
            f,
            time_s=log["time"] + args.dt,
            reference_time_s=log["time"],
            qpos=log["qpos"],
            qvel=log["qvel"],
            feet=log["feet"],
            tread_load_n=log["extra"],
            torque_nm=log["torque"],
        )
    print(
        result["physical_round_trip_verified"],
        result["first_turn_on_platform_s"],
        metrics["max_stance_slip_mm"],
        sum(m["support"]["all_expected_feet_ge_1n"] for m in result["milestones"]),
        flush=True,
    )


if __name__ == "__main__":
    main()
