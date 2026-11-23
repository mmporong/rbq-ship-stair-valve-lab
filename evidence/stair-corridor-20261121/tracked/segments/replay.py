"""Replay one side of the closed valve from its planned initial state."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

from replay_stair_mission import TRACK_FIELDS, subset  # noqa: E402
from replay_stair_sequence import DWELL_S, measure_milestone  # noqa: E402
from stair_mission_limits import configure_mission_torque_limits  # noqa: E402
from stair_mission_reference import file_sha256, scheduled_landings  # noqa: E402
from stair_mission_route import route  # noqa: E402
from stair_mission_trace import (  # noqa: E402
    PAD_PAIRS,
    SLIDES,
    measured_walk,
    observe,
    trace_arrays,
)

from go2_wall import grasp_motion, stair_valve, transition  # noqa: E402
from go2_wall.stair_walking_control import StairWalkingControl  # noqa: E402

WINDOWS = {"up": (89.6, 189.6, 0.016), "down": (544.6, 651.7, 0.0)}


def run(walk_path, direction, output):
    output.mkdir(parents=True, exist_ok=False)
    geometry, sequence, study, _, ends, anchor = route(50.0, 0.19)
    begin, end, margin = WINDOWS[direction]
    with np.load(walk_path, allow_pickle=False) as saved:
        ends, profile = scheduled_landings(saved, ends, anchor, 50.0, 0.19)
        if profile is None:
            raise ValueError("Supply the fingerprinted lateral walking plan")
        time = saved["time"]
        start, stop = np.interp([begin, end], saved["reference_time"], time)
        walk = {k: saved[k] for k in ("time", *TRACK_FIELDS)}
        track = subset(walk, (time >= start - 1e-6) & (time <= stop + 1e-6))
    track["time"] -= track["time"][0]
    duration = float(track["time"][-1])
    case = study["candidates"][4]
    model, data = stair_valve.build_grasp_scene(
        50, 0.19, case["centre_m"], case["start_deg"], case["sense"], 0.12, 10.0
    )
    data.qpos[model.joint(stair_valve.VALVE_HINGE).qposadr[0]] = np.pi / 2
    for name in SLIDES:
        track["qpos"][0, model.joint(name).qposadr[0]] = grasp_motion.OPEN_FINGER_M
    limits = configure_mission_torque_limits(model, "rb1")
    control = StairWalkingControl(model, track, [(0.0, duration, margin)], 0.005, 0.6)

    def open_hand(_, position, velocity):
        return grasp_motion.finger_command(
            -1.0, position, velocity, boundaries={"approach": 0.0}, force_limit_n=100.0
        )

    metrics, log = transition.execute(
        track,
        None,
        physics={"timestep": 0.001},
        builder=lambda: (model, data),
        watch=observe,
        gains_scale=sequence["gain_scale"],
        swing_from_plan=sequence["swing_from_plan"],
        actuation_period_s=0.005,
        task_contact_pairs=PAD_PAIRS,
        actuator_command=open_hand,
        limits_nm=limits,
        walking_control=control,
    )
    trace, loads = trace_arrays(model, log, ["return"] * len(log["time"]), 0.6)
    path = output / "segment.npz"
    with path.open("xb") as out:
        np.savez_compressed(
            out,
            time=log["time"],
            base=log["base"],
            qpos=log["qpos"],
            qvel=log["qvel"],
            feet=log["feet"],
            loads=loads,
            slip=trace["stance_slip_m"],
            track=trace["base_track_m"],
            unexpected=trace["unexpected_contact_force_n"],
            hinge=trace["valve_angle_deg"],
            attempt=trace["qp_attempt"],
            fell=trace["fell"],
        )
    # The first full support window begins after this separately initialized segment starts.
    planned = [
        (t - start, feet, leg) for t, feet, leg in ends if start + DWELL_S <= t <= stop + 1e-6
    ]
    observed = {**log, "extra": loads}
    landings = [measure_milestone(observed, geometry, *goal) for goal in planned]
    result = dict(
        scope="Closed-valve segment initialized once at its scheduled starting pose",
        original_walk_phase_s=[begin, end],
        seconds=duration,
        physics_step_s=0.001,
        actuation_period_s=0.005,
        raw_sample_period_s=float(np.diff(log["time"][:2])[0]),
        walking_sha256=file_sha256(walk_path),
        raw_sha256=file_sha256(path),
        walking=measured_walk(geometry, model, planned, log, loads, trace, limits, full=False),
        landings=landings,
        loaded_landings=sum(m["support"]["all_expected_feet_ge_1n"] for m in landings),
        metrics=metrics,
    )
    with (output / "result.json").open("x", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{result['loaded_landings']}/{len(planned)} landings, {file_sha256(path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=WINDOWS)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--walk", type=Path, default=Path(__file__).resolve().parents[1] / "walk50.npz"
    )
    args = parser.parse_args()
    run(args.walk, args.direction, args.output)
