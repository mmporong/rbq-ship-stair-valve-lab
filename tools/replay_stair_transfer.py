"""Replay a connected foot transfer in contact physics with RBQ-10+RB1."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from probe_stair_transfer import END_HOLD_S, START_HOLD_S, transfer_segments

from go2_wall import stairs, transition
from go2_wall.robot import LEGS
from go2_wall.statics import torque_limits

TIMESTEP_S = 0.005
LOAD_THRESHOLD_N = 1.0
SURFACES = ("floor", *(f"tread{k}" for k in range(1, 11)), "platform")


def stance_from(points: dict) -> dict:
    return {leg: (np.array(points[leg]), "floor") for leg in LEGS}


def expected_surfaces(stance: dict, rise: float, radius: float) -> list[int]:
    return [round((stance[leg][0][2] - radius) / rise) for leg in LEGS]


def support_loads(model, data) -> np.ndarray:
    feet = {model.geom(leg).id: k for k, leg in enumerate(LEGS)}
    surfaces = {model.geom(name).id: k for k, name in enumerate(SURFACES)}
    load = np.zeros((len(feet), len(surfaces)))
    wrench = np.zeros(6)
    for i, contact in enumerate(data.contact):
        if contact.efc_address < 0:
            continue
        surface, foot = map(int, contact.geom)
        up = contact.frame[2]
        if surface in feet:
            surface, foot, up = foot, surface, -up
        if up > 0.5 and surface in surfaces and foot in feet:
            mujoco.mj_contactForce(model, data, i, wrench)
            load[feet[foot], surfaces[surface]] += max(0.0, wrench[0] * up)
    return load


def replay(geometry: dict, name: str, mode: str, gain: float) -> dict:
    move = geometry["moves"][name]
    path = move["sampled_connection"]
    if path is None:
        return {"move": name, "reason": "no sampled static connection"}
    slope, rise = geometry["slope_deg"], geometry["rise_m"]

    def builder():
        return stairs.build_stairs(slope, rise)

    model, data = builder()
    limits = torque_limits("rb1")
    np.testing.assert_array_equal(model.actuator_ctrlrange[:, 1], limits)
    before = stance_from(geometry["stances"][move["start_stance"]]["foot_centres_xyz_m"])
    after = stance_from(move["target_foot_centres_xyz_m"])
    plan = transfer_segments(before, after, move["leg"], path)
    track = transition.sample_segments(model, data, plan, limits, geometry["arm_stow_rad"])
    summary, log = transition.execute(
        track,
        None,
        physics={"timestep": TIMESTEP_S},
        gains_scale=gain,
        builder=builder,
        watch=support_loads,
        mode=mode,
        limits_nm=limits,
    )
    initial = expected_surfaces(before, rise, geometry["foot_radius_m"])
    final = expected_surfaces(after, rise, geometry["foot_radius_m"])
    start_window = (log["time"] >= START_HOLD_S / 2) & (log["time"] < START_HOLD_S)
    end_window = log["time"] >= track["time"][-1] - END_HOLD_S / 2
    initial_load = log["extra"][start_window][:, np.arange(len(LEGS)), initial]
    final_load = log["extra"][end_window][:, np.arange(len(LEGS)), final]
    return {
        "move": name,
        "mode": mode,
        "gain_scale": gain,
        "physics_timestep_s": TIMESTEP_S,
        "contact_record_interval_s": float(log["time"][1] - log["time"][0]),
        "initial_surfaces": dict(zip(LEGS, (SURFACES[i] for i in initial), strict=True)),
        "final_surfaces": dict(zip(LEGS, (SURFACES[i] for i in final), strict=True)),
        "initial_contact_dwell": bool(
            initial_load.size and np.all(initial_load >= LOAD_THRESHOLD_N)
        ),
        "final_contact_dwell": bool(final_load.size and np.all(final_load >= LOAD_THRESHOLD_N)),
        "last_normal_load_n": dict(zip(LEGS, final_load[-1].tolist(), strict=True)),
        "planned_max_torque_ratio": float(track["ratio"].max()),
        "actual_first_base_xyz_m": log["base"][0, :3].tolist(),
        "actual_last_base_xyz_m": log["base"][-1, :3].tolist(),
        "last_foot_centres_xyz_m": dict(zip(LEGS, log["feet"][-1].tolist(), strict=True)),
        "last_qpos": log["qpos"][-1].tolist(),
        "controller_metrics": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("poses", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--move", default="floor_front_left_to_1")
    parser.add_argument("--mode", choices=("force", "joint"), default="force")
    parser.add_argument("--gain", type=float, default=1.0)
    args = parser.parse_args()
    if args.out.exists() or not np.isfinite(args.gain) or args.gain <= 0:
        parser.error("Use a new output path and positive finite gains")
    geometry = json.loads(args.poses.read_text(encoding="utf-8"))
    stairs.ARM = geometry["arm"]
    stairs.use_stow()
    result = replay(geometry, args.move, args.mode, args.gain)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("initial_contact_dwell", "final_contact_dwell", "controller_metrics")
                if key in result
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
