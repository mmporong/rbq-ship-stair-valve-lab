"""Render measured link loads from the preregistered first evaluation trajectory."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from rbq_articulated_control import digest
from render_stair_mission import build_model, render_video, video_samples

from go2_wall import stair_valve
from go2_wall.stair_loads import StairLoads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "reference", "cases", "selection", "states", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--valve-close-up", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("video already exists")
    case = json.loads(args.cases.read_bytes())[0]["case_id"]
    mode = json.loads(args.selection.read_bytes())["adopted_candidate"]
    actual = args.states / f"{case}--{mode}.npz"
    scene = json.loads(args.scene.read_bytes())
    model, data = build_model(scene)
    observer = StairLoads(model)
    with (
        np.load(actual, allow_pickle=False) as states,
        np.load(args.reference, allow_pickle=False) as ref,
    ):
        if (str(states["case_id"]), str(states["mode"]), str(states["scene_sha256"])) != (
            case,
            mode,
            digest(args.scene),
        ):
            raise ValueError("The selected trial does not match the scene or preregistered case")
        time = states["time_s"]
        start, end = float(time[0]), float(time[-1])
        speed = 8.0
        if args.valve_close_up:
            approach = np.flatnonzero(ref["stage"][: len(time)] == "approach")
            if not len(approach):
                raise ValueError("The trial ended before the valve approach")
            start, speed = float(time[approach[0]]), 2.0
        frame_indices = video_samples(time, start, end, speed)
        indices = np.unique(np.r_[frame_indices, len(time) - 1])
        mission = {"time_s": time[indices], "stage": ref["stage"][indices]}
        if not np.array_equal(
            indices[video_samples(mission["time_s"], start, end, speed)], frame_indices
        ):
            raise ValueError("Offline sampling changed the underlying physics frames")
        for field in ("qpos", "qvel", "ctrl"):
            mission[field] = states[field][indices]
    rows = []
    for i, stamp in enumerate(mission["time_s"]):
        data.time = stamp
        for field in ("qpos", "qvel", "ctrl"):
            getattr(data, field)[:] = mission[field][i]
        mujoco.mj_forward(model, data)
        sample = observer.sample(data)
        grasp = stair_valve.observe_grasp(model, data)
        for key in (
            "valve_angle_deg",
            "valve_speed_rad_s",
            "hinge_friction_torque_nm",
            "pad_closing_torque_nm",
            "pad_normal_force_n",
        ):
            sample[key] = grasp[key]
        sample["torque_nm"] = data.actuator_force[:18].copy()
        sample["foot_force_world_n"] = sample["foot_surface_force_world_n"].sum(axis=1)
        sample["tread_load_n"] = sample["foot_surface_force_world_n"][:, :, 2]
        rows.append(sample)
    mission.update({key: np.asarray([row[key] for row in rows]) for key in rows[0]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    render_video(
        scene,
        mission,
        args.out,
        start,
        end,
        speed=speed,
        follow_base=not args.valve_close_up,
        valve_view=args.valve_close_up,
        body_stress=True,
    )
    receipt = {
        "case_id": case,
        "candidate_id": mode,
        "state_sha256": digest(actual),
        "scene_sha256": digest(args.scene),
        "reference_sha256": digest(args.reference),
        "selection_sha256": digest(args.selection),
        "state_count": len(time),
        "frame_state_indices": frame_indices.tolist(),
        "terminal_state_index": len(time) - 1,
        "video_sha256": digest(args.out),
        "start_s": start,
        "playback_speed": speed,
        "subtitle_sha256": digest(args.out.with_suffix(".ass")),
        "surface_sha256": digest(args.out.with_suffix(".surface.json")),
        "links_sha256": digest(args.out.with_suffix(".links.json")),
        "end_s": end,
        "final_valve_deg": float(mission["valve_angle_deg"][-1]),
        "load_method": "mj_forward and mj_rnePostConstraint on the held-out physical states",
    }
    with args.out.with_suffix(".receipt.json").open("x", encoding="utf-8") as out:
        json.dump(receipt, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
