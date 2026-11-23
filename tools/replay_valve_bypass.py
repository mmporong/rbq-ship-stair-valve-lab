"""Replay a closed-valve floor approach and record physical contact and tracking events."""

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from stair_bypass_inputs import provenance, sha256_file
from stair_mission_route import route

from go2_wall import grasp_motion, stair_valve, transition
from go2_wall.robot import LEGS, world_geoms
from go2_wall.statics import torque_limits

TRACK_KEYS = ("time", "qpos", "target", "torque", "tip", "stance", "wall", "cap", "feet")
END_S = 107.5


def contact_sample(log: dict, first: dict | None) -> dict | None:
    if first is None:
        return None
    times = log["physics_time_s"]
    index = int(np.argmin(np.abs(times - first["time_s"])))
    return {
        "sample_index": index,
        "time_s": float(times[index]),
        "max_joint_torque_nm": float(np.max(np.abs(log["torque"][index]))),
        "foot_force_world_n": log["force"][index].tolist(),
        "base_track_mm": float(log["base_track_m"][index] * 1e3),
        "stance_slip_mm": float(log["stance_slip_m"][index] * 1e3),
        "fell": bool(log["fell"][index]),
        "valve_angle_deg": float(log["extra"][index, 0]),
    }


def load_track(path: Path, end_s: float = END_S) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        mask = saved["time"] <= end_s + 1e-9
        return {name: saved[name][mask].copy() for name in TRACK_KEYS}


def splice_track(walk: dict, accepted_path: Path) -> tuple[dict, dict]:
    with np.load(accepted_path, allow_pickle=False) as accepted:
        accepted_mask = accepted["time"] <= walk["time"][-1] + 1e-9
        times = accepted["time"][accepted_mask]
        if not len(times):
            raise ValueError("accepted control starts after --end-s")
        start, end = float(times[0]), float(times[-1])
        indices = np.flatnonzero((walk["time"] >= start - 1e-9) & (walk["time"] <= end + 1e-9))
        if len(indices) != len(times) or not np.allclose(walk["time"][indices], times, atol=1e-9):
            raise ValueError("accepted control timestamps do not align with walk")
        continuity = {
            "start_s": start,
            "qpos_jump_m_or_rad": float(
                np.max(np.abs(walk["qpos"][indices[0]] - accepted["qpos"][accepted_mask][0]))
            ),
            "target_jump_rad": float(
                np.max(np.abs(walk["target"][indices[0]] - accepted["target"][accepted_mask][0]))
            ),
            "foot_jump_m": float(
                np.max(
                    np.linalg.norm(
                        walk["feet"][indices[0]] - accepted["feet"][accepted_mask][0], axis=1
                    )
                )
            ),
        }
        out = {name: value.copy() for name, value in walk.items()}
        for name in TRACK_KEYS:
            if name != "time":
                out[name][indices] = accepted[name][accepted_mask]
    return out, continuity


class ValveWatch:
    def __init__(self, model):
        self.lever = model.geom(stair_valve.VALVE_LEVER).id
        self.feet = {model.geom(leg).id for leg in LEGS}
        self.terrain = set(world_geoms(model))
        self.first = None
        self.peak = {"force_n": 0.0}
        self.first_unexpected = None
        self.peak_unexpected = {"force_n": 0.0}
        self.wrench = np.zeros(6)

    def __call__(self, model, data):
        observation = stair_valve.observe_grasp(model, data)
        contacts = []
        largest = None
        for index in range(data.ncon):
            contact = data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            if (first in self.feet and second in self.terrain) or (
                second in self.feet and first in self.terrain
            ):
                continue
            mujoco.mj_contactForce(model, data, index, self.wrench)
            force = float(np.linalg.norm(self.wrench[:3]))
            if largest is None or force > largest[0]:
                largest = (force, first, second)
            if self.lever not in contact.geom:
                continue
            other = contact.geom2 if contact.geom1 == self.lever else contact.geom1
            name = model.geom(other).name or model.body(model.geom_bodyid[other]).name
            row = {"time_s": float(data.time), "part": name, "force_n": force}
            contacts.append(row)
            if force > self.peak["force_n"]:
                self.peak = row
            if self.first is None and force > 1.0:
                self.first = row
        if largest is not None:
            force, first, second = largest
            if force > self.peak_unexpected["force_n"] or (
                force > 10 and self.first_unexpected is None
            ):
                row = {
                    "time_s": float(data.time),
                    "force_n": force,
                    "bodies": [model.body(model.geom_bodyid[g]).name for g in (first, second)],
                    "geoms": [model.geom(g).name or f"geom[{g}]" for g in (first, second)],
                }
                if force > self.peak_unexpected["force_n"]:
                    self.peak_unexpected = row
                if force > 10 and self.first_unexpected is None:
                    self.first_unexpected = row
        return np.r_[
            observation["valve_angle_deg"],
            observation["valve_speed_rad_s"],
            observation["pad_normal_force_n"],
            observation["pad_closing_torque_nm"],
            observation["max_unexpected_lever_contact_n"],
            max((item["force_n"] for item in contacts), default=0.0),
        ]


def diagnostic_trace(model, log):
    columns = [model.joint(name).qposadr[0] for name in stair_valve.GRIPPER_JOINTS]
    keys = (
        "time",
        "physics_time_s",
        "extra",
        "unexpected_contact_n",
        "base_track_m",
        "stance_slip_m",
        "fell",
        "attempt",
        "torque",
        "force",
    )
    return {
        **{key: log[key] for key in keys},
        "finger_slide_m": log["qpos"][:, columns],
        "finger_command_n": log["ctrl"][:, model.actuator(stair_valve.GRIPPER_ACTUATOR).id],
        "peak_requested_torque_ratio": np.max(
            np.abs(log["robot_torque_requested_nm"]) / torque_limits("rb1"), axis=1
        ),
    }


def run_case(track, study, sequence, out: Path):
    saved = study["candidates"][4]
    model, data = stair_valve.build_grasp_scene(
        study["slope_deg"],
        study["rise_m"],
        saved["centre_m"],
        saved["start_deg"],
        saved["sense"],
        saved["handle_length_m"],
        saved["resistance_nm"],
    )
    data.qpos[model.joint(stair_valve.VALVE_HINGE).qposadr[0]] = np.pi / 2
    for name in stair_valve.GRIPPER_JOINTS:
        data.qpos[model.joint(name).qposadr[0]] = grasp_motion.OPEN_FINGER_M
        track["qpos"][0, model.joint(name).qposadr[0]] = grasp_motion.OPEN_FINGER_M
    mujoco.mj_forward(model, data)
    watch = ValveWatch(model)
    metrics, log = transition.execute(
        track,
        None,
        builder=lambda: (model, data),
        watch=watch,
        limits_nm=torque_limits("rb1"),
        gains_scale=1.5,
        swing_from_plan=sequence["swing_from_plan"],
        log_every_step=True,
        actuator_command=lambda *_: 100.0,
    )
    out.mkdir(parents=True, exist_ok=False)
    with (out / "raw.npz").open("xb") as stream:
        np.savez_compressed(stream, **log)
    channels = diagnostic_trace(model, log)
    with (out / "trace.npz").open("xb") as stream:
        np.savez_compressed(stream, **channels)
    extra = log["extra"]
    result = {
        "scope": "cold-closed valve walking diagnostic",
        "duration_s": float(log["physics_time_s"][-1]),
        "timestep_s": float(model.opt.timestep),
        "cold_initial_hinge_deg": 90.0,
        "external_force_n": 0.0,
        "gain": 1.5,
        "metrics": metrics,
        "first_lever_contact": watch.first,
        "peak_lever_contact": watch.peak,
        "first_unexpected_contact_over_10n": watch.first_unexpected,
        "peak_unexpected_contact": watch.peak_unexpected,
        "min_finger_slide_m": channels["finger_slide_m"].min(axis=0).tolist(),
        "raw_sha256": sha256_file(out / "raw.npz"),
        "trace_sha256": sha256_file(out / "trace.npz"),
        "valve_angle_range_deg": [float(extra[:, 0].min()), float(extra[:, 0].max())],
        "peak_pad_normal_n": float(extra[:, 2:4].max()),
        "peak_pad_torque_nm": float(np.abs(extra[:, 4]).max()),
        "first_contact_sample": contact_sample(log, watch.first),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("walk", type=Path)
    parser.add_argument("accepted", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--end-s", type=float, default=END_S)
    parser.add_argument(
        "--manifest", type=Path, help="check the published replay sources and inputs"
    )
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output exists: {args.out}")
    if not np.isfinite(args.end_s) or args.end_s <= 0:
        parser.error("--end-s must be positive")
    sources = provenance("tools/replay_valve_bypass.py")
    inputs = {"walk": sha256_file(args.walk), "accepted": sha256_file(args.accepted)}
    thread_environment = {
        key: os.getenv(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
    }
    if args.manifest:
        saved = json.loads(args.manifest.read_bytes())
        if thread_environment != saved["thread_environment"]:
            parser.error("BLAS and OpenMP thread settings differ from the published run")
        if (
            sources != saved["replay_provenance"]
            or inputs["walk"] != saved["shared_walk"]["sha256"]
        ):
            parser.error("The checked files differ from the published bypass inputs")
        cases = (saved["verified_prefix"], saved["extension_attempt"])
        if not any(
            c["input_sha256"] == inputs["accepted"] and c["end_s"] == args.end_s for c in cases
        ):
            parser.error("The control track or end time differs from the published case")
    geometry, sequence, study, *_ = route(50.0, 0.19)
    baseline = load_track(args.walk, args.end_s)
    accepted, continuity = splice_track(baseline, args.accepted)
    args.out.mkdir(parents=True)
    baseline_result = (
        None if args.accepted_only else run_case(baseline, study, sequence, args.out / "baseline")
    )
    accepted_result = run_case(accepted, study, sequence, args.out / "accepted")
    comparison = {
        "geometry": {"slope_deg": geometry["slope_deg"], "rise_m": geometry["rise_m"]},
        "continuity": continuity,
        "baseline": baseline_result,
        "accepted": accepted_result,
        "source_sha256": inputs,
        "replay_provenance": sources,
        "thread_environment": thread_environment,
    }
    (args.out / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"continuity": continuity, "baseline": baseline_result, "accepted": accepted_result}
        )
    )


if __name__ == "__main__":
    main()
