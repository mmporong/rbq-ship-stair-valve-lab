"""Build the frozen N1 walking reference on the selected stair geometry."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from replay_stair_sequence import segments_and_milestones
from replay_stair_transfer import expected_surfaces

from go2_wall import stair_valve, stairs, transition
from go2_wall.robot import ROOT
from go2_wall.statics import torque_limits

ASCENT = ROOT / "evidence/t1-stair-ascent-20261120"
FRONT_PAIR = ROOT / "evidence/t1-stair-posture-20261120/front-pair.json"
VALVE = ROOT / "evidence/t3-valve-20261120/fixed-body-candidates.json"
PLANNER_SOURCES = (
    "tools/stair_mission_route.py",
    "tools/replay_stair_sequence.py",
    "tools/probe_stair_transfer.py",
    "tools/replay_stair_entry.py",
    "tools/replay_stair_transfer.py",
    "src/go2_wall/pose_search.py",
    "src/go2_wall/stairs.py",
    "src/go2_wall/transition.py",
    "src/go2_wall/robot.py",
    "src/go2_wall/statics.py",
)


def source_hashes(paths):
    return {
        path: hashlib.sha256((ROOT / path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in paths
    }


def planner_source_hashes():
    return source_hashes(PLANNER_SOURCES)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def route(slope, rise):
    geometry = load(ASCENT / "target-round-trip-geometry.json")
    sequence = load(ASCENT / "target-round-trip-plan.json")
    study = load(VALVE)
    segments, ends = segments_and_milestones(geometry, load(FRONT_PAIR), sequence["moves"])
    reference_rise = geometry["rise_m"]
    foot_radius = geometry["foot_radius_m"]
    anchor = next(
        i
        for i, (_, stance, _) in enumerate(ends)
        if tuple(expected_surfaces(stance, reference_rise, foot_radius)) == (5, 5, 3, 3)
    )
    if slope == geometry["slope_deg"] and rise == reference_rise:
        return geometry, sequence, study, segments, ends, anchor

    old_going = stairs.going_of(geometry["slope_deg"], reference_rise)
    going_change = stairs.going_of(slope, rise) - old_going
    rise_change = rise - reference_rise
    n_treads = 10

    def foot(xyz):
        point = np.array(xyz, dtype=float)
        level = round((point[2] - foot_radius) / reference_rise)
        point[0] += max(0, level - 1) * going_change
        point[2] += level * rise_change
        return point

    def base(pose):
        out = list(pose)
        progress = float(np.clip(out[0] / (n_treads * old_going), 0, 1))
        out[0] += progress * n_treads * going_change
        out[2] += progress * (n_treads + 1) * rise_change
        return out

    def support(stance):
        return {leg: (foot(xyz), surface) for leg, (xyz, surface) in stance.items()}

    for seg in segments:
        for key in ("base", "from", "to"):
            if key in seg:
                seg[key] = base(seg[key])
        for key in ("stance", "before", "after"):
            if key in seg:
                seg[key] = support(seg[key])
        if seg.get("swing_bases"):
            seg["swing_bases"] = [base(pose) for pose in seg["swing_bases"]]
    ends = [(t, support(stance), leg) for t, stance, leg in ends]
    for case in study["candidates"]:
        case["base_pose"] = base(case["base_pose"])
        case["feet_xyz_m"] = {leg: foot(xyz).tolist() for leg, xyz in case["feet_xyz_m"].items()}
        case["centre_m"] = (
            np.asarray(case["centre_m"]) + [4 * going_change, 0, 5 * rise_change]
        ).tolist()
    geometry.update(slope_deg=slope, rise_m=rise)
    study.update(slope_deg=slope, rise_m=rise)
    return geometry, sequence, study, segments, ends, anchor


def sample_walk(geometry, plan):
    stairs.ARM = geometry["arm"]
    stairs.use_stow()
    model, data = stairs.build_stairs(geometry["slope_deg"], geometry["rise_m"])
    return transition.sample_segments(
        model, data, plan, torque_limits(geometry["arm"]), geometry["arm_stow_rad"]
    )


def valve_clearance(study, times, poses, anchor_s, *, candidate=4):
    """Find lever overlaps on the planned walk, with the handle shut after the work stance."""
    saved = study["candidates"][candidate]
    model, data = stair_valve.build_grasp_scene(
        study["slope_deg"],
        study["rise_m"],
        saved["centre_m"],
        saved["start_deg"],
        saved["sense"],
        saved["handle_length_m"],
        saved["resistance_nm"],
    )
    lever = model.geom(stair_valve.VALVE_LEVER).id
    hinge_q = model.joint(stair_valve.VALVE_HINGE).qposadr[0]
    hits = {}
    first = None
    max_overlap = None
    for t, pose in zip(times, poses, strict=True):
        data.qpos[: len(pose)] = pose
        data.qpos[hinge_q] = 0 if t <= anchor_s else np.pi / 2
        mujoco.mj_fwdPosition(model, data)
        for contact in data.contact[: data.ncon]:
            if lever not in contact.geom or contact.dist >= -1e-4:
                continue
            other = contact.geom2 if contact.geom1 == lever else contact.geom1
            body = model.body(model.geom_bodyid[other]).name
            depth = float(-contact.dist)
            row = {"time_s": float(t), "body": body, "penetration_m": depth}
            if first is None:
                first = row
            if max_overlap is None or depth > max_overlap["penetration_m"]:
                max_overlap = row
            item = hits.setdefault(body, {"first_s": float(t), "last_s": float(t), "max_m": 0.0})
            item.update(last_s=float(t), max_m=max(depth, item["max_m"]))
    return {
        "kind": "kinematic",
        "candidate": candidate,
        "anchor_s": anchor_s,
        "sample_count": len(times),
        "first": first,
        "deepest": max_overlap,
        "by_body": hits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--slope", type=float, default=50.0)
    parser.add_argument("--rise", type=float, default=0.19)
    parser.add_argument("--check-valve", type=Path, help="scan a saved walk for handle overlaps")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output exists: {args.out}")
    geometry, _, study, plan, ends, anchor = route(args.slope, args.rise)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.check_valve:
        with np.load(args.check_valve, allow_pickle=False) as walk:
            if (float(walk["slope_deg"]), float(walk["rise_m"])) != (args.slope, args.rise):
                parser.error("Walking cache has different geometry")
            report = valve_clearance(study, walk["time"], walk["qpos"], ends[anchor][0])
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(report["first"], report["deepest"])
        return
    failure = args.out.with_suffix(".failure.json")
    if failure.exists():
        parser.error(f"Previous failure report exists: {failure}")
    version = planner_source_hashes()
    try:
        track = sample_walk(geometry, plan)
    except transition.InfeasibleSegment as error:
        with failure.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "outcome": "planning_failed",
                    "slope_deg": args.slope,
                    "rise_m": args.rise,
                    "planner_source_sha256": version,
                    "message": str(error),
                    "first_failure": error.details,
                },
                stream,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
        raise SystemExit(f"{error}; evidence {failure}") from error
    with args.out.open("xb") as stream:
        np.savez_compressed(
            stream,
            **track,
            slope_deg=args.slope,
            rise_m=args.rise,
            anchor_time_s=ends[anchor][0],
            planner_source_sha256=json.dumps(version, sort_keys=True),
        )
    print(f"{len(track['time'])} samples; valve stance at {ends[anchor][0]:.2f}s", flush=True)


if __name__ == "__main__":
    main()
