"""Compare two independent recordings with the same scene, commands and physics step."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCENE_FIELDS = (
    "extent",
    "settings",
    "centre_m",
    "start_deg",
    "sense",
    "timestep_s",
    "impratio",
    "gain",
    "arm_limits",
    "joint_limits_nm",
    "anchor_time_s",
    "motion",
    "planner_source_sha256",
    "walking_control",
    "opening_approach",
    "valve_operation",
    "station_base_pose",
    "post_retract_station_s",
)
ADDED_SENSOR = "pad_closing_tangent_force_n"
LOAD_CHANNELS = {
    "joint_force_world_n",
    "joint_moment_world_nm",
    "joint_off_axis_bending_nm",
    "joint_reaction_n",
    "rb1_mount_wrench_world",
    "foot_surface_force_world_n",
}


def compare(first, second, *, allow_load_observer=False):
    scenes = [json.loads((path / "scene.json").read_bytes()) for path in (first, second)]
    mismatches = [k for k in SCENE_FIELDS if scenes[0].get(k) != scenes[1].get(k)]
    references = [s.get("walking_reference") or {} for s in scenes]
    if any(references[0].get(k) != references[1].get(k) for k in ("sha256", "footprint_profile")):
        mismatches.append("walking_reference")
    servo_periods = [s.get("actuation_period_s", s["timestep_s"]) for s in scenes]
    if servo_periods[0] != servo_periods[1]:
        mismatches.append("actuation_period_s")
    for field in ("jaw_load_bias_m", "stance_spring_gain_n_per_m", "extra_settle_s"):
        if scenes[0].get(field, 0.0) != scenes[1].get(field, 0.0):
            mismatches.append(field)
    if any(s.get("stance_spring_gain_n_per_m", 0.0) for s in scenes):
        if scenes[0].get("stance_spring_active_s") != scenes[1].get("stance_spring_active_s"):
            mismatches.append("stance_spring_active_s")
    sources = [s["source_sha256"] for s in scenes]
    changed_sources = sorted(
        k for k in sources[0].keys() | sources[1].keys() if sources[0].get(k) != sources[1].get(k)
    )
    fingerprints = []
    for path in (first, second):
        with (path / "mission.npz").open("rb") as stream:
            fingerprints.append(hashlib.file_digest(stream, "sha256").hexdigest())
    with (
        np.load(first / "mission.npz", allow_pickle=False) as a,
        np.load(second / "mission.npz", allow_pickle=False) as b,
    ):
        fields = set(a.files) & set(b.files)
        extra = sorted(set(a.files) ^ set(b.files))
        allowed = {ADDED_SENSOR}
        if allow_load_observer and set(extra) & LOAD_CHANNELS:
            recorded_groups = [LOAD_CHANNELS.intersection(x.files) for x in (a, b)]
            if not (LOAD_CHANNELS in recorded_groups and set() in recorded_groups):
                raise ValueError("Unmatched raw fields contain an incomplete load recording")
            allowed |= LOAD_CHANNELS
        if set(extra) - allowed:
            raise ValueError(f"Unmatched raw fields: {extra}")
        differences = {}
        for key in sorted(fields):
            x, y = a[key], b[key]
            if x.shape != y.shape or x.dtype != y.dtype:
                differences[key] = {
                    "shapes": [x.shape, y.shape],
                    "dtypes": [str(x.dtype), str(y.dtype)],
                }
                continue
            x_bytes = np.ascontiguousarray(x).view(np.uint8).reshape(len(x), -1)
            y_bytes = np.ascontiguousarray(y).view(np.uint8).reshape(len(y), -1)
            if not np.array_equal(x_bytes, y_bytes):
                differences[key] = {
                    "first_sample": int(np.flatnonzero(np.any(x_bytes != y_bytes, axis=1))[0])
                }
        counts = [len(a["time_s"]), len(b["time_s"])]
    return {
        "raw_sha256": fingerprints,
        "physics_steps": counts,
        "actuation_period_s": servo_periods,
        "exact_raw_file_match": fingerprints[0] == fingerprints[1],
        "scene_field_differences": mismatches,
        "source_files_changed": changed_sources,
        "checked_raw_fields": sorted(fields),
        "unmatched_sensor_fields": extra,
        "raw_differences": differences,
        "same_scene_and_common_samples": not mismatches and not differences,
        "same_source_scene_and_common_samples": not changed_sources
        and not mismatches
        and not differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--require-same-source",
        action="store_true",
        help="Require identical controller source hashes as well as the same scene and samples",
    )
    parser.add_argument(
        "--allow-load-observer",
        action="store_true",
        help="Compare common samples when one run also recorded all six structural load channels",
    )
    args = parser.parse_args()
    result = compare(args.first, args.second, allow_load_observer=args.allow_load_observer)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        f"{len(result['checked_raw_fields'])} shared arrays, {result['physics_steps']} steps, "
        f"exact match: {result['same_scene_and_common_samples']}"
    )
    criterion = (
        "same_source_scene_and_common_samples"
        if args.require_same_source
        else "same_scene_and_common_samples"
    )
    if not result[criterion]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
