"""Cross-check measured load peaks and every video timestamp against raw physics samples."""

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from go2_wall.stair_loads import FOOT_NAMES, JOINT_NAMES, SURFACE_NAMES
from go2_wall.statics import torque_limits


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def movie_info(video):
    executable = shutil.which("ffprobe")
    options = [
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,avg_frame_rate,duration",
        "-of",
        "json",
    ]
    if executable:
        args = [executable, *options, str(video.resolve())]
    else:
        args = ["wsl.exe", "--exec", "ffprobe", *options, video.name]
    output = subprocess.check_output(args, cwd=video.parent, encoding="utf-8")
    return json.loads(output)["streams"][0]


def nearest_samples(labels, times):
    after = np.minimum(np.searchsorted(times, labels), len(times) - 1)
    before = np.maximum(after - 1, 0)
    return np.where(abs(times[after] - labels) < abs(times[before] - labels), after, before)


def check_walking_captions(subtitle, labels, times, raw, *, required=False):
    """Check the walking measurements printed alongside each physical timestamp."""
    closest = nearest_samples(labels, times)
    checks = {}
    for title, key, scale, decimals in (
        ("지금까지 최대 지지발 미끄러짐", "stance_slip_m", 1000.0, 3),
        ("비허용 접촉력", "unexpected_contact_force_n", 1.0, 2),
    ):
        if required or title in subtitle:
            if key not in raw:
                raise ValueError(f"Missing raw walking measurements for {key}")
            recorded = re.findall(re.escape(title) + r" ([0-9.]+)", subtitle)
            measured = [f"{value * scale:.{decimals}f}" for value in raw[key][closest]]
            if recorded != measured:
                raise ValueError(f"Walking captions differ from recorded {key}")
            checks[key] = len(recorded)
    return checks


def verify(run_dir, video, speed, start=None, *, require_walking_captions=False):
    manifest = json.loads((run_dir / "scene.json").read_text(encoding="utf-8"))
    peaks = json.loads(video.with_suffix(".peaks.json").read_text(encoding="utf-8"))
    info = movie_info(video)
    num, denom = (int(part) for part in info["avg_frame_rate"].split("/"))
    fps = num / denom
    subtitle = video.with_suffix(".ass").read_text(encoding="utf-8")
    labels = np.asarray([float(t) for t in re.findall(r"원시각 ([0-9.]+)s", subtitle)])
    if len(labels) != int(info["nb_frames"]) or not len(labels):
        raise ValueError("Video frame count and raw-time subtitle count do not match")
    raw = run_dir / "mission.npz"
    with np.load(raw, allow_pickle=False) as data:
        times = data["time_s"]
        left, right = peaks["time_range_s"]
        mask = (times >= left) & (times <= right)
        indices = np.flatnonzero(mask)
        origin = left if start is None else start
        frame_times = origin + np.arange(len(labels)) * speed / fps
        frame_error = np.abs(labels - frame_times)
        # Captions print milliseconds. The selected physical sample may follow the frame time.
        limit = min(speed / fps, manifest["timestep_s"] + 0.00051)
        if np.max(frame_error) > limit:
            raise ValueError(f"Caption differs from frame time by {np.max(frame_error)} s")
        walking_captions = check_walking_captions(
            subtitle, labels, times, data, required=require_walking_captions
        )
        stress_count = 0
        if manifest.get("valve_operation") == "open":
            from stair_stress_heatmap import stress_series

            rows, frames = np.unique(nearest_samples(labels, times), return_inverse=True)
            sample = {
                key: data[key][rows]
                for key in ("time_s", "stage", "qpos", "rb1_mount_wrench_world")
            }
            values = stress_series(sample)["von_mises_mpa"][frames]
            captions = re.findall(r"가정 장착판 등가응력 ([0-9.]+)MPa", subtitle)
            if captions != [f"{value:.2f}" for value in values]:
                raise ValueError("Mount stress captions differ from the recorded wrench and pose")
            stress_count = len(captions)
        limits = np.asarray(manifest.get("joint_limits_nm", torque_limits("rb1")))
        forces = np.linalg.norm(data["foot_surface_force_world_n"][mask], axis=-1)
        torque = np.abs(data["torque_nm"][mask])
        arrays = {
            "joint_torque_ratio": (torque / limits, JOINT_NAMES),
            "foot_surface_force_n": (
                forces.reshape(len(indices), -1),
                [f"{foot}/{surface}" for foot in FOOT_NAMES for surface in SURFACE_NAMES],
            ),
            "pad_closing_torque_nm": (np.abs(data["pad_closing_torque_nm"][mask, None]), None),
        }
        if "joint_torque_nm" in peaks["metrics"]:
            arrays["joint_torque_nm"] = (torque, JOINT_NAMES)
        checks = {}
        for name, (values, items) in arrays.items():
            sample, column = np.unravel_index(np.argmax(values), values.shape)
            entry = peaks["metrics"][name]
            value, time_s = float(values[sample, column]), float(times[indices[sample]])
            if (
                entry["value"] != value
                or entry["time_s"] != time_s
                or entry["sample_index"] != sample
            ):
                raise ValueError(f"Raw peak/time does not match {name}")
            if items is not None and entry["item"] != items[column]:
                raise ValueError(f"Raw sensor name does not match {name}")
            checks[name] = {"value": value, "time_s": time_s, "raw_index": int(indices[sample])}
    files = [raw, video, video.with_suffix(".ass"), video.with_suffix(".peaks.json")]
    load_table = video.with_suffix(".loads.csv")
    if load_table.exists():
        with load_table.open(encoding="utf-8", newline="") as stream:
            rows = {(r["quantity"], r["component"]): r for r in csv.DictReader(stream)}
        sensors = {
            "joint_torque_ratio": (
                "joint_torque_ratio",
                peaks["metrics"]["joint_torque_ratio"]["item"],
            ),
            "foot_surface_force_n": (
                "foot_surface_reaction",
                peaks["metrics"]["foot_surface_force_n"]["item"],
            ),
            "pad_closing_torque_nm": ("pad_closing_torque", "pads"),
        }
        if "joint_torque_nm" in checks:
            sensors["joint_torque_nm"] = (
                "joint_torque",
                peaks["metrics"]["joint_torque_nm"]["item"],
            )
        for name, sensor in sensors.items():
            row = rows[sensor]
            if (
                abs(float(row["value_at_absolute_peak"])) != checks[name]["value"]
                or float(row["time_s"]) != checks[name]["time_s"]
            ):
                raise ValueError(f"Load table disagrees with raw peak for {sensor}")
        files.append(load_table)
    heatmap = video.with_suffix(".heatmap.html")
    if heatmap.exists():
        files.append(heatmap)
    return {
        "scope": manifest["extent"],
        "verified": True,
        "peaks": checks,
        "frame_count": len(labels),
        "fps": fps,
        "playback_speed": speed,
        "duration_s": float(info["duration"]),
        "max_caption_time_error_s": float(frame_error.max()),
        **({"walking_caption_samples": walking_captions} if walking_captions else {}),
        **({"mount_stress_caption_samples": stress_count} if stress_count else {}),
        "physical_seconds_per_frame": speed / fps,
        "sha256": {p.name: digest(p) for p in files},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--start", type=float, help="physical start time of a custom window")
    parser.add_argument(
        "--require-walking-captions",
        action="store_true",
        help="require slip and unexpected-contact measurements on every frame",
    )
    args = parser.parse_args()
    result = verify(
        args.run_dir,
        args.video,
        args.speed,
        args.start,
        require_walking_captions=args.require_walking_captions,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{result['frame_count']} frames and {len(result['peaks'])} raw peaks verified")


if __name__ == "__main__":
    main()
