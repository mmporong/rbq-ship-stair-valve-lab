"""Estimate and render spatial RBQ link stresses from recorded six-axis joint loads.

The vendor model does not provide link wall thicknesses or material cards.  This module
therefore reports a conditional thin-wall section estimate.  The section outer dimensions
follow the visible/collision envelope in the pinned model and the wall thickness is stated
for every component family.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from go2_wall.stair_loads import JOINT_NAMES

MPA_SCALE = 20.0
CHASSIS_MPA_SCALE = 80.0
BIN_S = 0.5
RAINBOW = np.array(
    (
        (38, 70, 183),
        (0, 172, 193),
        (38, 177, 76),
        (255, 214, 0),
        (213, 45, 38),
    ),
    dtype=float,
)

# Outer section dimensions come from the model's link envelopes. Thickness is an
# explicitly conditional value because the RBQ/RB1 meshes contain no wall specification.
SECTION_FAMILIES = {
    "hip": {"width_m": 0.080, "height_m": 0.055, "wall_m": 0.003},
    "thigh": {"width_m": 0.070, "height_m": 0.045, "wall_m": 0.003},
    "calf": {"width_m": 0.045, "height_m": 0.030, "wall_m": 0.003},
    "arm_base": {"width_m": 0.080, "height_m": 0.070, "wall_m": 0.004},
    "arm_upper": {"width_m": 0.090, "height_m": 0.070, "wall_m": 0.004},
    "arm_fore": {"width_m": 0.070, "height_m": 0.055, "wall_m": 0.003},
    "arm_wrist": {"width_m": 0.060, "height_m": 0.050, "wall_m": 0.003},
}
JOINT_FAMILIES = (
    "hip",
    "thigh",
    "calf",
) * 4 + (
    "arm_base",
    "arm_upper",
    "arm_fore",
    "arm_fore",
    "arm_wrist",
    "arm_wrist",
)
CHASSIS_ZONE_LABELS = ("FL chassis", "FR chassis", "RL chassis", "RR chassis")
HIP_INDICES = tuple(index for index, name in enumerate(JOINT_NAMES) if "_hip_" in name)
CHASSIS_SOURCE_LABELS = (*CHASSIS_ZONE_LABELS, "RB1 mount conditional plate")
CHASSIS_ANCHOR_XY = np.array(
    ((0.25, 0.10), (0.25, -0.10), (-0.25, 0.10), (-0.25, -0.10), (0.0, 0.0))
)


def _section_properties(spec: dict[str, float]) -> tuple[float, float, float]:
    width, height, wall = spec["width_m"], spec["height_m"], spec["wall_m"]
    inner_width, inner_height = width - 2 * wall, height - 2 * wall
    if min(inner_width, inner_height, wall) <= 0:
        raise ValueError("thin-wall section dimensions must leave a positive hollow interior")
    area = width * height - inner_width * inner_height
    inertia_y = (width * height**3 - inner_width * inner_height**3) / 12
    inertia_z = (height * width**3 - inner_height * inner_width**3) / 12
    section_modulus = min(inertia_y / (height / 2), inertia_z / (width / 2))
    median_area = (width - wall) * (height - wall)
    return area, section_modulus, median_area


def rainbow_rgb(values: np.ndarray | float, maximum: float = MPA_SCALE) -> np.ndarray:
    """Map MPa values to the fixed blue-cyan-green-yellow-red scale."""
    share = np.clip(np.asarray(values, dtype=float) / maximum, 0, 1)
    position = share * (len(RAINBOW) - 1)
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, len(RAINBOW) - 1)
    mix = position - low
    return (RAINBOW[low] * (1 - mix[..., None]) + RAINBOW[high] * mix[..., None]) / 255


def link_stress_series(mission: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return conditional equivalent stress for all 18 joint-attached link sections."""
    force = np.asarray(mission["joint_force_world_n"], dtype=float)
    moment = np.asarray(mission["joint_moment_world_nm"], dtype=float)
    bending = np.asarray(mission["joint_off_axis_bending_nm"], dtype=float)
    expected = (len(mission["time_s"]), len(JOINT_NAMES))
    if force.shape != (*expected, 3) or moment.shape != (*expected, 3) or bending.shape != expected:
        raise ValueError("joint force and moment arrays do not match the 18-link time series")
    if not all(np.isfinite(value).all() for value in (force, moment, bending)):
        raise ValueError("joint load arrays must be finite")

    force_magnitude = np.linalg.norm(force, axis=2)
    moment_magnitude = np.linalg.norm(moment, axis=2)
    torsion = np.sqrt(np.maximum(moment_magnitude**2 - bending**2, 0))
    axial, bending_stress, torsion_stress = [], [], []
    for index, family in enumerate(JOINT_FAMILIES):
        area, section_modulus, median_area = _section_properties(SECTION_FAMILIES[family])
        axial.append(force_magnitude[:, index] / area / 1e6)
        bending_stress.append(bending[:, index] / section_modulus / 1e6)
        wall = SECTION_FAMILIES[family]["wall_m"]
        torsion_stress.append(torsion[:, index] / (2 * median_area * wall) / 1e6)
    axial = np.asarray(axial).T
    bending_stress = np.asarray(bending_stress).T
    torsion_stress = np.asarray(torsion_stress).T
    equivalent = np.sqrt((axial + bending_stress) ** 2 + 3 * torsion_stress**2)
    return {
        "axial_mpa": axial,
        "bending_mpa": bending_stress,
        "torsion_mpa": torsion_stress,
        "equivalent_mpa": equivalent,
        "chassis_zone_mpa": equivalent[:, HIP_INDICES],
    }


def link_geom_targets(model) -> list[list[int]]:
    """Return visible/collision geometry IDs attached to each measured joint body."""
    targets = []
    for name in JOINT_NAMES:
        body = int(model.jnt_bodyid[model.joint(name).id])
        geoms = [index for index in range(model.ngeom) if model.geom_bodyid[index] == body]
        if not geoms:
            raise ValueError(f"joint body has no renderable surface: {name}")
        targets.append(geoms)
    return targets


CHASSIS_DIAGNOSTIC_XY = np.array(
    [(x, y) for x in np.linspace(-0.4, 0.4, 17) for y in np.linspace(-0.1475, 0.1475, 7)]
)


def chassis_point_weights(local_xy: np.ndarray) -> np.ndarray:
    """Inverse-distance weights from four hips and the arm mount at body-local points."""
    points = np.asarray(local_xy, dtype=float)
    squared = np.sum((points[..., None, :] - CHASSIS_ANCHOR_XY) ** 2, axis=-1)
    weights = 1 / (squared + 0.035**2)
    return weights / weights.sum(axis=-1, keepdims=True)


CHASSIS_DIAGNOSTIC_WEIGHTS = chassis_point_weights(CHASSIS_DIAGNOSTIC_XY)


def chassis_source_series(
    mission: dict[str, np.ndarray], equivalent_mpa: np.ndarray
) -> np.ndarray:
    """Combine hip-section estimates with the separate conditional RB1 mounting-plate model."""
    from stair_stress_heatmap import stress_series

    mount = stress_series(mission)["von_mises_mpa"]
    return np.column_stack((equivalent_mpa[:, HIP_INDICES], mount))


def chassis_point_stress(local_xy: np.ndarray, source_stresses: np.ndarray) -> np.ndarray:
    values = np.asarray(source_stresses, dtype=float)
    if values.shape[-1] != len(CHASSIS_SOURCE_LABELS):
        raise ValueError("chassis interpolation needs four hip values and one mount value")
    return chassis_point_weights(local_xy) @ values


def chassis_field_diagnostics(times: np.ndarray, source_stresses: np.ndarray) -> dict:
    """Prove spatial variation and temporal change on fixed body-local mesh samples."""
    best_range, best_index, best_min, best_max = -np.inf, -1, np.nan, np.nan
    for start in range(0, len(source_stresses), 5000):
        field = source_stresses[start : start + 5000] @ CHASSIS_DIAGNOSTIC_WEIGHTS.T
        ranges = np.ptp(field, axis=1)
        local = int(np.argmax(ranges))
        if ranges[local] > best_range:
            best_range = float(ranges[local])
            best_index = start + local
            best_min = float(field[local].min())
            best_max = float(field[local].max())
    mean_field = source_stresses @ CHASSIS_DIAGNOSTIC_WEIGHTS.mean(axis=0)
    return {
        "maximum_spatial_range_mpa": best_range,
        "maximum_spatial_range_time_s": float(times[best_index]),
        "field_min_at_that_time_mpa": best_min,
        "field_max_at_that_time_mpa": best_max,
        "mean_field_temporal_range_mpa": float(np.ptp(mean_field)),
    }


def colour_chassis_pixels(
    frame: np.ndarray,
    segmentation: np.ndarray,
    depth: np.ndarray,
    camera,
    base_body,
    geom_ids: tuple[int, ...],
    source_stresses: np.ndarray,
) -> dict:
    """Reproject visible chassis geom pixels and colour the actual rendered mesh silhouette."""
    mask = np.isin(segmentation[..., 0], geom_ids) & (segmentation[..., 1] == 5)
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError("rendered frame contains no chassis mesh pixels")
    distance = depth[mask].astype(float)
    forward = np.asarray(camera.forward, dtype=float)
    up = np.asarray(camera.up, dtype=float)
    right = np.cross(forward, up)
    half_height = frame.shape[0] / 2
    horizontal = ((columns + 0.5) - frame.shape[1] / 2) / half_height
    vertical = (half_height - (rows + 0.5)) / half_height
    projection = float(camera.frustum_top / camera.frustum_near)
    rays = forward + projection * (horizontal[:, None] * right + vertical[:, None] * up)
    world = np.asarray(camera.pos) + distance[:, None] * rays
    rotation = np.asarray(base_body.xmat).reshape(3, 3)
    local = (world - np.asarray(base_body.xpos)) @ rotation
    local_xy = np.clip(local[:, :2], (-0.412, -0.1475), (0.412, 0.1475))
    stress = chassis_point_stress(local_xy, source_stresses)
    heat = rainbow_rgb(stress, CHASSIS_MPA_SCALE) * 255
    shade = np.clip(frame[mask].mean(axis=1, keepdims=True) / 115, 0.58, 1.08)
    frame[mask] = np.rint(np.clip(heat * shade, 0, 255)).astype(np.uint8)
    bands = np.digitize(stress, (20.0, 40.0, 60.0))
    return {
        "mesh_pixel_count": int(mask.sum()),
        "unique_20mpa_colour_bands": int(len(np.unique(bands))),
        "visible_min_mpa": float(stress.min()),
        "visible_max_mpa": float(stress.max()),
        "visible_mean_mpa": float(stress.mean()),
    }


def paint_fixed_legend(frame: np.ndarray) -> None:
    """Draw fixed link and chassis bars; subtitle text supplies numeric tick labels."""
    x0, x1 = frame.shape[1] - 350, frame.shape[1] - 28
    for y0, maximum in ((frame.shape[0] - 58, CHASSIS_MPA_SCALE), (frame.shape[0] - 28, MPA_SCALE)):
        y1 = y0 + 12
        values = np.linspace(0, maximum, x1 - x0)
        frame[y0:y1, x0:x1] = np.rint(rainbow_rgb(values, maximum) * 255).astype(np.uint8)
        frame[y0 - 2 : y0, x0 - 2 : x1 + 2] = 255
        frame[y1 : y1 + 2, x0 - 2 : x1 + 2] = 255
        for tick in np.linspace(x0, x1 - 1, 5).astype(int):
            frame[y0 - 2 : y1 + 2, tick : tick + 2] = 255


def _movie_info(video: Path) -> dict:
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
    if executable := shutil.which("ffprobe"):
        command = [executable, *options, str(video.resolve())]
        cwd = None
    else:
        command = ["wsl.exe", "--exec", "ffprobe", *options, video.name]
        cwd = video.parent
    return json.loads(subprocess.check_output(command, cwd=cwd, encoding="utf-8"))["streams"][0]


def _surface_resolved(sample: dict) -> bool:
    """Old IDW renders proved two 20 MPa bands; trunk-beam renders prove a spatial range."""
    if "unique_20mpa_colour_bands" in sample:
        return sample["unique_20mpa_colour_bands"] >= 2
    return sample["mesh_pixel_count"] > 0 and sample["visible_max_mpa"] > sample["visible_min_mpa"]


def verify_render(mission: dict[str, np.ndarray], video: Path) -> dict:
    subtitle = video.with_suffix(".ass")
    text = subtitle.read_text(encoding="utf-8")
    labels = np.asarray([float(value) for value in re.findall(r"원시각 ([0-9.]+)s", text)])
    captions = re.findall(r"가정 단면 최대 ([^ ]+) ([0-9.]+)MPa", text)
    info = _movie_info(video)
    if not len(labels) or len(labels) != len(captions) or len(labels) != int(info["nb_frames"]):
        raise ValueError("video frames, physical timestamps and body-stress captions must match")
    links_path = video.with_suffix(".links.json")
    if links_path.exists():
        # Spatial link renders store the per-frame link peaks they painted and captioned.
        frames = json.loads(links_path.read_text(encoding="utf-8"))["frames"]
        if not np.allclose([frame["time_s"] for frame in frames], labels, atol=5e-4):
            raise ValueError("link stress sidecar times differ from the captions")
        equivalent = np.asarray([frame["links_mpa"] for frame in frames])
    else:
        times = mission["time_s"]
        after = np.minimum(np.searchsorted(times, labels), len(times) - 1)
        before = np.maximum(after - 1, 0)
        indices = np.where(
            abs(times[after] - labels) < abs(times[before] - labels), after, before
        )
        equivalent = link_stress_series(mission)["equivalent_mpa"][indices]
    peaks = np.argmax(equivalent, axis=1)
    expected = [(JOINT_NAMES[link], f"{row[link]:.2f}") for row, link in zip(equivalent, peaks)]
    if captions != expected:
        raise ValueError("body-stress captions differ from recorded six-axis loads")
    result = {
        "video": video.name,
        "video_sha256": _sha256(video),
        "subtitle_sha256": _sha256(subtitle),
        "frames": len(labels),
        "duration_s": float(info["duration"]),
        "avg_frame_rate": info["avg_frame_rate"],
        "caption_samples_verified": len(captions),
    }
    surface_path = video.with_suffix(".surface.json")
    if surface_path.exists():
        surface = json.loads(surface_path.read_text(encoding="utf-8"))
        if surface.get("added_overlay_geoms") != 0:
            raise ValueError("body stress render must not add chassis overlay geoms")
        samples = surface.get("samples", [])
        if samples and not all(_surface_resolved(sample) for sample in samples):
            raise ValueError("representative chassis mesh masks must show a spatial field")
        if len(samples) == 2 and surface.get("visible_mean_change_mpa", 0) <= 0:
            raise ValueError("representative chassis mesh masks must change over time")
        result["surface_sha256"] = _sha256(surface_path)
        result["surface_mask_verification"] = surface
    return result


def build_artifacts(
    mission_path: Path,
    output_json: Path,
    output_csv: Path,
    videos: tuple[Path, ...] = (),
) -> dict:
    with np.load(mission_path, allow_pickle=False) as saved:
        required = (
            "time_s",
            "stage",
            "joint_force_world_n",
            "joint_moment_world_nm",
            "joint_off_axis_bending_nm",
            "qpos",
            "rb1_mount_wrench_world",
        )
        mission = {name: saved[name] for name in required}
    stress = link_stress_series(mission)
    equivalent = stress["equivalent_mpa"]
    times, stages = mission["time_s"], mission["stage"]
    chassis_sources = chassis_source_series(mission, equivalent)
    chassis_diagnostics = chassis_field_diagnostics(times, chassis_sources)
    peak_flat = int(np.argmax(equivalent))
    sample, link = np.unravel_index(peak_flat, equivalent.shape)
    sections = {
        name: {**SECTION_FAMILIES[name], "source": "pinned model envelope plus assumed wall"}
        for name in SECTION_FAMILIES
    }
    summary = {
        "schema": "rbq_link_conditional_stress_v1",
        "mission_sha256": _sha256(mission_path),
        "samples": len(times),
        "scale_mpa": [0.0, MPA_SCALE],
        "chassis_scale_mpa": [0.0, CHASSIS_MPA_SCALE],
        "colour_order": ["blue", "cyan", "green", "yellow", "red"],
        "formula": "sqrt((|F|/A + |M_bend|/Z_min)^2 + 3*(|M_torsion|/(2*A_m*t))^2)",
        "load_origin": "MuJoCo mj_rnePostConstraint parent reaction at each of 18 joint anchors",
        "scope": (
            "conditional thin-wall link-section estimate; "
            "not an FEA result or component rating"
        ),
        "sections": sections,
        "joint_families": dict(zip(JOINT_NAMES, JOINT_FAMILIES, strict=True)),
        "peak": {
            "link": JOINT_NAMES[link],
            "time_s": float(times[sample]),
            "stage": str(stages[sample]),
            "equivalent_mpa": float(equivalent[sample, link]),
        },
        "chassis_zones": dict(
            zip(CHASSIS_ZONE_LABELS, (JOINT_NAMES[i] for i in HIP_INDICES), strict=True)
        ),
        "chassis_surface": {
            "render_target": "visible pixels of the original MuJoCo base geoms only",
            "added_overlay_geoms": 0,
            "diagnostic_body_local_samples": len(CHASSIS_DIAGNOSTIC_XY),
            "sources": list(CHASSIS_SOURCE_LABELS),
            "method": (
                "geom segmentation and depth/camera reprojection to base-local XY, followed by "
                "inverse-distance interpolation with 35 mm smoothing radius"
            ),
            "warning": (
                "spatial visualization combines four conditional hip link sections with the "
                "separate Go2+D1 conditional RB1 mounting-plate estimate; it is not FEA"
            ),
            **chassis_diagnostics,
        },
        "renders": [verify_render(mission, video) for video in videos],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    edge_start = np.floor(times[0] / BIN_S) * BIN_S
    edge_end = (np.floor(times[-1] / BIN_S) + 2) * BIN_S
    edges = np.arange(edge_start, edge_end, BIN_S)
    bins = np.minimum(np.searchsorted(edges, times, side="right") - 1, len(edges) - 2)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("bin_start_s", "link", "peak_equivalent_mpa", "peak_time_s", "stage"))
        for bin_index in range(len(edges) - 1):
            rows = np.flatnonzero(bins == bin_index)
            if not len(rows):
                continue
            components = (*JOINT_NAMES, *CHASSIS_SOURCE_LABELS)
            component_values = np.column_stack((equivalent, chassis_sources))
            for component_index, name in enumerate(components):
                peak = rows[np.argmax(component_values[rows, component_index])]
                writer.writerow(
                    (
                        f"{edges[bin_index]:.3f}",
                        name,
                        component_values[peak, component_index],
                        times[peak],
                        stages[peak],
                    )
                )
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mission", type=Path)
    parser.add_argument("output", type=Path, help="JSON metadata output")
    parser.add_argument("--video", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.output.suffix.lower() != ".json":
        parser.error("output must end in .json")
    summary = build_artifacts(
        args.mission,
        args.output,
        args.output.with_suffix(".csv"),
        tuple(args.video),
    )
    print(json.dumps(summary["peak"], ensure_ascii=False))


if __name__ == "__main__":
    main()
