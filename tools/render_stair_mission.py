"""Render and summarize one RBQ stair-valve mission.

Usage:
  python tools/render_stair_mission.py RUN_DIR [OUT.mp4] [--start S] [--end S]

RUN_DIR contains ``scene.json`` and ``mission.npz``. The default video is
``RUN_DIR/mission.mp4``; the default companion files are ``heatmap.html`` and ``peaks.json``.
The same command and metrics render a full mission or a short failure/stage window.
"""

import argparse
import csv
import html
import json
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from link_surface_stress import (
    link_label,
    link_peaks,
    link_state,
    link_topology,
    location_label,
    paint_links,
    paint_trunk,
    trunk_profile,
    trunk_state,
)
from stair_mission_limits import apply_motor_torque_ranges
from stair_render_style import (
    BAR_X1,
    INSET_H,
    INSET_SLOTS,
    INSET_W,
    RANK_ROWS,
    RANK_STEP,
    RANK_TOP,
    configure_visual,
    follow_sun,
    frost_panel,
    hud_styles,
    inset_camera,
    legend_events,
    paint_legend,
    paint_rank_bars,
    panel_static_events,
    paste_inset,
    studio_decor,
)

from go2_wall import stair_valve
from go2_wall.stair_loads import FOOT_NAMES, JOINT_NAMES, MOUNT_WRENCH_FIELDS, SURFACE_NAMES
from go2_wall.statics import torque_limits

WIDTH, HEIGHT, FPS = 1280, 720, 25
BIN_S = 0.5
STAGES_KO = {
    "prewalk": "작업 자세까지 보행",
    "lower": "몸통 낮춤",
    "position": "밸브 작업 자세로 몸통 이동",
    "approach": "손 접근",
    "contact": "파지 시도",
    "operate": "회전",
    "settle": "정착",
    "hold": "유지",
    "release": "손 놓기",
    "retract": "팔 접기",
    "return": "조작 후 보행",
    "station": "작업 발판 정지 구간",
}
REQUIRED = (
    "time_s",
    "stage",
    "qpos",
    "qvel",
    "ctrl",
    "torque_nm",
    "foot_force_world_n",
    "tread_load_n",
    "joint_force_world_n",
    "joint_moment_world_nm",
    "joint_off_axis_bending_nm",
    "joint_reaction_n",
    "rb1_mount_wrench_world",
    "foot_surface_force_world_n",
    "valve_angle_deg",
    "valve_speed_rad_s",
    "hinge_friction_torque_nm",
    "pad_closing_torque_nm",
    "pad_normal_force_n",
)
RAMP = ("#dbeafe", "#93c5fd", "#3b82f6", "#1d4ed8", "#172554")


def load_run(folder: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
    with np.load(folder / "mission.npz", allow_pickle=False) as saved:
        mission = {name: saved[name] for name in saved.files}
    missing = [name for name in REQUIRED if name not in mission]
    if missing:
        raise ValueError(f"mission.npz is missing: {', '.join(missing)}")
    count = len(mission["time_s"])
    if count == 0 or any(len(mission[name]) != count for name in REQUIRED):
        raise ValueError("all mission arrays must share a non-empty time axis")
    if np.any(np.diff(mission["time_s"]) <= 0):
        raise ValueError("time_s must be strictly increasing")
    return manifest, mission


def build_model(manifest: dict, studio: bool = False) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile the mission scene; ``studio`` adds render-only sky, floor texture and light."""
    settings = manifest["settings"]
    model, data = stair_valve.build_grasp_scene(
        settings["slope_deg"],
        settings["rise_m"],
        manifest["centre_m"],
        manifest["start_deg"],
        manifest["sense"],
        settings["handle_length_m"],
        settings["resistance_nm"],
        floor_mu=settings["floor_mu"],
        pad_mu=settings["pad_mu"],
        finger_force_n=settings["finger_force_n"],
        slide_loss_n=settings["slide_loss_n"],
        decorate=studio_decor if studio else None,
    )
    model.opt.timestep = manifest["timestep_s"]
    model.opt.impratio = manifest["impratio"]
    if "joint_limits_nm" in manifest:
        apply_motor_torque_ranges(model, manifest["joint_limits_nm"])
    return model, data


def stage_labels(stage: np.ndarray, manifest: dict) -> np.ndarray:
    if np.issubdtype(stage.dtype, np.integer):
        names = manifest.get("stage_names")
        if not names or np.any((stage < 0) | (stage >= len(names))):
            raise ValueError("numeric stage values require matching scene.json stage_names")
        return np.asarray(names)[stage]
    return stage.astype(str)


def mission_window(
    mission: dict[str, np.ndarray], start: float, end: float
) -> dict[str, np.ndarray]:
    """Return the inclusive time-window view used by every companion artifact."""
    count = len(mission["time_s"])
    mask = (mission["time_s"] >= start) & (mission["time_s"] <= end)
    if not mask.any():
        raise ValueError("mission window contains no samples")
    if mask.all():
        return mission
    return {
        name: value[mask] if value.ndim and len(value) == count else value
        for name, value in mission.items()
    }


def _peak(times, stages, values, labels=None) -> dict:
    flat = int(np.nanargmax(values))
    index = np.unravel_index(flat, values.shape)
    sample = index[0]
    result = {
        "value": float(values[index]),
        "time_s": float(times[sample]),
        "stage": str(stages[sample]),
        "sample_index": int(sample),
    }
    if labels is not None and len(index) > 1:
        result["item"] = str(labels[index[1]])
    return result


def torque_ratios(manifest, mission):
    limits = np.asarray(manifest.get("joint_limits_nm", torque_limits("rb1")))
    if limits.shape != (18,) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("The run must specify positive torque limits for all 18 joints")
    return np.abs(mission["torque_nm"]) / limits


def peak_summary(manifest: dict, mission: dict[str, np.ndarray]) -> dict:
    times = mission["time_s"]
    stages = stage_labels(mission["stage"], manifest)
    ratio = torque_ratios(manifest, mission)
    foot = np.linalg.norm(mission["foot_surface_force_world_n"], axis=-1)
    foot_labels = [
        f"{foot_name}/{surface}" for foot_name in FOOT_NAMES for surface in SURFACE_NAMES
    ]
    mount = mission["rb1_mount_wrench_world"]
    metrics = {
        "joint_torque_nm": _peak(times, stages, np.abs(mission["torque_nm"]), JOINT_NAMES),
        "joint_torque_ratio": _peak(times, stages, ratio, JOINT_NAMES),
        "joint_off_axis_bending_nm": _peak(
            times, stages, mission["joint_off_axis_bending_nm"], JOINT_NAMES
        ),
        "joint_reaction_n": _peak(times, stages, mission["joint_reaction_n"], JOINT_NAMES),
        "rb1_mount_force_n": _peak(times, stages, np.linalg.norm(mount[:, :3], axis=1)),
        "rb1_mount_moment_nm": _peak(times, stages, np.linalg.norm(mount[:, 3:], axis=1)),
        "foot_surface_force_n": _peak(times, stages, foot.reshape(len(times), -1), foot_labels),
        "hinge_friction_torque_nm": _peak(
            times, stages, np.abs(mission["hinge_friction_torque_nm"])
        ),
        "pad_closing_torque_nm": _peak(times, stages, np.abs(mission["pad_closing_torque_nm"])),
        "pad_normal_force_n": _peak(
            times, stages, mission["pad_normal_force_n"], ("upper", "lower")
        ),
    }
    if "pad_closing_tangent_force_n" in mission:
        metrics["pad_closing_tangent_force_n"] = _peak(
            times, stages, mission["pad_closing_tangent_force_n"]
        )
    return {
        "time_range_s": [float(times[0]), float(times[-1])],
        "mission_seconds": float(times[-1] - times[0] + manifest["timestep_s"]),
        "mount_reference": "RB1_Base origin; force XYZ then moment XYZ in world axes",
        "metrics": metrics,
    }


def load_rows(manifest, mission):
    times = mission["time_s"]
    stages = stage_labels(mission["stage"], manifest)
    feet = [f"{foot}/{surface}" for foot in FOOT_NAMES for surface in SURFACE_NAMES]
    mount = mission["rb1_mount_wrench_world"]
    groups = [
        ("joint_torque", "N m", JOINT_NAMES, mission["torque_nm"]),
        ("joint_torque_ratio", "1", JOINT_NAMES, torque_ratios(manifest, mission)),
        ("joint_reaction", "N", JOINT_NAMES, mission["joint_reaction_n"]),
        ("off_axis_bending", "N m", JOINT_NAMES, mission["joint_off_axis_bending_nm"]),
        (
            "foot_surface_reaction",
            "N",
            feet,
            np.linalg.norm(mission["foot_surface_force_world_n"], axis=-1).reshape(len(times), -1),
        ),
        ("foot_surface_vertical", "N", feet, mission["tread_load_n"].reshape(len(times), -1)),
        ("mount_force", "N", MOUNT_WRENCH_FIELDS[:3], mount[:, :3]),
        ("mount_moment", "N m", MOUNT_WRENCH_FIELDS[3:], mount[:, 3:]),
        ("hinge_resistance", "N m", ("hinge",), mission["hinge_friction_torque_nm"][:, None]),
        ("pad_closing_torque", "N m", ("pads",), mission["pad_closing_torque_nm"][:, None]),
        ("pad_normal_force", "N", ("upper", "lower"), mission["pad_normal_force_n"]),
    ]
    if "pad_closing_tangent_force_n" in mission:
        groups.append(
            (
                "pad_closing_tangent_force",
                "N",
                ("pads",),
                mission["pad_closing_tangent_force_n"][:, None],
            )
        )
    for metric, unit, names, values in groups:
        for column, name in enumerate(names):
            series = values[:, column]
            index = int(np.argmax(np.abs(series)))
            yield (
                metric,
                name,
                unit,
                float(series[index]),
                float(times[index]),
                str(stages[index]),
            )


def _binned(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = times[0] + np.arange(int((times[-1] - times[0]) / BIN_S) + 2) * BIN_S
    boundaries = np.searchsorted(times, edges)
    boundaries[-1] = len(times)
    rows = np.zeros((values.shape[1], len(edges) - 1))
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        if start < end:
            rows[:, index] = values[start:end].max(axis=0)
    return edges, rows


def _colour(value: float, top: float) -> str:
    share = min(max(value / top, 0.0), 1.0) if top else 0.0
    return RAMP[min(int(share * len(RAMP)), len(RAMP) - 1)]


def _heat_table(title: str, times: np.ndarray, values: np.ndarray, labels, unit: str) -> str:
    edges, rows = _binned(times, values)
    top = float(rows.max())
    body = []
    stride = max(1, (len(edges) - 1) // 8)
    ticks = "".join(
        f'<th colspan="{min(stride, len(edges) - 1 - i)}">{edges[i]:.1f} s</th>'
        for i in range(0, len(edges) - 1, stride)
    )
    for label, row in zip(labels, rows):
        cells = "".join(
            f'<td style="background:{_colour(float(value), top)}" '
            f'title="{edges[index]:.1f}s · {value:.3g}{unit}"></td>'
            for index, value in enumerate(row)
        )
        body.append(f"<tr><th>{html.escape(str(label))}</th>{cells}</tr>")
    return (
        f"<section><h2>{html.escape(title)}</h2><div class=scroll><table>"
        f"<thead><tr><th>물리 시각</th>{ticks}</tr></thead>{''.join(body)}</table></div></section>"
    )


def heatmap_page(manifest: dict, mission: dict[str, np.ndarray]) -> str:
    times = mission["time_s"]
    ratio = torque_ratios(manifest, mission)
    foot = np.linalg.norm(mission["foot_surface_force_world_n"], axis=-1).max(axis=2)
    tread = mission["tread_load_n"].max(axis=1)
    mount = np.abs(mission["rb1_mount_wrench_world"])
    pad_forces = mission["pad_normal_force_n"]
    pad_labels = ["위 패드 법선", "아래 패드 법선"]
    if "pad_closing_tangent_force_n" in mission:
        pad_forces = np.column_stack((pad_forces, np.abs(mission["pad_closing_tangent_force_n"])))
        pad_labels.append("접선 합력 절댓값")
    stages = stage_labels(mission["stage"], manifest)
    sections = [
        _heat_table("관절 토크 절댓값", times, np.abs(mission["torque_nm"]), JOINT_NAMES, " N·m"),
        _heat_table("관절 토크 비율", times, ratio, JOINT_NAMES, ""),
        _heat_table("관절 반력", times, mission["joint_reaction_n"], JOINT_NAMES, " N"),
        _heat_table(
            "관절 비축 굽힘 모멘트",
            times,
            mission["joint_off_axis_bending_nm"],
            JOINT_NAMES,
            "N·m",
        ),
        _heat_table("발별 최대 지지면 반력", times, foot, FOOT_NAMES, "N"),
        _heat_table("지지면별 최대 수직 반력", times, tread, SURFACE_NAMES, "N"),
        _heat_table("장착부 힘의 축별 절댓값", times, mount[:, :3], ("X", "Y", "Z"), " N"),
        _heat_table("장착부 모멘트의 축별 절댓값", times, mount[:, 3:], ("X", "Y", "Z"), " N·m"),
        _heat_table("패드 접촉력", times, pad_forces, pad_labels, " N"),
        _heat_table(
            "밸브와 파지 토크",
            times,
            np.column_stack(
                (
                    np.abs(mission["hinge_friction_torque_nm"]),
                    np.abs(mission["pad_closing_torque_nm"]),
                )
            ),
            (
                "힌지 저항",
                "패드 열림 토크" if manifest.get("valve_operation") == "open" else "패드 닫힘 토크",
            ),
            " N·m",
        ),
    ]
    stage_text = " → ".join(STAGES_KO.get(stage, stage) for stage in dict.fromkeys(stages.tolist()))
    return (
        "<!doctype html><html lang=ko><meta charset=utf-8><style>"
        "body{margin:0;background:#f7f8fa;font:13px Malgun Gothic,sans-serif;color:#172033}"
        "main{max-width:1200px;margin:auto;padding:28px;background:white}h1{font-size:22px}"
        ".scroll{overflow:auto}table{border-collapse:collapse}th{text-align:right;white-space:nowrap}"
        "th,td{height:15px;padding:0 4px}td{min-width:5px;border-right:1px solid white}"
        "</style><main><h1>RBQ 계단 밸브 임무 하중</h1>"
        f"<p>{html.escape(stage_text)}</p><p>각 칸은 0.5초 안의 최댓값입니다. "
        "장착부 6축은 RB1_Base 원점 기준이며 방향은 세계 좌표계를 따릅니다.</p>"
        f"{''.join(sections)}</main></html>"
    )


def _rgba(hex_colour: str) -> list[float]:
    return [int(hex_colour[index : index + 2], 16) / 255 for index in (1, 3, 5)] + [1.0]


def _heat_targets(model: mujoco.MjModel) -> list[list[int]]:
    targets = []
    for name in JOINT_NAMES:
        body = model.jnt_bodyid[model.joint(name).id]
        geoms = [index for index in range(model.ngeom) if model.geom_bodyid[index] == body]
        model.geom_matid[geoms] = -1
        targets.append(geoms)
    return targets


def _encoder(output: Path, subtitle: Path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subtitle_arg, output_arg = subtitle.as_posix(), output.as_posix()
        command = [ffmpeg]
    else:
        drive, tail = subtitle.resolve().as_posix().split(":", 1)
        subtitle_arg = f"/mnt/{drive.lower()}{tail}"
        drive, tail = output.resolve().as_posix().split(":", 1)
        output_arg = f"/mnt/{drive.lower()}{tail}"
        command = ["wsl.exe", "-e", "ffmpeg"]
    command += [
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-vf",
        f"subtitles={subtitle_arg}:fontsdir=/mnt/c/Windows/Fonts",
        "-c:v",
        "libx264",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        output_arg,
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def _clock(seconds: float) -> str:
    total = round(seconds * 100)
    hours, rest = divmod(total, 360000)
    minutes, rest = divmod(rest, 6000)
    return f"{hours}:{minutes:02d}:{rest // 100:02d}.{rest % 100:02d}"


def video_samples(times, start, end, speed):
    stamps = np.arange(start, end, speed / FPS)
    return np.minimum(np.searchsorted(times, stamps), len(times) - 1)


def rank_order(values: np.ndarray) -> np.ndarray:
    return np.argsort(-values, kind="stable")[:RANK_ROWS]


def _hud_events(prefix: str, hud: dict, index: int) -> list[str]:
    """Per-frame inset captions and the live ranking of the 18 link peaks."""
    values, where = hud["peaks"][index], hud["where"][index]
    lines = []
    for slot, link in enumerate(hud["insets"]):
        x, y = INSET_SLOTS[slot]
        label_y = y + INSET_H + 6
        place = location_label(link, hud["clip_where"][link])
        lines.append(
            f"{prefix},HudText,{{\\an7\\pos({x},{label_y})}}"
            f"{slot + 1} · {link_label(link)} · {place}"
        )
        lines.append(
            f"{prefix},HudMuted,{{\\an7\\pos({x},{label_y + 20})}}"
            f"지금 {values[link]:.1f}MPa · 이 영상 최대 {hud['clip_max'][link]:.1f}MPa"
        )
    for row, link in enumerate(rank_order(values)):
        y = RANK_TOP + row * RANK_STEP
        lines.append(
            f"{prefix},HudText,{{\\an7\\pos({INSET_SLOTS[0][0]},{y})}}"
            f"{link_label(link)} · {location_label(link, where[link])}"
        )
        lines.append(f"{prefix},HudText,{{\\an9\\pos({BAR_X1},{y})}}{values[link]:.1f}MPa")
    return lines


def _subtitles(
    path: Path,
    manifest: dict,
    mission: dict,
    start: float,
    end: float,
    speed: float,
    walking_metrics: bool = False,
    mount_stress=None,
    body_stress=None,
) -> None:
    times = mission["time_s"]
    stages = stage_labels(mission["stage"], manifest)
    ratio = torque_ratios(manifest, mission)
    foot = np.linalg.norm(mission["foot_surface_force_world_n"], axis=-1).max(axis=(1, 2))
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {WIDTH}",
        f"PlayResY: {HEIGHT}",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, "
            "Bold, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV"
        ),
        ("Style: Box,Malgun Gothic,25,&H00FFFFFF,&H00000000,&H60000000,0,3,2,0,7,24,24,20"),
        ("Style: LegendName,Malgun Gothic,16,&H00FFFFFF,&H00000000,&H60000000,0,3,1,0,9,0,0,0"),
        ("Style: LegendTick,Malgun Gothic,15,&H00FFFFFF,&H00000000,&H60000000,0,3,1,0,8,0,0,0"),
        ("Style: BoxSmall,Malgun Gothic,20,&H00FFFFFF,&H00000000,&H70000000,0,3,2,0,7,20,20,16"),
        *hud_styles(),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Text",
    ]
    extent = {
        "approach": "접근 구간",
        "through-task": "조작 구간 포함",
        "full": "전체 재생",
        "open-at-station": "밸브 개방 후 발판 정지",
    }
    title = extent.get(manifest.get("extent"), "물리 재생")
    if start > times[0] or end < times[-1]:
        title = "구간 발췌"
    for frame, index in enumerate(video_samples(times, start, end, speed)):
        stage = STAGES_KO.get(stages[index], stages[index])
        style = "Box"
        if body_stress is not None:
            style = "BoxSmall"
            values, where = body_stress["peaks"], body_stress["where"]
            link = int(np.argmax(values[index]))
            text = (
                f"{body_stress['title']} · {stage} · 원시각 {times[index]:.3f}s · {speed:g}×\\N"
                f"사지 가정 단면 최대 {JOINT_NAMES[link]} {values[index, link]:.2f}MPa"
                f" ({link_label(link)} {location_label(link, where[index, link])})"
                f" · 몸통 {body_stress['trunk'][index]:.2f}MPa\\N"
                f"발판 {foot[index]:.1f}N · 밸브 {mission['valve_angle_deg'][index]:.2f}°\\N"
                "몸통·다리·팔: 연결부 6축 반력으로 구한 단면 응력 분포 · 가정 단면 추정"
            )
        else:
            torque = np.abs(mission["torque_nm"][index]).max()
            bending = mission["joint_off_axis_bending_nm"][index].max()
            hinge = mission["hinge_friction_torque_nm"][index]
            pad = mission["pad_closing_torque_nm"][index]
            upper, lower = mission["pad_normal_force_n"][index]
            text = (
                f"{title} {end - start:.1f}초 · {speed:g}× 재생 · {stage} "
                f"· 원시각 {times[index]:.3f}s\\N"
                f"최대 축 토크 {torque:.1f}N·m  "
                f"최대 한계 사용률 {ratio[index].max() * 100:.1f}%\\N"
                f"최대 비축 굽힘 {bending:.1f}N·m\\N"
                f"발판 반력 {foot[index]:.1f}N  "
                f"밸브 {mission['valve_angle_deg'][index]:.2f}°\\N"
                f"힌지 {hinge:.2f}N·m  패드 {pad:.2f}N·m\\N"
                f"패드 법선력  위 {upper:.1f}N  아래 {lower:.1f}N"
            )
            if walking_metrics and "stance_slip_m" in mission:
                text += (
                    f"\\N지금까지 최대 지지발 미끄러짐 "
                    f"{mission['stance_slip_m'][index] * 1000:.3f}mm"
                )
            if walking_metrics and "unexpected_contact_force_n" in mission:
                text += f"\\N비허용 접촉력 {mission['unexpected_contact_force_n'][index]:.2f}N"
            if mount_stress is not None:
                values, scale = mount_stress
                text += (
                    f"\\N가정 장착판 등가응력 {values[index]:.2f}MPa "
                    f"· 장착점 색 0~{scale:g}MPa"
                )
        subtitle_start = _clock(frame / FPS)
        subtitle_end = _clock((frame + 1) / FPS)
        lines.append(f"Dialogue: 0,{subtitle_start},{subtitle_end},{style},{text}")
        if body_stress is not None:
            lines.extend(
                _hud_events(f"Dialogue: 2,{subtitle_start},{subtitle_end}", body_stress, index)
            )
    if body_stress is not None:
        finish = _clock((end - start) / speed)
        prefix = f"Dialogue: 1,0:00:00.00,{finish}"
        lines.extend(legend_events(prefix, body_stress["scale"]))
        lines.extend(panel_static_events(prefix))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_layers(renderer: mujoco.Renderer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    segmentation = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    renderer.enable_depth_rendering()
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    return frame, segmentation, depth


def _link_states(model, data, mission, indices, topology):
    """Kinematics-only pass: link and trunk end wrenches at every video sample."""
    states, trunks, peaks, where, trunk_peaks = {}, {}, {}, {}, {}
    for index in np.unique(indices):
        data.qpos[:] = mission["qpos"][index]
        mujoco.mj_kinematics(model, data)
        state = link_state(
            topology,
            data,
            mission["joint_force_world_n"][index],
            mission["joint_moment_world_nm"][index],
            mission["foot_surface_force_world_n"][index].sum(axis=1),
        )
        trunk = trunk_state(
            model,
            data,
            mission["joint_force_world_n"][index],
            mission["joint_moment_world_nm"][index],
            mission["rb1_mount_wrench_world"][index],
        )
        states[int(index)], trunks[int(index)] = state, trunk
        peaks[int(index)], where[int(index)] = link_peaks(state, topology)
        trunk_peaks[int(index)] = float(trunk_profile(trunk)[0].max())
    return states, trunks, peaks, where, trunk_peaks


def render_video(
    manifest: dict,
    mission: dict,
    output: Path,
    start: float,
    end: float,
    speed: float = 1.0,
    valve_view: bool = False,
    follow_base: bool = False,
    body_stress: bool = False,
) -> None:
    model, data = build_model(manifest, studio=True)
    if mission["qpos"].shape[1] != model.nq:
        raise ValueError("qpos width does not match scene model")
    model.vis.global_.offwidth, model.vis.global_.offheight = WIDTH, HEIGHT
    configure_visual(model)
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    times = mission["time_s"]
    samples = video_samples(times, start, end, speed)
    mount_stress = None
    hud = None
    targets = []
    base_geoms = []
    if body_stress:
        from stair_body_stress import MPA_SCALE, rainbow_rgb

        topology = link_topology(model)
        base_body = model.body("base").id
        base_geoms = [
            index for index in range(model.ngeom) if model.geom_bodyid[index] == base_body
        ]
        neutral = [geom for geoms in topology.geoms for geom in geoms] + base_geoms
        model.geom_matid[neutral] = -1
        model.geom_rgba[neutral] = (0.80, 0.81, 0.83, 1.0)
        for name in ("stringerL", "stringerR"):
            model.geom(name).rgba[3] = 0.16
        # The vendor-red lever would read as a stress hotspot; render it as plain steel.
        model.geom(stair_valve.VALVE_LEVER).rgba[:3] = (0.55, 0.57, 0.60)
        states, trunks, peak_map, where_map, trunk_map = _link_states(
            model, data, mission, samples, topology
        )
        peaks = np.zeros((len(times), len(JOINT_NAMES)))
        where = np.zeros_like(peaks)
        trunk_peaks = np.zeros(len(times))
        for index, value in peak_map.items():
            peaks[index], where[index] = value, where_map[index]
            trunk_peaks[index] = trunk_map[index]
        sampled = np.unique(samples)
        clip_peak_rows = sampled[np.argmax(peaks[sampled], axis=0)]
        clip_max = peaks[clip_peak_rows, np.arange(len(JOINT_NAMES))]
        clip_where = where[clip_peak_rows, np.arange(len(JOINT_NAMES))]
        hud = {
            "peaks": peaks,
            "where": where,
            "insets": [int(k) for k in np.argsort(-clip_max, kind="stable")[: len(INSET_SLOTS)]],
            "clip_max": clip_max,
            "clip_where": clip_where,
            "trunk": trunk_peaks,
            "scale": MPA_SCALE,
            "title": "50° 선박형 계단 밸브 개방",
        }
        inset_renderer = mujoco.Renderer(model, INSET_H, INSET_W)
        # Shadows in the small side views only showed acne speckles on the floor.
        inset_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

        def limb_colour(values, top=MPA_SCALE):
            return rainbow_rgb(values, top)

    elif manifest.get("valve_operation") == "open":
        from stair_stress_heatmap import add_mount_marker, stress_series

        from go2_wall.robot import load_parameters

        scale = float(load_parameters()["aluminum_6061_t6_yield_mpa"]["value"])
        mount_stress = (stress_series(mission)["von_mises_mpa"], scale)
        for name in ("stringerL", "stringerR"):
            model.geom(name).rgba[3] = 0.16
    else:
        targets = _heat_targets(model)
    subtitle = output.with_suffix(".ass")
    _subtitles(
        subtitle,
        manifest,
        mission,
        start,
        end,
        speed,
        walking_metrics=follow_base,
        mount_stress=mount_stress,
        body_stress=hud,
    )
    encoder = _encoder(output, subtitle)
    camera = mujoco.MjvCamera()
    settings = manifest["settings"]
    run = settings["rise_m"] / np.tan(np.radians(settings["slope_deg"])) * 10
    camera.lookat[:] = [run / 2, 0.0, settings["rise_m"] * 5]
    camera.distance, camera.azimuth, camera.elevation = max(3.0, run * 1.5), 50, -15
    if valve_view:
        camera.lookat[:] = np.asarray(manifest["centre_m"]) + [-0.04, 0.0, -0.05]
        camera.distance, camera.azimuth, camera.elevation = 0.64, 75, -14
    elif follow_base:
        camera.distance, camera.azimuth, camera.elevation = 2.45, 65, -20
    ratio = torque_ratios(manifest, mission)
    surface_checks = []
    frame_records = []
    shade_reference = None
    check_frames = set()
    if body_stress and follow_base:
        check_frames = {
            int(np.argmin(np.abs(times[samples] - target)))
            for target in (35.205, 66.242)
            if times[samples[0]] <= target <= times[samples[-1]]
        }
    for frame_number, index in enumerate(samples):
        data.qpos[:] = mission["qpos"][index]
        data.qvel[:] = mission["qvel"][index]
        if follow_base:
            camera.lookat[:] = data.qpos[:3]
        follow_sun(model, data.qpos[:3])
        for joint, geoms in enumerate(targets):
            model.geom_rgba[geoms] = _rgba(_colour(float(ratio[index, joint]), 1.0))
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        if mount_stress is not None:
            add_mount_marker(
                renderer.scene, data.body("RB1_Base").xpos, mount_stress[0][index], mount_stress[1]
            )
        if not body_stress:
            encoder.stdin.write(renderer.render().tobytes())
            continue
        frame, segmentation, depth = _render_layers(renderer)
        state, trunk = states[int(index)], trunks[int(index)]
        if shade_reference is None:
            robot = np.isin(segmentation[..., 0], neutral) & (segmentation[..., 1] == 5)
            shade_reference = float(np.percentile(frame[robot].mean(axis=1), 90))
        metrics = paint_trunk(
            frame,
            segmentation,
            depth,
            renderer.scene.camera[0],
            trunk,
            base_geoms,
            limb_colour,
            shade_reference,
        )
        paint_links(
            frame,
            segmentation,
            depth,
            renderer.scene.camera[0],
            state,
            topology,
            limb_colour,
            shade_reference,
        )
        if frame_number in check_frames:
            surface_checks.append({"time_s": float(times[index]), **metrics})
        frost_panel(frame)
        for slot, link in enumerate(hud["insets"]):
            # Centre between the peak position and the link midpoint so the whole link shows.
            share = 0.5 * (hud["clip_where"][link] + 0.5)
            point = state.p[link] + share * (state.c[link] - state.p[link])
            side = JOINT_NAMES[link][1] == "L" or JOINT_NAMES[link].startswith("arm")
            inset_renderer.update_scene(data, inset_camera(point, data.body("base").xmat, side))
            inset, inset_segmentation, inset_depth = _render_layers(inset_renderer)
            paint_trunk(
                inset,
                inset_segmentation,
                inset_depth,
                inset_renderer.scene.camera[0],
                trunk,
                base_geoms,
                limb_colour,
                shade_reference,
            )
            paint_links(
                inset,
                inset_segmentation,
                inset_depth,
                inset_renderer.scene.camera[0],
                state,
                topology,
                limb_colour,
                shade_reference,
            )
            paste_inset(frame, inset, slot, limb_colour(hud["peaks"][index, link]))
        order = rank_order(hud["peaks"][index])
        paint_rank_bars(
            frame, hud["peaks"][index, order], limb_colour, np.full(len(order), MPA_SCALE)
        )
        paint_legend(frame, limb_colour, hud["scale"])
        frame_records.append(
            {
                "time_s": float(times[index]),
                "links_mpa": [float(v) for v in hud["peaks"][index]],
                "peak_position": [round(float(v), 3) for v in hud["where"][index]],
                "trunk_mpa": float(hud["trunk"][index]),
            }
        )
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    code = encoder.wait()
    renderer.close()
    if body_stress:
        inset_renderer.close()
    if code:
        raise RuntimeError(f"ffmpeg exited with {code}")
    if body_stress:
        surface = {
            "method": (
                "MuJoCo geom segmentation plus depth reprojected onto the trunk axis; "
                "thin-wall section stress from the four hip and RB1 mount wrenches"
            ),
            "base_geom_ids": base_geoms,
            "added_overlay_geoms": 0,
            "samples": surface_checks,
        }
        if len(surface_checks) == 2:
            surface["visible_mean_change_mpa"] = abs(
                surface_checks[1]["visible_mean_mpa"]
                - surface_checks[0]["visible_mean_mpa"]
            )
        output.with_suffix(".surface.json").write_text(
            json.dumps(surface, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        links = {
            "schema": "rbq_link_surface_stress_v1",
            "method": (
                "per-pixel thin-wall von Mises from both end wrenches of each link, "
                "blended linearly along the proximal-to-distal segment; see "
                "tools/link_surface_stress.py"
            ),
            "joint_names": list(JOINT_NAMES),
            "trunk_method": (
                "base collision box 0.8 x 0.2 x 0.148 m with an assumed 3 mm wall; loads are the "
                "four hip reactions, the RB1 mount wrench and the trunk's own gravity and inertia"
            ),
            "scale_mpa": [0.0, float(hud["scale"])],
            "inset_links": [JOINT_NAMES[k] for k in hud["insets"]],
            "clip_max_mpa": [round(float(v), 4) for v in hud["clip_max"]],
            "frames": frame_records,
        }
        output.with_suffix(".links.json").write_text(
            json.dumps(links, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="directory with scene.json and mission.npz")
    parser.add_argument("output", type=Path, nargs="?", help="default: RUN_DIR/mission.mp4")
    parser.add_argument(
        "--start", type=float, help="first physical timestamp; default: first sample"
    )
    parser.add_argument("--end", type=float, help="last physical timestamp; default: last sample")
    parser.add_argument(
        "--no-video", action="store_true", help="write the heatmap, peaks and load table"
    )
    camera = parser.add_mutually_exclusive_group()
    camera.add_argument(
        "--valve-view", action="store_true", help="show the valve and both pads up close"
    )
    camera.add_argument(
        "--follow-base", action="store_true", help="keep the robot in frame from floor to platform"
    )
    parser.add_argument("--speed", type=float, default=1.0, help="video speed; default: 1")
    parser.add_argument(
        "--body-stress",
        action="store_true",
        help="colour chassis, leg and arm surfaces by conditional six-axis link stress",
    )
    args = parser.parse_args()
    manifest, mission = load_run(args.run_dir)
    start = float(mission["time_s"][0] if args.start is None else args.start)
    end = float(mission["time_s"][-1] if args.end is None else args.end)
    if not mission["time_s"][0] <= start < end <= mission["time_s"][-1]:
        parser.error("window must lie inside time_s and have start < end")
    if args.speed <= 0:
        parser.error("speed must be positive")
    output = args.output or args.run_dir / "mission.mp4"
    custom_window = args.start is not None or args.end is not None or args.output is not None
    peaks = output.with_suffix(".peaks.json") if custom_window else args.run_dir / "peaks.json"
    heatmap = (
        output.with_suffix(".heatmap.html") if custom_window else args.run_dir / "heatmap.html"
    )
    view = mission_window(mission, start, end)
    peaks.write_text(
        json.dumps(peak_summary(manifest, view), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    heatmap.write_text(heatmap_page(manifest, view), encoding="utf-8")
    loads = output.with_suffix(".loads.csv") if custom_window else args.run_dir / "loads.csv"
    with loads.open("w", encoding="utf-8", newline="") as stream:
        table = csv.writer(stream)
        table.writerow(
            ("quantity", "component", "unit", "value_at_absolute_peak", "time_s", "stage")
        )
        table.writerows(load_rows(manifest, view))
    if not args.no_video:
        render_video(
            manifest,
            mission,
            output,
            start,
            end,
            args.speed,
            args.valve_view,
            args.follow_base,
            args.body_stress,
        )


if __name__ == "__main__":
    main()
