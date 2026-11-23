"""Render the mission log to an MP4 with load captions, using ffmpeg inside WSL.

Usage: python tools/render_mission.py <dir with mission.npz and loads.npz> <out.mp4> [--heat]

With --heat each link is coloured by the torque ratio of the joint that drives it and the
bracket plate by its stress, on the same blue ramp as tools/load_heatmap.py.

Frames come from the logged states at 25 fps. Captions are an ASS subtitle file burned in by
ffmpeg (libass) with the Windows Malgun Gothic font, since WSL has no Korean font installed.
"""

import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from load_heatmap import colour

from go2_wall import lever as lv
from go2_wall.statics import joint_names

WIDTH, HEIGHT, FPS = 1280, 720, 25
PHASE_NAMES = ("벽 오르기", "레버 조작", "내려오기")


def wsl_path(path: Path) -> str:
    """Translate a Windows path to its /mnt path in WSL.

    >>> wsl_path(Path("C:/Users/x/a.mp4"))
    '/mnt/c/Users/x/a.mp4'
    """
    text = path.resolve().as_posix() if path.drive == "" else path.as_posix()
    drive, rest = text.split(":", 1)
    return f"/mnt/{drive.lower()}{rest}"


def clock(seconds: float) -> str:
    """ASS time stamp.

    >>> clock(61.25)
    '0:01:01.25'
    """
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{int(hours)}:{int(minutes):02d}:{secs:05.2f}"


def heat_targets(model) -> tuple[list[tuple[int, list[int]]], list[int]]:
    """Visual geoms of each joint's body, keyed by its index in joint_names(), and the plate."""
    names = joint_names()
    pairs = []
    for index, name in enumerate(names):
        body = model.jnt_bodyid[model.joint(name).id]
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
        pairs.append((index, geoms))
    plate = [model.geom("bracket_plate").id]
    for _, geoms in pairs:
        model.geom_matid[geoms] = -1
    model.geom_matid[plate] = -1
    return pairs, plate


def rgba(hex_colour: str) -> list[float]:
    return [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)] + [1.0]


def captions(mission: dict, loads: dict, path: Path, heat: bool = False) -> None:
    names = joint_names()
    short = {n: n.replace("_joint", "").replace("arm_", "팔 ") for n in names}
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {WIDTH}",
        f"PlayResY: {HEIGHT}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV",
        "Style: Box,Malgun Gothic,26,&H00202020,&H00FFFFFF,&H40FFFFFF,0,3,8,0,7,30,30,24",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Text",
    ]
    times = mission["log_time"]
    torque = np.abs(mission["log_torque"])
    force = mission["log_force"]
    angle = np.degrees(mission["log_extra"])
    ratio = loads["torque_ratio"]
    stress = loads["bracket_mpa"]
    phase = loads["phase"]
    step = 0.5
    for start in np.arange(0, times[-1], step):
        mask = (times >= start) & (times < start + step)
        if not mask.any():
            continue
        k = np.where(mask)[0][-1]
        worst = int(np.argmax(ratio[mask].max(axis=0)))
        calf = max(torque[mask][:, names.index("RL_calf_joint")].max(), 0)
        wall = np.abs(force[mask][:, :2, 0]).max()
        text = (
            f"{PHASE_NAMES[int(phase[k])]}   {times[k]:5.1f} s\\N"
            f"최대 관절 부하 {ratio[mask].max() * 100:4.0f}%  ({short[names[worst]]})\\N"
            f"뒷다리 종아리 토크 {calf:5.1f} Nm   앞발 벽 반력 {wall:5.1f} N\\N"
            f"레버 {angle[k]:4.1f}°   브래킷 응력 {stress[mask].max():4.1f} MPa"
        )
        if heat:
            text += "\\N몸체 색: 관절 토크 비율 (연함 0% → 진함 100%)"
        lines.append(f"Dialogue: 0,{clock(start)},{clock(start + step)},Box,{text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    folder, out = Path(sys.argv[1]), Path(sys.argv[2])
    heat = "--heat" in sys.argv[3:]
    mission = dict(np.load(folder / "mission.npz"))
    loads = dict(np.load(folder / "loads.npz"))
    subtitle = folder / ("captions_heat.ass" if heat else "captions.ass")
    captions(mission, loads, subtitle, heat)
    model, data = lv.build_scene()
    model.vis.global_.offwidth, model.vis.global_.offheight = WIDTH, HEIGHT
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.65, 0.65, 0.65]
    model.geom_rgba[model.geom("floor").id] = [0.82, 0.83, 0.85, 1]
    model.geom_rgba[model.geom("wall").id] = [0.93, 0.94, 0.95, 1]
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    if heat:
        pairs, plate = heat_targets(model)
        stress_top = float(np.ceil(loads["bracket_mpa"].max()))
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.42, 0.0, 0.62]
    camera.distance, camera.azimuth, camera.elevation = 2.3, 50, -15
    command = [
        "wsl.exe",
        "-e",
        "ffmpeg",
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
        f"subtitles={wsl_path(subtitle)}:fontsdir=/mnt/c/Windows/Fonts",
        "-c:v",
        "libx264",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        wsl_path(out),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    times = mission["log_time"]
    for t in np.arange(0, times[-1], 1 / FPS):
        k = min(int(np.searchsorted(times, t)), len(times) - 1)
        data.qpos[:] = mission["log_qpos"][k]
        if heat:
            for index, geoms in pairs:
                model.geom_rgba[geoms] = rgba(colour(loads["torque_ratio"][k][index], 1.0))
            model.geom_rgba[plate] = rgba(colour(loads["bracket_mpa"][k], stress_top))
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        encoder.stdin.write(renderer.render().tobytes())
    encoder.stdin.close()
    encoder.wait()
    renderer.close()
    print("written", out, "exit", encoder.returncode)


if __name__ == "__main__":
    main()
