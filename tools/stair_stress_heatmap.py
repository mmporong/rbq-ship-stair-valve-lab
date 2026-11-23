# ruff: noqa: E501
"""Render a rainbow stress report from a captured stair-valve mission.

The RBQ asset has no documented mounting section. This report applies the measured RB1
mount wrench to the older Go2+D1 bracket assumption as an explicitly conditional
sensitivity calculation. Other link reactions remain forces and moments.
"""

import argparse
import base64
import csv
import hashlib
import html
import json
import shutil
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

import numpy as np

from go2_wall.loads import bracket_section
from go2_wall.robot import load_parameters

BIN_S = 0.5
RAINBOW = (
    (0.00, (38, 70, 183)),
    (0.25, (0, 172, 193)),
    (0.50, (38, 177, 76)),
    (0.75, (255, 214, 0)),
    (1.00, (213, 45, 38)),
)
COMPONENT = "RBQ 장착부 하중을 받은 Go2+D1 가정 단면"
STAGE_KO = {
    "prewalk": "계단 접근",
    "position": "작업 자세 맞춤",
    "approach": "손 접근",
    "contact": "파지",
    "operate": "밸브 회전",
    "settle": "각도 안정화",
    "hold": "열린 각도 유지",
    "release": "손 놓기",
    "retract": "팔 회수",
    "station": "발판 정지",
}
EDGE_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rainbow_colour(value: float, maximum: float) -> str:
    """Map 0..maximum to blue, cyan, green, yellow, and red."""
    share = float(np.clip(value / maximum, 0.0, 1.0)) if maximum > 0 else 0.0
    for (low, first), (high, second) in zip(RAINBOW, RAINBOW[1:], strict=True):
        if share <= high:
            mix = (share - low) / (high - low)
            rgb = np.rint(np.asarray(first) + mix * (np.asarray(second) - first)).astype(int)
            return "#" + "".join(f"{channel:02x}" for channel in rgb)
    return "#d52d26"


def _base_rotations(qpos: np.ndarray) -> np.ndarray:
    if qpos.ndim != 2 or qpos.shape[1] < 7:
        raise ValueError("qpos must include a free-joint position and quaternion")
    quaternion = qpos[:, 3:7]
    norm = np.linalg.norm(quaternion, axis=1)
    if np.any(norm == 0):
        raise ValueError("base quaternion must be nonzero")
    w, x, y, z = (quaternion / norm[:, None]).T
    rotations = np.empty((len(qpos), 3, 3))
    rotations[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotations[:, 0, 1] = 2 * (x * y - w * z)
    rotations[:, 0, 2] = 2 * (x * z + w * y)
    rotations[:, 1, 0] = 2 * (x * y + w * z)
    rotations[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotations[:, 1, 2] = 2 * (y * z - w * x)
    rotations[:, 2, 0] = 2 * (x * z - w * y)
    rotations[:, 2, 1] = 2 * (y * z + w * x)
    rotations[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotations


def stress_series(mission: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Calculate mounting-plate stress from recorded wrenches without replaying physics."""
    required = {"time_s", "stage", "qpos", "rb1_mount_wrench_world"}
    missing = required.difference(mission)
    if missing:
        raise ValueError(f"mission is missing load channels: {sorted(missing)}")
    times = np.asarray(mission["time_s"], dtype=float)
    stages = np.asarray(mission["stage"], dtype=str)
    wrench = np.asarray(mission["rb1_mount_wrench_world"], dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("mission must contain finite time samples")
    if stages.shape != times.shape or wrench.shape != (len(times), 6):
        raise ValueError("time, stage, and mount-wrench shapes are inconsistent")
    if not np.all(np.isfinite(wrench)) or not np.all(np.diff(times) > 0):
        raise ValueError("mission loads must be finite and time must be monotonic")

    params = load_parameters()
    cantilever = params["bracket_bolt_line"]["value"]["cantilever_m"]
    rotations = _base_rotations(np.asarray(mission["qpos"], dtype=float))
    if len(rotations) != len(times) or not np.isfinite(rotations).all():
        raise ValueError("one finite base orientation is required for every load sample")
    base_force = np.einsum("nji,nj->ni", rotations, -wrench[:, :3])
    base_moment = np.einsum("nji,nj->ni", rotations, -wrench[:, 3:])
    lever = np.zeros_like(base_force)
    lever[:, 0] = cantilever
    bolt_moment = base_moment + np.cross(lever, base_force)
    section = bracket_section()
    vertical = np.abs(bolt_moment[:, 1]) / section["z_vertical_m3"] / 1e6
    lateral = np.abs(bolt_moment[:, 2]) / section["z_lateral_m3"] / 1e6
    torsion = np.abs(bolt_moment[:, 0]) / section["torsion_m3"] / 1e6
    von_mises = np.sqrt((vertical + lateral) ** 2 + 3 * torsion**2)
    return {
        "time_s": times,
        "stage": stages,
        "bending_vertical_mpa": vertical,
        "bending_lateral_mpa": lateral,
        "torsion_mpa": torsion,
        "von_mises_mpa": von_mises,
        "moment_bolt_line_nm": bolt_moment,
        "force_base_frame_n": base_force,
    }


def binned_peak(
    times: np.ndarray, values: np.ndarray, bin_s: float = BIN_S
) -> tuple[np.ndarray, np.ndarray]:
    if bin_s <= 0:
        raise ValueError("bin_s must be positive")
    start = np.floor(times[0] / bin_s) * bin_s
    count = max(1, int(np.floor((times[-1] - start) / bin_s)) + 1)
    peaks = np.zeros(count)
    indices = np.minimum(((times - start) / bin_s).astype(int), count - 1)
    np.maximum.at(peaks, indices, values)
    return start + np.arange(count) * bin_s, peaks


def bending_section(summary):
    """Absolute longitudinal bending stress over the assumed rectangular bolt section."""
    section = summary["section"]
    _, moment_y, moment_z = summary["peak"]["moment_bolt_line_nm"]
    y = np.linspace(-1, 1, 61)
    z = np.linspace(1, -1, 23)
    return (
        np.abs(
            moment_y * z[:, None] / section["z_vertical_m3"]
            - moment_z * y[None, :] / section["z_lateral_m3"]
        )
        / 1e6
    )


def section_svg(summary):
    field = bending_section(summary)
    maximum = float(field.max())
    rows, columns = field.shape
    cells = "".join(
        f'<rect x="{x}" y="{z}" width="1" height="1" fill="{rainbow_colour(value, max(maximum, 1e-12))}"/>'
        for z, row in enumerate(field)
        for x, value in enumerate(row)
    )
    return (
        f'<svg style="width:100%;height:100px" preserveAspectRatio="none" viewBox="0 0 {columns} {rows}" '
        f'shape-rendering="crispEdges" role="img" aria-label="볼트선 단면의 굽힘응력 분포">{cells}</svg>'
        f"<small>폭 80mm × 두께 6mm 단면을 확대한 그림 · 굽힘응력 절댓값 0~{maximum:.2f} MPa · "
        "파랑에서 빨강 순</small>"
    )


def safety_factor(value):
    return "∞" if value is None else f"{value:.2f}"


def _legend(name: str, maximum: float) -> str:
    ticks = "".join(
        f'<span style="left:{share:.0%}">{share * maximum:.1f}</span>'
        for share in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    return (
        f'<div class="legend" style="background:linear-gradient(90deg,'
        f'{",".join(f"#{r:02x}{g:02x}{b:02x} {share:.0%}" for share, (r, g, b) in RAINBOW)})"></div>'
        f'<div class="ticks">{ticks}</div><p class="unit">{html.escape(name)} · MPa</p>'
    )


def _timeline(times: np.ndarray, values: np.ndarray, maximum: float) -> str:
    cell_w = 4
    width = max(700, len(values) * cell_w)
    cells = []
    for index, (time_s, value) in enumerate(zip(times, values, strict=True)):
        tip = html.escape(f"{time_s:.1f}~{time_s + BIN_S:.1f}초 · {value:.3f}MPa")
        cells.append(
            f'<rect x="{index * cell_w}" y="0" width="{cell_w}" height="30" '
            f'fill="{rainbow_colour(float(value), maximum)}"><title>{tip}</title></rect>'
        )
    ticks = []
    for index in np.unique(np.linspace(0, len(times) - 1, min(6, len(times)), dtype=int)):
        anchor = "start" if index == 0 else "end" if index == len(times) - 1 else "middle"
        ticks.append(
            f'<text x="{index * cell_w}" y="47" text-anchor="{anchor}" fill="#48566d" '
            f'font-size="12">{times[index]:g}초</text>'
        )
    return f'<div class="scroll"><svg viewBox="0 0 {width} 50" width="{width}" height="50">{"".join(cells + ticks)}</svg></div>'


def peak_card_html(summary: dict) -> str:
    """Standalone card used for the PNG preview."""
    peak = summary["peak"]
    yield_mpa = summary["material"]["yield_mpa"]
    fixed_ticks = "".join(
        f"<span>{yield_mpa * share:.0f}{' MPa' if share == 1 else ''}</span>"
        for share in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    sensitivity = "".join(
        f"<tr><td>{row['thickness_mm']:.0f} mm</td><td>{row['peak_mpa']:.3f} MPa</td>"
        f"<td>{safety_factor(row['safety_factor_to_yield'])}</td></tr>"
        for row in summary["thickness_sensitivity"]
    )
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f7fb;color:#172033;font:20px/1.45 "Malgun Gothic",sans-serif}}
main{{width:1200px;min-height:800px;padding:32px 44px;background:#fff}}h1{{margin:0;font-size:35px}}h2{{font-size:24px;margin:12px 0}}.sub{{color:#596579;margin:4px 0 18px}}
.row{{display:grid;grid-template-columns:620px 1fr;gap:34px}}.panel{{border:2px solid #dfe5ee;border-radius:16px;padding:22px}}
.value{{font-size:48px;font-weight:800;margin:8px 0}}.stage{{color:#596579}}table{{border-collapse:collapse;width:100%;font-size:18px}}td,th{{padding:7px;border-bottom:1px solid #e3e8f0;text-align:right}}td:first-child,th:first-child{{text-align:left}}
.legend{{height:28px;border-radius:7px;background:linear-gradient(90deg,#2646b7,#00acc1,#26b14c,#ffd600,#d52d26)}}
.ticks{{display:flex;justify-content:space-between;font-size:15px;color:#596579}}.warning{{font-size:17px;color:#7a4218;background:#fff5df;border-radius:10px;padding:13px 16px;margin-top:18px}}
</style></head><body><main><h1>RBQ 장착부 하중의 가정 단면 추정 응력</h1>
<p class="sub">밸브 작업 기록 · 파랑 → 청록 → 초록 → 노랑 → 빨강 · 단위 MPa</p>
<div class="row"><section class="panel"><h2>피크 시각의 단면 굽힘응력</h2>{section_svg(summary)}<h2>굽힘·비틀림 등가응력 상한</h2>
<div class="value">{peak["von_mises_mpa"]:.3f} MPa</div><div class="stage">피크 {peak["time_s"]:.3f}초 · {html.escape(STAGE_KO.get(peak["stage"], peak["stage"]))}</div>
<h2>항복강도 고정 범례</h2><div class="legend"></div><div class="ticks">{fixed_ticks}</div></section>
<section class="panel"><h2>두께 민감도</h2><table><tr><th>가정 두께</th><th>피크</th><th>안전율</th></tr>{sensitivity}</table>
<p><b>기준 단면</b><br>폭 80 mm × 두께 6 mm<br>가정 외팔 길이 150 mm<br>6061-T6 가정 항복 276 MPa</p></section></div>
<p class="warning">RBQ 장착판의 치수와 재질은 미확인이다. RBQ 모델에서 측정한 장착부 힘·모멘트를 Go2+D1용 가정 단면에 적용해 굽힘·비틀림의 상한을 계산했다.</p>
</main></body></html>"""


def _browser_path(browser: Path | None) -> Path:
    if browser is None:
        discovered = (
            shutil.which("msedge") or shutil.which("chromium") or shutil.which("google-chrome")
        )
        browser = (
            Path(discovered)
            if discovered
            else next((path for path in EDGE_CANDIDATES if path.is_file()), None)
        )
    if browser is None:
        raise RuntimeError("A Chromium browser is required to render the PNG preview")
    return browser


def _render_html_png(page_html: str, output: Path, browser: Path | None = None) -> Path:
    """Render one fixed-size HTML card through an installed Chromium browser."""
    browser = _browser_path(browser)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stair-stress-") as temporary:
        temp = Path(temporary)
        page = temp / "peak.html"
        page.write_text(page_html, encoding="utf-8")
        completed = subprocess.run(
            [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--user-data-dir={temp / 'browser'}",
                "--window-size=1200,800",
                f"--screenshot={output}",
                page.resolve().as_uri(),
            ],
            check=False,
            capture_output=True,
            timeout=30,
        )
    if completed.returncode or not output.is_file():
        error = completed.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"PNG rendering failed with code {completed.returncode}: {error}")
    return output


def render_peak_png(summary: dict, output: Path, browser: Path | None = None) -> Path:
    return _render_html_png(peak_card_html(summary), output, browser)


def _write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    """Write an RGB uint8 array as PNG with only the standard library."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("PNG input must be an H x W x 3 uint8 array")

    def chunk(name: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(name + payload)
        return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", checksum)

    height, width = rgb.shape[:2]
    scanlines = b"".join(b"\0" + row.tobytes() for row in rgb)
    raw = b"\x89PNG\r\n\x1a\n"
    raw += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    raw += chunk(b"IDAT", zlib.compress(scanlines, 6))
    raw += chunk(b"IEND", b"")
    path.write_bytes(raw)


def pose_card_html(summary: dict, scene_png: Path) -> str:
    encoded = base64.b64encode(scene_png.read_bytes()).decode("ascii")
    peak = summary["peak"]
    yield_mpa = summary["material"]["yield_mpa"]
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><style>
*{{box-sizing:border-box}}body{{margin:0;background:#eef2f7;color:#172033;font:19px/1.45 "Malgun Gothic",sans-serif}}
main{{width:1200px;height:800px;background:#fff;padding:28px 38px}}h1{{margin:0;font-size:31px}}.sub{{margin:3px 0 15px;color:#596579}}
.scene{{width:1124px;height:500px;object-fit:cover;border:2px solid #dfe5ee;border-radius:12px}}
.legend{{height:25px;border-radius:6px;background:linear-gradient(90deg,#2646b7,#00acc1,#26b14c,#ffd600,#d52d26)}}
.ticks{{display:flex;justify-content:space-between;font-size:14px;color:#596579}}.footer{{display:grid;grid-template-columns:1fr 2fr;gap:22px;align-items:center;margin-top:9px}}
.value{{font-size:28px;font-weight:800}}.note{{font-size:16px;color:#7a4218;background:#fff5df;border-radius:9px;padding:9px 13px}}
</style></head><body><main><h1>피크 응력 시점의 RBQ-10+RB1 자세</h1>
<p class="sub">{peak["time_s"]:.3f}초 · {html.escape(STAGE_KO.get(peak["stage"], peak["stage"]))} · 색 구체는 모델의 RB1_Base 원점</p>
<img class="scene" src="data:image/png;base64,{encoded}"><div class="footer"><div class="value">{peak["von_mises_mpa"]:.3f} MPa</div>
<div><div class="legend"></div><div class="ticks"><span>0</span><span>{yield_mpa * 0.25:.0f}</span><span>{yield_mpa * 0.5:.0f}</span><span>{yield_mpa * 0.75:.0f}</span><span>{yield_mpa:.0f} MPa</span></div></div></div>
<p class="note">로봇과 계단은 원래 색을 사용하고 측면 지지대는 반투명하게 표시했다. 장착점의 색은 Go2+D1 가정 단면에서 추정한 응력이다.</p>
</main></body></html>"""


def add_mount_marker(scene, world_position, stress_mpa, scale_mpa):
    import mujoco

    colour = rainbow_colour(stress_mpa, scale_mpa)
    rgba = [int(colour[index : index + 2], 16) / 255 for index in (1, 3, 5)] + [1.0]
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.055, 0.0, 0.0]),
        world_position,
        np.eye(3).ravel(),
        np.asarray(rgba),
    )
    scene.ngeom += 1


def render_pose_png(
    run_dir: Path, summary: dict, output: Path, browser: Path | None = None
) -> Path:
    """Render the peak MuJoCo pose with one stress-coloured marker at RB1_Base."""
    import mujoco
    from render_stair_mission import build_model

    manifest = json.loads((run_dir / "scene.json").read_text(encoding="utf-8"))
    with np.load(run_dir / "mission.npz") as raw:
        sample = summary["peak"]["sample_index"]
        qpos, qvel = raw["qpos"][sample], raw["qvel"][sample]
        feet = raw["feet_xyz_m"][sample]
    model, data = build_model(manifest)
    if len(qpos) != model.nq or len(qvel) != model.nv:
        raise ValueError("peak pose does not match the reconstructed scene")
    data.qpos[:], data.qvel[:] = qpos, qvel
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    landmarks = np.vstack((feet, data.body("RB1_Base").xpos, data.site("arm_tip").xpos))
    camera.lookat[:] = (landmarks.min(axis=0) + landmarks.max(axis=0)) / 2
    camera.distance, camera.azimuth, camera.elevation = 2.65, 65, -20
    for name in ("stringerL", "stringerR"):
        model.geom(name).rgba[3] = 0.16
    model.vis.headlight.ambient[:] = [0.55, 0.55, 0.55]
    model.vis.global_.offwidth, model.vis.global_.offheight = 1124, 590
    with mujoco.Renderer(model, height=590, width=1124) as renderer:
        renderer.update_scene(data, camera)
        add_mount_marker(
            renderer.scene,
            data.body("RB1_Base").xpos,
            summary["peak"]["von_mises_mpa"],
            summary["material"]["yield_mpa"],
        )
        frame = renderer.render()
    with tempfile.TemporaryDirectory(prefix="stair-pose-") as temporary:
        scene_png = Path(temporary) / "scene.png"
        _write_rgb_png(scene_png, frame)
        return _render_html_png(pose_card_html(summary, scene_png), output, browser)


def report_html(summary: dict, series: dict[str, np.ndarray]) -> str:
    times, peaks = binned_peak(series["time_s"], series["von_mises_mpa"])
    yield_mpa = summary["material"]["yield_mpa"]
    detail_max = summary["scales_mpa"]["detail_max"]
    peak_time = summary["peak"]["time_s"]
    stage = summary["peak"]["stage"]
    peak_stage = html.escape(STAGE_KO.get(stage, stage))
    section = summary["section"]
    stages = "".join(
        f"<tr><td>{html.escape(STAGE_KO.get(row['stage'], row['stage']))}</td>"
        f"<td>{row['peak_mpa']:.3f}</td><td>{row['time_s']:.3f}</td></tr>"
        for row in summary["stage_peaks"]
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RBQ-10+RB1 밸브 작업 추정 응력</title><style>
body{{margin:0;background:#f5f7fb;color:#172033;font:14px/1.55 "Malgun Gothic",sans-serif}}
main{{max-width:1120px;margin:auto;padding:28px}}section{{background:#fff;border:1px solid #dfe5ee;border-radius:12px;padding:18px;margin:14px 0}}
h1{{font-size:24px;margin:0 0 6px}}h2{{font-size:17px;margin:0 0 10px}}.lead,.note{{color:#566176}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.legend{{height:18px;border-radius:4px}}.ticks{{height:20px;position:relative}}.ticks span{{position:absolute;transform:translateX(-50%);font-size:11px}}.ticks span:first-child{{transform:none}}.ticks span:last-child{{transform:translateX(-100%)}}
.unit{{margin:0;color:#566176}}.scroll{{overflow-x:auto;border-radius:5px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:5px 8px;border-bottom:1px solid #e8ecf2;text-align:right}}th:first-child,td:first-child{{text-align:left}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr}}main{{padding:16px}}}}
</style></head><body><main><h1>RBQ-10+RB1 밸브 작업 응력 히트맵</h1>
<p class="lead">무지개색은 낮은 응력의 파랑에서 높은 응력의 빨강으로 이어진다. RBQ 모델은 장착 단면을 제공하지 않으므로, 측정한 RB1 장착부 하중을 과거 Go2+D1용 가정 단면에 적용한 조건부 추정이다.</p>
<div class="grid"><section><h2>항복강도 고정 범례</h2>{_legend("0부터 6061-T6 가정 항복강도", yield_mpa)}</section>
<section><h2>관측 구간 확대 범례</h2>{_legend("0부터 관측 상세 상한", detail_max)}</section></div>
<section><h2>피크 시각의 볼트선 단면 굽힘응력</h2>{section_svg(summary)}
<p class="note">{peak_time:.3f}초 · {peak_stage}. 단면의 중립축에서 굽힘응력이 0에 가까워지고 가장자리로 갈수록 커진다.
같은 시각 굽힘·비틀림 등가응력 상한은 {summary["peak"]["von_mises_mpa"]:.3f} MPa이다.</p></section>
<section><h2>시간축 응력 · 고정 0~{yield_mpa:.0f}MPa</h2>{_timeline(times, peaks, yield_mpa)}{_legend("고정 범례", yield_mpa)}</section>
<section><h2>시간축 응력 · 확대 0~{detail_max:.1f}MPa</h2>{_timeline(times, peaks, detail_max)}{_legend("관측 상세 범례", detail_max)}</section>
<section><h2>동작별 최대 등가응력</h2><table><tr><th>동작</th><th>최대 상한 (MPa)</th><th>시각 (초)</th></tr>{stages}</table></section>
<section><h2>피크 성분</h2><table><tr><th>성분</th><th>값</th></tr>
<tr><td>von Mises 상한</td><td>{summary["peak"]["von_mises_mpa"]:.3f}MPa</td></tr>
<tr><td>수직 굽힘</td><td>{summary["peak"]["bending_vertical_mpa"]:.3f}MPa</td></tr>
<tr><td>횡 굽힘</td><td>{summary["peak"]["bending_lateral_mpa"]:.3f}MPa</td></tr>
<tr><td>비틀림 전단</td><td>{summary["peak"]["torsion_mpa"]:.3f}MPa</td></tr>
<tr><td>항복 대비 안전율</td><td>{safety_factor(summary["peak"]["safety_factor_to_yield"])}</td></tr></table></section>
<section><h2>두께 민감도</h2><table><tr><th>가정 단면</th><th>피크 응력</th><th>항복 대비 안전율</th></tr>
{"".join(f"<tr><td>폭 80mm × 두께 {row['thickness_mm']:.0f}mm</td><td>{row['peak_mpa']:.3f}MPa</td><td>{safety_factor(row['safety_factor_to_yield'])}</td></tr>" for row in summary["thickness_sensitivity"])}</table></section>
<section><h2>모델 범위와 가정</h2><p>기준 단면은 폭 {section["width_m"] * 1000:.0f}mm, 두께 {section["thickness_m"] * 1000:.0f}mm이며 팔 축은 가정 볼트선에서 {summary["section"]["cantilever_m"] * 1000:.0f}mm 앞에 있다. 형상과 6061-T6 재질은 Go2+D1 연구의 가정을 사용했다. 색으로 나타낸 값은 이 단면의 보 이론 추정치다. 실제 장착판의 안전성을 판단하려면 부품 도면과 재질을 확인하고, 볼트 구멍의 응력 집중과 체결부·용접부를 별도로 평가해야 한다.</p></section>
</main></body></html>"""


def build_report(
    mission_path: Path, output_html: Path, output_json: Path, output_csv: Path
) -> dict:
    with np.load(mission_path) as raw:
        required = ("time_s", "stage", "qpos", "rb1_mount_wrench_world")
        series = stress_series({name: raw[name] for name in required if name in raw.files})
    stress = series["von_mises_mpa"]
    peak = int(np.argmax(stress))
    params = load_parameters()
    section = bracket_section()
    yield_mpa = float(params["aluminum_6061_t6_yield_mpa"]["value"])
    detail_max = max(1.0, float(np.ceil(stress[peak] / 5.0) * 5.0))
    sensitivity = []
    reference_thickness = section["thickness_m"]
    stage_peaks = []
    for stage in dict.fromkeys(series["stage"].tolist()):
        indices = np.flatnonzero(series["stage"] == stage)
        index = indices[np.argmax(stress[indices])]
        stage_peaks.append(
            {
                "stage": stage,
                "peak_mpa": float(stress[index]),
                "time_s": float(series["time_s"][index]),
            }
        )
    for thickness_mm in (4.0, 6.0, 8.0):
        ratio = reference_thickness / (thickness_mm / 1000.0)
        vertical = series["bending_vertical_mpa"] * ratio**2
        lateral = series["bending_lateral_mpa"] * ratio
        torsion = series["torsion_mpa"] * ratio**2
        trace = np.sqrt((vertical + lateral) ** 2 + 3 * torsion**2)
        peak_row = int(np.argmax(trace))
        estimate = float(trace[peak_row])
        sensitivity.append(
            {
                "thickness_mm": thickness_mm,
                "peak_mpa": estimate,
                "time_s": float(series["time_s"][peak_row]),
                "safety_factor_to_yield": yield_mpa / estimate if estimate else None,
            }
        )
    summary = {
        "schema": "rbq_rb1_bracket_stress_v1",
        "mission": {
            "filename": mission_path.name,
            "sha256": _file_sha256(mission_path),
            "samples": len(stress),
        },
        "component": COMPONENT,
        "stress_model": "conditional cantilever-section upper bound using an older Go2+D1 assumption",
        "provenance_warning": "The RBQ model provides a mount wrench but no mounting section or material. The 80 x 6 mm section, 150 mm cantilever, and 6061-T6 material come from the Go2+D1 study and are not RBQ component specifications.",
        "section": {
            **section,
            "cantilever_m": float(params["bracket_bolt_line"]["value"]["cantilever_m"]),
            "grade": params["front_bracket"]["grade"],
        },
        "material": {
            "name": "6061-T6 aluminium plate (assumed)",
            "yield_mpa": yield_mpa,
            "grade": params["aluminum_6061_t6_yield_mpa"]["grade"],
        },
        "peak": {
            "sample_index": peak,
            "time_s": float(series["time_s"][peak]),
            "stage": str(series["stage"][peak]),
            "von_mises_mpa": float(stress[peak]),
            "bending_vertical_mpa": float(series["bending_vertical_mpa"][peak]),
            "bending_lateral_mpa": float(series["bending_lateral_mpa"][peak]),
            "torsion_mpa": float(series["torsion_mpa"][peak]),
            "safety_factor_to_yield": yield_mpa / float(stress[peak]) if stress[peak] else None,
            "force_base_frame_n": series["force_base_frame_n"][peak].tolist(),
            "moment_bolt_line_nm": series["moment_bolt_line_nm"][peak].tolist(),
        },
        "stage_peaks": stage_peaks,
        "scales_mpa": {
            "fixed_min": 0.0,
            "fixed_max": yield_mpa,
            "detail_min": 0.0,
            "detail_max": detail_max,
        },
        "thickness_sensitivity": sensitivity,
        "calculation_check": {
            "load_n": 10.0,
            "cantilever_m": 0.15,
            "moment_nm": 1.5,
            "vertical_section_modulus_m3": section["z_vertical_m3"],
            "expected_vertical_bending_mpa": 3.125,
            "formula": "(10 N * 0.15 m) / (4.8e-7 m^3) / 1e6 = 3.125 MPa",
        },
        "excluded": [
            "uniform axial stress and transverse shear",
            "bolt-hole stress concentration",
            "fastener and weld stress",
            "plate deflection",
            "leg-link stress",
            "arm-link stress",
        ],
    }
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_html.write_text(report_html(summary, series), encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "time_s",
                "stage",
                "bending_vertical_mpa",
                "bending_lateral_mpa",
                "torsion_mpa",
                "von_mises_mpa",
            )
        )
        writer.writerows(
            zip(
                *(
                    series[name]
                    for name in (
                        "time_s",
                        "stage",
                        "bending_vertical_mpa",
                        "bending_lateral_mpa",
                        "torsion_mpa",
                        "von_mises_mpa",
                    )
                ),
                strict=True,
            )
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mission", type=Path, help="mission.npz recorded with --capture-loads")
    parser.add_argument("output", type=Path, help="HTML report path")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".html":
        parser.error("output must end in .html")
    summary = build_report(
        args.mission, args.output, args.output.with_suffix(".json"), args.output.with_suffix(".csv")
    )
    peak_png = render_peak_png(summary, args.output.with_suffix(".png"))
    pose_png = render_pose_png(
        args.mission.parent, summary, args.output.with_name(args.output.stem + "-pose.png")
    )
    summary["artifacts"] = {
        peak_png.name: _file_sha256(peak_png),
        pose_png.name: _file_sha256(pose_png),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["peak"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
