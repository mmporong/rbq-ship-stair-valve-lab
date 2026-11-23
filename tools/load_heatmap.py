"""Write an HTML page with load heatmaps over the mission.

Usage: python tools/load_heatmap.py <dir with loads.npz and loads.json> <out.html>

Rows are joints, columns are 0.5 s bins, and each cell is the largest value in its bin.
Torque ratio, bending moment, and bracket stress have different units, so each gets its
own chart and scale. Colour is one sequential blue ramp, light to dark.
"""

import html
import json
import sys
from pathlib import Path

import numpy as np

from go2_wall.statics import joint_names

BIN_S = 0.5
RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]  # fmt: skip
PHASES = ("벽 오르기", "레버 조작", "내려오기")
LABEL = {"hip": "고관절 hip", "thigh": "허벅지 thigh", "calf": "종아리 calf"}


def colour(value: float, top: float) -> str:
    """Sequential colour for a value on 0..top.

    >>> colour(0.0, 1.0), colour(1.0, 1.0)
    ('#cde2fb', '#0d366b')
    """
    share = min(max(value / top, 0.0), 1.0) if top > 0 else 0.0
    position = share * (len(RAMP) - 1)
    low = int(np.floor(position))
    high = min(low + 1, len(RAMP) - 1)
    mix = position - low
    a = np.array([int(RAMP[low][i : i + 2], 16) for i in (1, 3, 5)])
    b = np.array([int(RAMP[high][i : i + 2], 16) for i in (1, 3, 5)])
    rgb = np.round(a + (b - a) * mix).astype(int)
    return "#" + "".join(f"{v:02x}" for v in rgb)


def binned(times: np.ndarray, values: np.ndarray, end: float) -> np.ndarray:
    edges = np.arange(0, end + BIN_S, BIN_S)
    out = np.zeros((values.shape[1], len(edges) - 1))
    for k in range(len(edges) - 1):
        mask = (times >= edges[k]) & (times < edges[k + 1])
        if mask.any():
            out[:, k] = values[mask].max(axis=0)
    return out


def row_label(name: str) -> str:
    if name.startswith("arm_joint"):
        return f"팔 J{name[-1]}"
    leg, part = name.split("_")[:2]
    side = {"FL": "앞 왼", "FR": "앞 오른", "RL": "뒤 왼", "RR": "뒤 오른"}[leg]
    return f"{side} {LABEL[part]}"


def heatmap(title, note, rows, labels, top, unit, fmt, phases, cell_w=6, cell_h=18) -> str:
    left, top_pad = 130, 30
    width = left + rows.shape[1] * cell_w + 10
    height = top_pad + rows.shape[0] * cell_h + 34
    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">']
    start = 0
    for index, name in enumerate(PHASES):
        stop = start + int(np.sum(phases == index))
        x0, x1 = left + start * cell_w, left + stop * cell_w
        parts.append(
            f'<rect x="{x0}" y="4" width="{x1 - x0 - 2}" height="18" rx="4" class="band"/>'
            f'<text x="{(x0 + x1) / 2}" y="17" class="band-text">{name}</text>'
        )
        start = stop
    for r, label in enumerate(labels):
        y = top_pad + r * cell_h
        parts.append(f'<text x="{left - 8}" y="{y + cell_h * 0.68}" class="row">{label}</text>')
        for c, value in enumerate(rows[r]):
            tip = html.escape(
                f"{label} · {c * BIN_S:.1f}~{(c + 1) * BIN_S:.1f}s · {PHASES[int(phases[c])]} · "
                f"{fmt(value)}{unit}"
            )
            parts.append(
                f'<rect x="{left + c * cell_w}" y="{y}" width="{cell_w - 1}" height="{cell_h - 2}"'
                f' fill="{colour(value, top)}" data-tip="{tip}" tabindex="-1"/>'
            )
    axis_y = top_pad + rows.shape[0] * cell_h + 14
    for second in range(0, int(rows.shape[1] * BIN_S) + 1, 10):
        x = left + second / BIN_S * cell_w
        parts.append(f'<text x="{x}" y="{axis_y}" class="tick">{second}s</text>')
    parts.append("</svg>")
    stops = "".join(
        f'<stop offset="{i / (len(RAMP) - 1):.3f}" stop-color="{c}"/>' for i, c in enumerate(RAMP)
    )
    gradient_id = f"g{abs(hash(title)) % 10**6}"
    legend = (
        f'<svg width="260" height="34" class="legend"><defs><linearGradient id="{gradient_id}">'
        f'{stops}</linearGradient></defs><rect x="0" y="4" width="200" height="12" rx="3" '
        f'fill="url(#{gradient_id})"/><text x="0" y="30" class="tick">0</text>'
        f'<text x="200" y="30" class="tick end">{fmt(top)}{unit}</text></svg>'
    )
    return (
        f'<section><h2>{title}</h2><p class="note">{note}</p>{legend}'
        f'<div class="scroll">{"".join(parts)}</div></section>'
    )


def peak_table(names, labels, ratio, phases) -> str:
    head = "".join(f"<th>{p}</th>" for p in PHASES)
    body = []
    for j, label in enumerate(labels):
        cells = []
        for index in range(len(PHASES)):
            mask = phases == index
            cells.append(f"<td>{ratio[j][mask].max() * 100:.0f}%</td>")
        body.append(f"<tr><th>{label}</th>{''.join(cells)}</tr>")
    return (
        "<section><h2>표: 단계별 최대 토크 비율</h2><table><thead><tr><th>관절</th>"
        f"{head}</tr></thead><tbody>{''.join(body)}</tbody></table></section>"
    )


STYLE = """
body{margin:0;background:#f9f9f7;color:#0b0b0b;
font:14px/1.5 "Malgun Gothic","Pretendard",sans-serif}
main{max-width:1100px;margin:0 auto;padding:28px 24px;background:#fcfcfb}
h1{font-size:22px;margin:0 0 6px}h2{font-size:16px;margin:28px 0 4px}
.note,.lead{color:#52514e;margin:0 0 8px}.scroll{overflow-x:auto}
svg text{font-family:inherit}.row{font-size:11px;fill:#52514e;text-anchor:end}
.tick{font-size:10px;fill:#52514e}.tick.end{text-anchor:end}
.band{fill:#f0efec}.band-text{font-size:11px;fill:#52514e;text-anchor:middle}
rect[data-tip]:hover{stroke:#0b0b0b;stroke-width:1.5}
#tip{position:fixed;pointer-events:none;background:#fff;border:1px solid #d6d5d0;border-radius:6px;
padding:4px 8px;font-size:12px;box-shadow:0 2px 6px rgba(0,0,0,.08);display:none}
table{border-collapse:collapse;font-size:12px}
th,td{padding:3px 10px;border-bottom:1px solid #e8e7e3;text-align:right}
thead th,tbody th{text-align:left;color:#52514e;font-weight:600}
"""
SCRIPT = """
const tip=document.getElementById('tip');
document.querySelectorAll('rect[data-tip]').forEach(r=>{
 r.addEventListener('pointermove',e=>{tip.textContent=r.dataset.tip;tip.style.display='block';
  tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';});
 r.addEventListener('pointerleave',()=>{tip.style.display='none';});});
"""


def main() -> None:
    folder, out = Path(sys.argv[1]), Path(sys.argv[2])
    loads = np.load(folder / "loads.npz")
    times = loads["time"]
    end = float(times[-1])
    phase_bins = binned(times, loads["phase"][:, None].astype(float), end)[0].astype(int)
    names = joint_names()
    labels = [row_label(n) for n in names]
    ratio = binned(times, loads["torque_ratio"], end)
    bend_names = [str(n) for n in loads["joint_names"]]
    bend = binned(times, loads["joint_bend"], end)
    stress = binned(times, loads["bracket_mpa"][:, None], end)
    summary = json.loads((folder / "loads.json").read_text(encoding="utf-8"))
    sections = [
        heatmap(
            "관절 토크 비율",
            "토크 ÷ 한계. 다리는 Go2 모델 한계, 팔은 정격 하중 역산 한계(9.97N·m)다. "
            "공개 URDF 한계로 바꾸면 팔 J2·J3는 약 3배로 커진다.",
            ratio,
            labels,
            1.0,
            "",
            lambda v: f"{v * 100:.0f}%",
            phase_bins,
        ),
        heatmap(
            "다리 관절 굽힘 모멘트",
            "모터 축이 아닌 방향의 모멘트로, 베어링과 링크가 받는다.",
            bend,
            [row_label(n + "_joint") for n in bend_names],
            float(np.ceil(bend.max())),
            " N·m",
            lambda v: f"{v:.1f}",
            phase_bins,
        ),
        heatmap(
            "브래킷 판 응력",
            f"보 이론 상한(볼트선 외팔보). 6061-T6 항복 {summary['yield_mpa']}MPa, "
            f"최대 {summary['bracket']['peak_von_mises_mpa']:.1f}MPa.",
            stress,
            ["브래킷 판"],
            float(np.ceil(stress.max())),
            " MPa",
            lambda v: f"{v:.1f}",
            phase_bins,
        ),
        peak_table(names, labels, ratio, phase_bins),
    ]
    page = (
        '<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1"><style>{STYLE}</style>'
        "<title>Go2+D1 벽 짚기 임무 하중 히트맵</title></head><body><main>"
        "<h1>Go2+D1 벽 짚기 임무 하중 히트맵</h1>"
        f'<p class="lead">오르기, 레버 조작, 내려오기 {end:.0f}초. '
        f"칸 하나는 {BIN_S}초 구간의 최대값이며, "
        "칸에 마우스를 올리면 값이 나온다. 진할수록 크다.</p>"
        f'{"".join(sections)}<div id="tip"></div><script>{SCRIPT}</script></main></body></html>'
    )
    out.write_text(page, encoding="utf-8")
    print("written", out, len(page) // 1024, "KB")


if __name__ == "__main__":
    main()
