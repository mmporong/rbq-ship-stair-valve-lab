"""Translate a saved four-foot ascent cycle to the next treads for physical replay."""

import argparse
import json
from pathlib import Path

import numpy as np

from go2_wall.robot import LEGS
from go2_wall.stairs import going_of


def translate_base(base: list, offset: np.ndarray) -> list:
    return [*(np.asarray(base[:3]) + offset).tolist(), *base[3:]]


def repeat_cycle(
    sequence: dict, slope: float, rise: float, count: int, wrap_shift_s: float
) -> dict:
    moves = sequence["moves"]
    cycle = moves[-4:]
    if {move["leg"] for move in cycle} != set(LEGS):
        raise ValueError("The last four moves must advance one foot each")

    earlier = {move["leg"]: np.asarray(move["target_xyz_m"]) for move in moves[:-4]}
    stride = np.array([going_of(slope, rise), 0.0, rise])
    for move in cycle:
        leg = move["leg"]
        target = np.asarray(move["target_xyz_m"])
        if leg not in earlier or not np.allclose(target - earlier[leg], stride, rtol=0, atol=1e-9):
            raise ValueError(f"The saved cycle must advance {leg} by one tread")

    last = cycle[-1]["path"]
    previous = last.get("swing_bases", [last["swing_base"]])[-1]
    extension = []
    for number in range(1, count + 1):
        offset = number * stride
        for index, move in enumerate(cycle):
            target = np.asarray(move["target_xyz_m"]) + offset
            original = move["path"]
            base = translate_base(original["swing_base"], offset)
            path = {"start_base": previous, "swing_base": base}
            if "swing_bases" in original:
                path["swing_bases"] = [
                    translate_base(pose, offset) for pose in original["swing_bases"]
                ]
            shifted = {
                **move,
                "target_xyz_m": target.tolist(),
                "path": path,
            }
            if index == 0:
                shifted["shift_s"] = wrap_shift_s
            extension.append(shifted)
            previous = path.get("swing_bases", [base])[-1]
    return {**sequence, "moves": moves + extension}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("geometry", "plan", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--wrap-shift", type=float, required=True)
    args = parser.parse_args()
    if args.count < 1 or not np.isfinite(args.wrap_shift) or args.wrap_shift <= 0:
        parser.error("Count and shift duration must be positive")
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
    sequence = json.loads(args.plan.read_text(encoding="utf-8"))
    result = repeat_cycle(
        sequence, geometry["slope_deg"], geometry["rise_m"], args.count, args.wrap_shift
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
