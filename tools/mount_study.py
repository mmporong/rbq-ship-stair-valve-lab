"""Compare wall-braced and four-leg reach for several arm mount positions along the body."""

import json
import sys
from pathlib import Path

import numpy as np

from go2_wall.hold import max_static_push, render, run
from go2_wall.pose_search import four_leg_reach, pick, search

OUT = Path(sys.argv[1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for mount in (0.0, 0.05, 0.10, 0.15):
        found = search("scaled", 0.6, 0.6, 5.0, "derived", mount_x=mount)
        pose = pick(found, 0.5)
        row = {"solutions": len(found), "four_leg": four_leg_reach("scaled", mount)}
        heights = {}
        for item in found:
            key = item["pitch_deg"]
            heights[key] = max(heights.get(key, 0.0), item["tip_height_m"])
        row["height_by_pitch"] = {str(k): round(heights[k], 3) for k in sorted(heights)}
        if pose:
            row["pose"] = pose
            row["hold"] = run(pose, "scaled", 0.6, 0.6, 5.0, "derived", 5.0)
            row["push_limit"] = max_static_push(pose, "scaled", 0.6, 0.6, "derived")
            render(pose, "scaled", OUT / f"mount_{mount:.2f}.png")
        rows[f"{mount:.2f}"] = row
        print(mount, row["four_leg"], row["height_by_pitch"], flush=True)
        if pose:
            hold = row["hold"]
            print(
                "  braced",
                round(pose["tip_height_m"], 3),
                pose["pitch_deg"],
                round(pose["ratio"], 3),
                hold["passed"],
                round(np.float64(hold["max_ratio"]), 3),
                round(hold["max_foot_slip_mm"], 2),
                row["push_limit"],
                flush=True,
            )
    (OUT / "mount_study.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
