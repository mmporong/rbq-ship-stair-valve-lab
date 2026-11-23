"""Recheck both limiting 60-degree planning samples using the recorded selected posture."""

import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "src")]
from stair_mission_route import route, sample_walk  # noqa: E402

from go2_wall import stairs, transition  # noqa: E402


def selected_plan(original, result):
    plan = copy.deepcopy(original)

    def pitched(pose):
        value = list(pose)
        if value[0] >= -0.35:
            value[3] += np.deg2rad(10.0)
        return value

    for segment in plan:
        for key in ("base", "from", "to"):
            if key in segment:
                segment[key] = pitched(segment[key])
        if segment.get("swing_bases"):
            segment["swing_bases"] = [pitched(pose) for pose in segment["swing_bases"]]
    crossing = plan[36]
    original_end = crossing["to"].copy()
    high = original_end.copy()
    high[3] += np.deg2rad(5.0)
    crossing["to"] = high
    pose = result["diagnosed_geometry_rule"]["best_settle_pose"]
    settle = [*pose["base_xyz_m"], np.deg2rad(pose["pitch_deg"])]
    plan.insert(
        37,
        {
            "kind": "shift",
            "from": high,
            "to": settle,
            "stance": copy.deepcopy(crossing["stance"]),
            "seconds": crossing["seconds"],
        },
    )
    for index, segment in enumerate(plan[38:], start=38):
        changed = False
        for key in ("base", "from"):
            if key in segment and np.allclose(segment[key], original_end):
                segment[key] = settle
                changed = True
        if not changed and index > 42:
            break
    plan[39]["swing_curve"] = stairs.swing_over_tread
    return plan


def first_violation(geometry, plan):
    try:
        sample_walk(geometry, plan)
    except transition.InfeasibleSegment as error:
        return error.details
    raise AssertionError("Recorded planning failure was not reproduced")


def main():
    result = json.loads((HERE / "report.json").read_text(encoding="utf-8"))
    geometry, _, _, original, _, _ = route(60.0, 0.19)
    frozen = first_violation(geometry, original)
    reference = result["frozen_transform_first_violation"]
    assert frozen["segment"] == reference["segment"]
    assert np.isclose(frozen["planned_time_s"], reference["planned_time_s"], atol=1e-9, rtol=0)
    contact = max(
        (c for c in frozen["unexpected_overlaps"] if set(c["geoms"]) == set(reference["geoms"])),
        key=lambda c: c["penetration_m"],
    )
    np.testing.assert_allclose(
        contact["penetration_m"], reference["penetration_m"], atol=1e-9, rtol=0
    )
    altered = first_violation(geometry, selected_plan(original, result))
    target = result["best_connected_candidate"]
    assert altered["segment"] == target["segment"]
    assert not altered["unexpected_overlaps"]
    np.testing.assert_allclose(
        altered["planned_time_s"], target["planned_time_s"], atol=1e-9, rtol=0
    )
    np.testing.assert_allclose(
        max(altered["foot_ik_error_m"].values()), target["max_foot_ik_error_m"], atol=1e-8, rtol=0
    )
    print(
        frozen["planned_time_s"],
        contact["penetration_m"],
        altered["planned_time_s"],
        max(altered["foot_ik_error_m"].values()),
    )


if __name__ == "__main__":
    main()
