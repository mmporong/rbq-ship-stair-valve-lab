"""Run pose searches and physics holds over the graded parameter conditions."""

import json
import sys
from pathlib import Path

from go2_wall.hold import max_static_push, run
from go2_wall.pose_search import four_leg_reach, pick, search

OUT = Path(sys.argv[1])
SEARCHES = [
    ("scaled", 0.6, 0.6, 5.0, "derived"),
    ("servo_lumped", 0.6, 0.6, 5.0, "derived"),
    ("scaled", 0.6, 0.6, 5.0, "urdf"),
    ("scaled", 0.6, 0.6, 5.0, "osu"),
    ("scaled", 0.4, 0.4, 5.0, "derived"),
    ("scaled", 0.6, 0.3, 5.0, "derived"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    reference = None
    for case, floor, wall, push, arm in SEARCHES:
        found = search(case, floor, wall, push, arm)
        pose = pick(found, 0.5)
        row = {
            "case": case,
            "floor_mu": floor,
            "wall_mu": wall,
            "push_n": push,
            "arm": arm,
            "solutions": len(found),
        }
        if pose:
            result = run(pose, case, floor, wall, push, arm, 5.0)
            row.update(
                tip_height_m=pose["tip_height_m"],
                pitch_deg=pose["pitch_deg"],
                static_ratio=pose["ratio"],
                hold=result,
            )
            if reference is None:
                reference = pose
        rows.append(row)
        print(
            {k: v for k, v in row.items() if k != "hold"},
            row.get("hold", {}).get("passed"),
            flush=True,
        )
    variants = [
        ("mass_servo_lumped", dict(case="servo_lumped")),
        ("arm_limit_urdf", dict(arm="urdf")),
        ("arm_limit_osu", dict(arm="osu")),
        ("gains_half", dict(gains_scale=0.5)),
        ("gains_double", dict(gains_scale=2.0)),
        ("push_0", dict(push=0.0)),
        ("push_15", dict(push=15.0)),
        ("push_30", dict(push=30.0)),
        ("mu_0.4", dict(floor_mu=0.4, wall_mu=0.4)),
        ("mu_0.3", dict(floor_mu=0.3, wall_mu=0.3)),
        # Simulator settings from the MuJoCo 4.0 model tuning guide. Loads should not move.
        ("integrator_implicitfast", dict(physics={"integrator": "implicitfast"})),
        ("integrator_discrete", dict(physics={"integrator": "discrete"})),
        ("contact_solref_0.005", dict(physics={"contact_solref": 0.005})),
        ("impratio_10", dict(physics={"impratio": 10.0})),
        ("noslip_2", dict(physics={"noslip_iterations": 2})),
    ]
    fixed = {}
    for name, change in variants:
        settings = (
            dict(
                case="scaled",
                floor_mu=0.6,
                wall_mu=0.6,
                push=5.0,
                arm="derived",
                seconds=5.0,
                gains_scale=1.0,
            )
            | change
        )
        fixed[name] = run(reference, **settings)
        item = fixed[name]
        print(name, item["passed"], item.get("max_ratio"), item.get("max_foot_slip_mm"), flush=True)
    push_limits = {
        f"{case}/{arm}": max_static_push(reference, case, 0.6, 0.6, arm)
        for case in ("scaled", "servo_lumped")
        for arm in ("derived", "urdf", "osu")
    }
    (OUT / "matrix.json").write_text(
        json.dumps(
            {
                "searches": rows,
                "fixed_pose": fixed,
                "reference_pose": reference,
                "push_limits": push_limits,
                "four_leg_reach": {
                    case: four_leg_reach(case) for case in ("scaled", "servo_lumped")
                },
            },
            indent=1,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
