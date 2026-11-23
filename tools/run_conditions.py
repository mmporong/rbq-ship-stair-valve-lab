"""S8 (M4): run the whole mission under changed conditions with the nominal plan.

Usage: python tools/run_conditions.py <plan.json> <out dir>

The evaluation is fixed before any run: a condition succeeds when the round trip passes the
M2 checks (no fall, no saturation, base tracking <= 30 mm, return <= 20 mm, all feet down,
stance sliding <= 10 mm) and the lever passes the M3 checks (handle still before contact,
>= 40 deg reached, >= 38 deg kept, body held as in M1). The plan is never re-made: only the
true world changes, so the result shows how far the nominal plan and controller carry.
"""

import json
import multiprocessing
import sys
from pathlib import Path

from go2_wall import lever as lv

CONDITIONS = [
    ("nominal", {}),
    ("mu_0.4", {"floor_mu": 0.4, "wall_mu": 0.4}),
    ("mu_0.5", {"floor_mu": 0.5, "wall_mu": 0.5}),
    ("mu_0.8", {"floor_mu": 0.8, "wall_mu": 0.8}),
    ("wall_-20mm", {"wall_shift": -0.02}),
    ("wall_-10mm", {"wall_shift": -0.01}),
    ("wall_+10mm", {"wall_shift": 0.01}),
    ("wall_+20mm", {"wall_shift": 0.02}),
    ("lever_1.25m", {"lever": {"hinge_height_m": 1.25}}),
    ("lever_1.31m", {"lever": {"hinge_height_m": 1.31}}),
    ("force_3N", {"lever": {"tip_force_n": 3.0}}),
    ("force_8N", {"lever": {"tip_force_n": 8.0}}),
    ("force_10N", {"lever": {"tip_force_n": 10.0}}),
    ("mass_servo_lumped", {"case": "servo_lumped"}),
]


def run_one(job):
    name, change, plan_path = job
    saved = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    settings = dict(change)
    lever = lv.lever_parameters()
    lever.update(settings.pop("lever", {}))
    try:
        result, _, _ = lv.run(saved, lever=lever, pitch_deg=40.0, **settings)
    except RuntimeError as error:
        return name, {"success": False, "error": str(error), "change": change}
    trip, task = result["round_trip"], result["lever"]
    return name, {
        "success": bool(trip["passed"] and task["passed"]),
        "change": change,
        "round_trip": {
            key: trip[key]
            for key in (
                "passed",
                "fell",
                "saturated_steps",
                "max_ratio",
                "max_base_track_mm",
                "return_error_mm",
                "feet_down_at_end",
                "max_stance_slip_mm",
                "max_friction_use",
                "qp_failed_ticks",
            )
        },
        "lever": task,
    }


def main() -> None:
    plan_path, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(name, change, plan_path) for name, change in CONDITIONS]
    with multiprocessing.Pool(6) as pool:
        rows = dict(pool.map(run_one, jobs))
    (out / "conditions.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for name, row in rows.items():
        trip = row.get("round_trip", {})
        task = row.get("lever", {})
        print(
            f"{name:18s} success={row['success']!s:5s} "
            f"trip={trip.get('passed')} track={trip.get('max_base_track_mm', 0):5.1f}mm "
            f"slip={max(trip.get('max_stance_slip_mm', {0: 0}).values()):5.1f}mm "
            f"lever={task.get('lever_max_deg', 0):5.1f}deg ratio={trip.get('max_ratio', 0):.2f} "
            f"{row.get('error', '')}"
        )


if __name__ == "__main__":
    main()
