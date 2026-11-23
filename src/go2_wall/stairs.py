"""Ship stair scene and the N0 map: can the robot climb one gait cycle quasi-statically?

A ship stair (OSHA 1910.21(b)) rises at 50 to 70 degrees with open risers. Treads are boxes
whose top is at k * rise and whose back edge starts at (k - 1) * going, where going is
rise / tan(slope). Feet stand on tread centres. One gait cycle moves each foot up one tread,
one foot at a time, with the same plan checks as the wall task: static balance with the plan
friction margin, a horizontal push test, and clearance between the body and the stair.
"""

import argparse
import itertools
import json
import multiprocessing
from pathlib import Path

import mujoco
import numpy as np

from . import transition as tr
from .pose_search import FOOT_Y
from .robot import LEGS, RBQ_FOOT_Y, RBQ_HIP_X, build_robot, load_parameters
from .statics import torque_limits

GRID_GO2 = {
    "dx": np.round(np.arange(-0.15, 0.16, 0.03), 3),
    "dy": np.round(np.arange(-0.09, 0.10, 0.03), 3),
    "dz": np.round(np.arange(0.15, 0.46, 0.03), 3),
    "pitch_offsets": range(-30, 11, 5),
}
GRID_RBQ = {
    "dx": np.round(np.arange(-0.21, 0.22, 0.03), 3),
    "dy": np.round(np.arange(-0.09, 0.10, 0.03), 3),
    "dz": np.round(np.arange(0.24, 0.67, 0.03), 3),
    "pitch_offsets": range(-30, 11, 5),
}
"""RBQ legs are 0.33 + 0.33 m against 0.213 + 0.213 m on Go2, so the base search band is set
higher and longer by about the same ratio."""
GRID = dict(GRID_GO2)
FIRST_REAR_TREAD = 2
ARM = "d1"
"""Arm on the robot for stair work: "d1" or "z1" on Go2, "rb1" on RBQ-10. Set before building
scenes; use_stow() then switches the stow, the base grid and the foot spacing."""
STOW = {"arm_joint2": -1.57, "arm_joint3": 1.44, "arm_joint5": 0.39}
"""Arm stow for stairs: the lowest arm centre of mass (0.19 m above the base origin, against
0.32 m for the wall-task tuck) among collision-free poses whose gravity torque stays within
59% of the public URDF limits at 0 and 60 deg body pitch. The wall-task tuck needs 114% at
J2 when the body is level. Grid search over J2, J3, J5; see docs/stairs-feasibility.md."""
STRINGER_GAP_M = 0.02

_STATE: dict = {}


def stair_parameters() -> dict:
    return dict(load_parameters()["ship_stair"]["value"])


def going_of(slope_deg: float, rise: float) -> float:
    """Horizontal run per step.

    >>> round(going_of(60, 0.2), 4)
    0.1155
    """
    return float(rise / np.tan(np.radians(slope_deg)))


def tread_centre(k: int, slope_deg: float, rise: float, depth: float) -> np.ndarray:
    """Top-centre of tread k (k = 0 is the floor in front of the stair)."""
    if k == 0:
        return np.array([-0.25, 0.0, 0.0])
    going = going_of(slope_deg, rise)
    return np.array([(k - 1) * going + depth / 2, 0.0, k * rise])


def add_ship_stair(
    body: mujoco.MjsBody,
    slope_deg: float,
    rise: float,
    treads: int,
    floor_mu: float,
) -> None:
    """Add the T0 treads, stringers and upper landing along +x from x = 0."""
    stair = stair_parameters()
    depth, thick, width = stair["tread_depth_m"], stair["tread_thickness_m"], stair["clear_width_m"]
    going = going_of(slope_deg, rise)
    for k in range(1, treads + 1):
        geom = body.add_geom(
            name=f"tread{k}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[depth / 2, width / 2, thick / 2],
            pos=[(k - 1) * going + depth / 2, 0.0, k * rise - thick / 2],
            rgba=[0.62, 0.64, 0.68, 1],
        )
        geom.friction = [floor_mu, 0.02, 0.01]
        geom.priority = 2
    length = np.hypot(treads * going, treads * rise) + 0.3
    angle = np.arctan2(rise, going)
    centre = np.array([treads * going / 2, 0.0, treads * rise / 2])
    for side in (-1, 1):
        stringer = body.add_geom(
            name=f"stringer{'L' if side > 0 else 'R'}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[length / 2, 0.01, 0.12],
            pos=(centre + [0.0, side * (width / 2 + STRINGER_GAP_M), 0.0]).tolist(),
            quat=[np.cos(-angle / 2), 0.0, np.sin(-angle / 2), 0.0],
            rgba=[0.45, 0.47, 0.5, 1],
        )
        stringer.priority = 2
    platform = body.add_geom(
        name="platform",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.6, 0.6, thick / 2],
        pos=[treads * going + 0.6, 0.0, (treads + 1) * rise - thick / 2],
        rgba=[0.62, 0.64, 0.68, 1],
    )
    platform.friction = [floor_mu, 0.02, 0.01]
    platform.priority = 2


def build_stairs(
    slope_deg: float,
    rise: float,
    treads: int = 10,
    floor_mu: float = 0.6,
    case: str = "scaled",
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Robot on a floor with a ship stair rising along +x from x = 0."""
    spec = build_robot(case=case, wall_x=None, floor_mu=floor_mu, arm=ARM)
    add_ship_stair(spec.worldbody, slope_deg, rise, treads, floor_mu)
    model = spec.compile()
    return model, mujoco.MjData(model)


def foot_y() -> float:
    return RBQ_FOOT_Y if ARM == "rb1" else FOOT_Y


def hip_span() -> float:
    """Front-to-rear hip distance: 2 x 0.1934 m on Go2, 2 x 0.31218 m on RBQ-10."""
    return 2 * (RBQ_HIP_X if ARM == "rb1" else 0.1934)


def stance_on(rear: int, front: int, slope_deg: float, rise: float, depth: float, radius: float):
    """Feet on tread centres: rear feet on `rear`, front feet on `front`."""
    out = {}
    side = foot_y()
    for leg in LEGS:
        k = front if leg[0] == "F" else rear
        point = tread_centre(k, slope_deg, rise, depth)
        out[leg] = (point + [0.0, side if leg[1] == "L" else -side, radius], "floor")
    return out


def swing_over_tread(start: tuple, end: tuple, u: float) -> np.ndarray:
    """Raise the foot above both tread heights before crossing the riser."""
    first, last = start[0].copy(), end[0].copy()
    first[2] = last[2] = max(first[2], last[2]) + tr.LIFT_M
    if u < 0.25:
        return start[0] + tr.smooth(u / 0.25) * (first - start[0])
    if u < 0.75:
        return first + tr.smooth((u - 0.25) / 0.5) * (last - first)
    return last + tr.smooth((u - 0.75) / 0.25) * (end[0] - last)


def cycle(rear: int, gap: int, slope_deg, rise, depth, radius, order=("RL", "RR", "FL", "FR")):
    """Start stance and the stance after each swing, every foot up one tread."""
    start = stance_on(rear, rear + gap, slope_deg, rise, depth, radius)
    sequence, current = [], dict(start)
    for leg in order:
        k = (rear + gap if leg[0] == "F" else rear) + 1
        point = tread_centre(k, slope_deg, rise, depth)
        current = dict(current)
        side = foot_y()
        current[leg] = (point + [0.0, side if leg[1] == "L" else -side, radius], "floor")
        sequence.append((leg, current))
    return start, sequence


STOW_Z1 = {"arm_joint2": 0.0, "arm_joint3": 0.0, "arm_joint5": 1.34}
"""Z1 stow from tools/find_stow.py: arm centre of mass 0.18 m above and 0.06 m ahead of the base
origin, gravity torque 17% of the Menagerie limits at 0 and 60 deg pitch."""


STOW_RB1 = {"arm_joint2": -1.6, "arm_joint5": -2.967}
"""RB1 stow from tools/find_stow.py: upper arm laid back along the top of the body, wrist folded;
arm centre of mass 0.145 m above and 0.007 m behind the base origin, gravity torque 21% of the
assumed 100 N m limits at 0 and 60 deg pitch."""


def use_stow() -> None:
    """Make the shared planner helpers use the stair stow and the grid for the chosen body."""
    tr.TUCK = dict({"z1": STOW_Z1, "rb1": STOW_RB1}.get(ARM, STOW))
    GRID.clear()
    GRID.update(GRID_RBQ if ARM == "rb1" else GRID_GO2)


def _init(slope_deg: float, rise: float, arm: str) -> None:
    global ARM
    ARM = arm
    use_stow()
    _STATE["pair"] = build_stairs(slope_deg, rise)
    _STATE["limits"] = torque_limits("derived" if arm == "d1" else arm)


def candidates(stance: dict, slope_deg: float, pitch_offset: int) -> list[tuple]:
    centre = np.mean([item[0] for item in stance.values()], axis=0)
    pitch = float(np.radians(np.clip(slope_deg + pitch_offset, 0, 80)))
    return [
        (float(centre[0] + dx), float(dy), float(centre[2] + dz), pitch)
        for dx, dy, dz in itertools.product(GRID["dx"], GRID["dy"], GRID["dz"])
    ]


def _quick(job, *, swing_curve=tr.swing_point) -> list[tuple[float, tuple]]:
    """Score bases for one pitch by the static ratio at the top of the swing (or a stance)."""
    offset, before, after, leg, slope_deg = job
    model, data = _STATE["pair"]
    feet = {name: item[0] for name, item in after.items()}
    if leg is None:
        contacts = tr.contacts_for(before)
    else:
        feet[leg] = swing_curve(before[leg], after[leg], 0.5)
        contacts = tr.contacts_for(before, skip=leg)
    scored = []
    for base in candidates(before, slope_deg, offset):
        solution = tr.evaluate(
            model, data, base, feet, contacts, tr.tuck_arm(), _STATE["limits"], robust=True
        )
        if solution is not None and solution.ratio <= 1.0:
            scored.append((solution.ratio, base))
    return scored


def _path_job(job, *, swing_curve=tr.swing_point) -> float | None:
    previous, base, before, after, leg = job
    model, data = _STATE["pair"]
    return tr.path_cost(
        model, data, previous, base, before, after, leg, _STATE["limits"], swing_curve=swing_curve
    )


def plan_cycle(slope_deg: float, rise: float, gap: int, limit: int = 60, branch: int = 3) -> dict:
    """Search a start base and the four swings of one gait cycle."""
    use_stow()
    stair = stair_parameters()
    model, _ = build_stairs(slope_deg, rise)
    radius = model.geom_size[model.geom("FL").id][0]
    start, sequence = cycle(FIRST_REAR_TREAD, gap, slope_deg, rise, stair["tread_depth_m"], radius)
    offsets = list(GRID["pitch_offsets"])
    befores = [start] + [after for _, after in sequence[:-1]]
    with multiprocessing.Pool(
        len(offsets), initializer=_init, initargs=(slope_deg, rise, ARM)
    ) as pool:
        stand = sorted(
            item
            for part in pool.map(_quick, [(o, start, start, None, slope_deg) for o in offsets])
            for item in part
        )
        if not stand:
            return {"found": False, "reason": "no start stance", "candidates": [0]}
        jobs = [
            (o, befores[k], after, leg, slope_deg)
            for k, (leg, after) in enumerate(sequence)
            for o in offsets
        ]
        parts = pool.map(_quick, jobs)
        scored = [
            sorted(x for part in parts[k * len(offsets) : (k + 1) * len(offsets)] for x in part)
            for k in range(len(sequence))
        ]
        counts = [len(stand)] + [len(item) for item in scored]
        if not all(scored):
            failed = [sequence[k][0] for k, item in enumerate(scored) if not item]
            return {"found": False, "reason": f"no pose for swing {failed}", "candidates": counts}

        def search(k: int, previous: tuple) -> list | None:
            if k == len(sequence):
                return []
            leg, after = sequence[k]
            pool_jobs = [(previous, c, befores[k], after, leg) for _, c in scored[k][:limit]]
            costs = pool.map(_path_job, pool_jobs)
            passing = sorted(
                (cost, job[1]) for cost, job in zip(costs, pool_jobs) if cost is not None
            )
            for cost, candidate in passing[:branch]:
                rest = search(k + 1, candidate)
                if rest is not None:
                    return [(candidate, cost)] + rest
            return None

        for _, base in stand[:branch]:
            found = search(0, base)
            if found is not None:
                return {
                    "found": True,
                    "start_base": [round(v, 4) for v in base],
                    "steps": [
                        {"leg": leg, "base": [round(v, 4) for v in b], "path_cost": round(c, 4)}
                        for (leg, _), (b, c) in zip(sequence, found)
                    ],
                    "candidates": counts,
                }
    return {"found": False, "reason": "no connected swing sequence", "candidates": counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", default="d1", choices=("d1", "z1", "rb1"))
    args = parser.parse_args()
    global ARM
    ARM = args.arm
    args.out.mkdir(parents=True, exist_ok=True)
    stair = stair_parameters()
    rows = []
    for slope in stair["slopes_deg"]:
        for rise in stair["rises_m"]:
            going = going_of(slope, rise)
            ideal = hip_span() * np.cos(np.radians(slope)) / going
            gaps = sorted({max(1, int(np.floor(ideal))), max(1, int(np.ceil(ideal)))})
            result = {"found": False, "reason": "not run"}
            for gap in gaps:
                result = plan_cycle(slope, rise, gap)
                result["gap"] = gap
                if result["found"]:
                    break
            row = {"slope_deg": slope, "rise_m": rise, "going_m": round(going, 4), **result}
            rows.append(row)
            print(
                slope,
                rise,
                row.get("gap"),
                row["found"],
                row.get("reason", ""),
                row.get("candidates"),
                flush=True,
            )
            (args.out / "stair_map.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
