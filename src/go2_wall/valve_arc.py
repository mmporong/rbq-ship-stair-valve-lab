"""Can the robot close a quarter-turn valve over its whole stroke while standing on a stair?

A valve stem points along y, so the lever turns in the vertical x-z plane in front of the
robot. For a stem centre C, a start angle, and a turning sense, the handle point P(theta) and
the closing direction t(theta) are sampled along the 90 degree stroke. At every sample the
robot may re-pose: the largest static force along t is taken over the robust stair stances and
several arm IK solutions (statics.max_tip_force). A stroke is feasible when the smallest of
these maxima reaches the handle force the valve needs.
"""

import argparse
import itertools
import json
import multiprocessing
from pathlib import Path

import numpy as np

from . import stairs as st
from . import transition as tr
from . import valve as vl
from .pose_search import clear_of_collisions
from .statics import max_tip_force, torque_limits

CENTRE_DX_M = (0.2, 0.3, 0.4)
CENTRE_DZ_M = (0.6, 0.8, 1.0, 1.2)
START_DEG = (0, 90, 180, 270)
SENSES = (1, -1)
SAMPLES = 7
BASES_PER_POINT = 40

_STATE: dict = {}


def stroke(centre: np.ndarray, start_deg: float, sense: int, length: float):
    """Handle points and closing directions over a 90 degree stroke.

    >>> points, dirs = stroke(np.zeros(3), 0.0, 1, 0.1)
    >>> points[0].round(3).tolist(), dirs[0].round(3).tolist()
    ([0.1, 0.0, 0.0], [0.0, 0.0, 1.0])
    """
    angles = np.radians(start_deg + sense * np.linspace(0.0, 90.0, SAMPLES))
    points = [centre + length * np.array([np.cos(a), 0.0, np.sin(a)]) for a in angles]
    dirs = [sense * np.array([-np.sin(a), 0.0, np.cos(a)]) + 0.0 for a in angles]
    return points, dirs


def limits_for(arm: str, arm_limits=None) -> np.ndarray:
    """Leg and arm limits; arm_limits (six values, N m) replaces the arm part when given."""
    limits = torque_limits(arm if arm in ("z1", "rb1") else vl.ARM_LIMIT_SET)
    if arm_limits is not None:
        limits[12:] = arm_limits
    return limits


def _init(
    arm: str, slope: float, rise: float, bases: list, share: float = 0.7, arm_limits=None
) -> None:
    tr.FRICTION_MARGIN = share
    st.ARM = arm
    st.use_stow()
    limits = limits_for(arm, arm_limits)
    _STATE.update(pair=st.build_stairs(slope, rise), limits=limits, bases=bases)
    vl._STATE.update(pair=_STATE["pair"], limits=_STATE["limits"])


def _point_max(job) -> float:
    """Largest force at one handle point and direction over stances and arm solutions."""
    point, direction, feet_state = job
    model, data = _STATE["pair"]
    feet = {leg: item[0] for leg, item in feet_state.items()}
    best = 0.0
    for base in _STATE["bases"]:
        if not tr.place(model, data, base, feet, tr.tuck_arm()):
            continue
        legs = data.qpos.copy()
        for seed in vl.ARM_SEEDS:
            data.qpos[:] = legs
            if vl.arm_ik(model, data, point, seed) > 1e-3 or not clear_of_collisions(model, data):
                continue
            site = model.site("arm_tip").id
            force = max_tip_force(
                model,
                data,
                tr.contacts_for(feet_state),
                _STATE["limits"],
                direction,
                data.site_xpos[site].copy(),
                model.site_bodyid[site],
            )
            best = max(best, force or 0.0)
    return best


def study(
    arm: str,
    slope: float,
    rise: float,
    gap: int,
    share: float = 0.7,
    handle: float | None = None,
    arm_limits=None,
) -> dict:
    """share is the part of the true friction the feet may use (plan default 0.7)."""
    tr.FRICTION_MARGIN = share
    st.ARM = arm
    st.use_stow()
    model, _ = st.build_stairs(slope, rise)
    radius = model.geom_size[model.geom("FL").id][0]
    feet_state = vl.stance(slope, rise, gap, radius)
    length = handle or vl.load_parameters()["valve_handle_length_m"]["value"]
    need = vl.needed_force() * vl.load_parameters()["valve_handle_length_m"]["value"] / length
    offsets = list(st.GRID["pitch_offsets"])
    with multiprocessing.Pool(
        len(offsets), initializer=_init, initargs=(arm, slope, rise, [], share, arm_limits)
    ) as pool:
        stances = sorted(
            x
            for part in pool.map(vl._robust_bases, [(o, feet_state, slope) for o in offsets])
            for x in part
        )
    bases = [b for _, b in stances[:: max(1, len(stances) // BASES_PER_POINT)]][:BASES_PER_POINT]
    front = np.mean([feet_state["FL"][0], feet_state["FR"][0]], axis=0)
    arcs, jobs = [], []
    for dx, dz, start, sense in itertools.product(CENTRE_DX_M, CENTRE_DZ_M, START_DEG, SENSES):
        centre = front + [dx, 0.0, dz]
        points, dirs = stroke(centre, start, sense, length)
        arcs.append({"centre_m": np.round(centre, 3).tolist(), "start_deg": start, "sense": sense})
        jobs += [(p, d, feet_state) for p, d in zip(points, dirs)]
    with multiprocessing.Pool(
        len(offsets), initializer=_init, initargs=(arm, slope, rise, bases, share, arm_limits)
    ) as pool:
        forces = pool.map(_point_max, jobs, chunksize=4)
    for index, arc in enumerate(arcs):
        values = forces[index * SAMPLES : (index + 1) * SAMPLES]
        arc["forces_n"] = [round(v, 1) for v in values]
        arc["min_n"] = round(min(values), 1)
        arc["feasible"] = bool(min(values) >= need)
    feasible = [a for a in arcs if a["feasible"]]
    return {
        "arm": arm,
        "slope_deg": slope,
        "rise_m": rise,
        "gap": gap,
        "friction_share": share,
        "arm_limits_nm": [round(float(v), 2) for v in limits_for(arm, arm_limits)[12:]],
        "needed_n": need,
        "handle_length_m": length,
        "robust_stances": len(stances),
        "bases_used": len(bases),
        "arcs": arcs,
        "feasible_arcs": len(feasible),
        "total_arcs": len(arcs),
        "best_min_n": max(a["min_n"] for a in arcs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", default="z1", choices=("d1", "z1", "rb1"))
    parser.add_argument("--slope", type=float, default=60.0)
    parser.add_argument("--rise", type=float, default=0.2)
    parser.add_argument("--gap", type=int, default=2)
    parser.add_argument("--friction-share", type=float, default=0.7)
    parser.add_argument("--handle", type=float, help="lever length in m (default: parameter file)")
    parser.add_argument("--arm-limits", type=float, nargs=6, help="replace the arm limits (N m)")
    parser.add_argument("--tag", default="", help="suffix for the output file name")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = study(
        args.arm, args.slope, args.rise, args.gap, args.friction_share, args.handle, args.arm_limits
    )
    share = int(round(args.friction_share * 100))
    lever = "" if args.handle is None else f"_lever{int(round(args.handle * 1000))}"
    tag = f"_{args.tag}" if args.tag else ""
    name = (
        f"arcs_{args.arm}_{int(args.slope)}deg_{int(args.rise * 1000)}mm_mu{share}{lever}{tag}.json"
    )
    (args.out / name).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(
        args.arm,
        args.slope,
        "stances",
        result["robust_stances"],
        "feasible arcs",
        result["feasible_arcs"],
        "/",
        result["total_arcs"],
        "best min",
        result["best_min_n"],
        "need",
        round(result["needed_n"], 1),
    )


if __name__ == "__main__":
    main()
