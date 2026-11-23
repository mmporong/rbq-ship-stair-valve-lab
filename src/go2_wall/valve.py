"""N2: how hard can Go2+D1 push a valve handle while standing on a ship stair?

The handle force needed is the manufacturer operating torque over the handle length (10 N m
over 0.12 m, 83 N). For each handle point in front of the robot and each push direction in
the vertical plane, the largest static force is found by linear programming (statics.
max_tip_force) with the public URDF arm limits. Two cases are compared:

- nominal: the stance with the lowest joint-limit ratio and the first arm IK solution;
- whole body: the best over the robust stances and several arm IK solutions.
"""

import argparse
import itertools
import json
import multiprocessing
from pathlib import Path

import mujoco
import numpy as np

from . import stairs as st
from . import transition as tr
from .pose_search import clear_of_collisions
from .robot import ARM_JOINTS, load_parameters
from .statics import max_tip_force, torque_limits

ARM_LIMIT_SET = "urdf"
ARM_TORQUE_OVERRIDE: float | None = None
"""If set, every arm joint limit becomes this value (N m), to ask what a stronger arm adds."""
ANGLES_DEG = tuple(range(-180, 180, 30))
HANDLE_DX_M = (0.15, 0.25, 0.35)
HANDLE_DZ_M = (0.5, 0.7, 0.9, 1.1)
ARM_SEEDS = (
    (0.0, 0.8, -0.8, 0.0, 0.0, 0.0),
    (0.0, 0.3, 0.3, 0.0, 0.6, 0.0),
    (0.0, 1.2, -1.4, 0.0, 0.4, 0.0),
    (0.0, -0.2, 0.9, 0.0, -0.5, 0.0),
    (0.0, 0.6, 0.6, 0.0, -1.0, 0.0),
)
BASES_KEPT = 60

_STATE: dict = {}


def needed_force() -> float:
    """Handle force for the M10HP 1/2 in reduced-bore valve on the assumed handle.

    >>> round(needed_force(), 1)
    83.3
    """
    params = load_parameters()
    torque = params["ball_valve_operating_torque_nm"]["value"]["half_inch_reduced_bore"]
    return torque / params["valve_handle_length_m"]["value"]


def arm_ik(model, data, target: np.ndarray, seed, iterations: int = 80) -> float:
    """Place the arm tip at `target` with all six arm joints (position only)."""
    joints = [model.joint(name) for name in ARM_JOINTS]
    for item, value in zip(joints, seed):
        data.qpos[item.qposadr[0]] = value
    site = model.site("arm_tip").id
    jacobian = np.zeros((3, model.nv))
    dofs = [item.dofadr[0] for item in joints]
    error = np.inf
    for _ in range(iterations):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        gap = target - data.site_xpos[site]
        error = float(np.linalg.norm(gap))
        if error < 5e-4:
            break
        mujoco.mj_jacSite(model, data, jacobian, None, site)
        local = jacobian[:, dofs]
        step = local.T @ np.linalg.solve(local @ local.T + 1e-3 * np.eye(3), gap)
        for item, delta in zip(joints, step):
            address = item.qposadr[0]
            data.qpos[address] = np.clip(
                data.qpos[address] + np.clip(delta, -0.2, 0.2), *item.range
            )
    mujoco.mj_kinematics(model, data)
    return error


def stance(slope_deg: float, rise: float, gap: int, radius: float) -> dict:
    depth = st.stair_parameters()["tread_depth_m"]
    return st.stance_on(
        st.FIRST_REAR_TREAD, st.FIRST_REAR_TREAD + gap, slope_deg, rise, depth, radius
    )


def handle_points(feet: dict) -> list[np.ndarray]:
    front = np.mean([feet["FL"][0], feet["FR"][0]], axis=0)
    return [front + [dx, 0.0, dz] for dx, dz in itertools.product(HANDLE_DX_M, HANDLE_DZ_M)]


def directions() -> list[np.ndarray]:
    return [np.array([np.cos(np.radians(a)), 0.0, np.sin(np.radians(a))]) for a in ANGLES_DEG]


def arm_limits() -> np.ndarray:
    limits = torque_limits(ARM_LIMIT_SET)
    if ARM_TORQUE_OVERRIDE is not None:
        limits[12:] = ARM_TORQUE_OVERRIDE
    return limits


def _init(slope_deg: float, rise: float, friction_share: float, override=None) -> None:
    global ARM_TORQUE_OVERRIDE
    ARM_TORQUE_OVERRIDE = override
    st.use_stow()
    tr.FRICTION_MARGIN = friction_share
    _STATE["pair"] = st.build_stairs(slope_deg, rise)
    _STATE["limits"] = arm_limits()


def _robust_bases(job) -> list[tuple[float, tuple]]:
    offset, feet_state, slope_deg = job
    model, data = _STATE["pair"]
    feet = {leg: item[0] for leg, item in feet_state.items()}
    out = []
    for base in st.candidates(feet_state, slope_deg, offset):
        solution = tr.evaluate(
            model, data, base, feet, tr.contacts_for(feet_state), tr.tuck_arm(), _STATE["limits"],
            robust=True,
        )  # fmt: skip
        if solution is not None:
            out.append((solution.ratio, base))
    return out


def capacity(model, data, base, feet_state, point, limits, seeds=ARM_SEEDS) -> list:
    """Max force per direction for each arm IK solution at this base; None where unreachable."""
    feet = {leg: item[0] for leg, item in feet_state.items()}
    if not tr.place(model, data, base, feet, tr.tuck_arm()):
        return []
    legs = data.qpos.copy()
    rows = []
    for seed in seeds:
        data.qpos[:] = legs
        if arm_ik(model, data, point, seed) > 1e-3 or not clear_of_collisions(model, data):
            continue
        arm = [float(data.qpos[model.joint(n).qposadr[0]]) for n in ARM_JOINTS]
        site = model.site("arm_tip").id
        forces = [
            max_tip_force(
                model,
                data,
                tr.contacts_for(feet_state),
                limits,
                direction,
                data.site_xpos[site].copy(),
                model.site_bodyid[site],
            )
            for direction in directions()
        ]
        rows.append({"arm": arm, "forces": forces})
    return rows


def _best_at_base(job):
    base, feet_state, points = job
    model, data = _STATE["pair"]
    out = []
    for point in points:
        best = [0.0] * len(ANGLES_DEG)
        for row in capacity(model, data, base, feet_state, point, _STATE["limits"]):
            best = [max(b, f or 0.0) for b, f in zip(best, row["forces"])]
        out.append(best)
    return base, out


def study(
    slope_deg: float, rise: float, gap: int, friction_share: float = 0.7, keep: int = BASES_KEPT
) -> dict:
    """friction_share is the share of the true friction the feet may use (plan default 0.7)."""
    st.use_stow()
    tr.FRICTION_MARGIN = friction_share
    model, data = st.build_stairs(slope_deg, rise)
    radius = model.geom_size[model.geom("FL").id][0]
    feet_state = stance(slope_deg, rise, gap, radius)
    points = handle_points(feet_state)
    limits = arm_limits()
    offsets = list(st.GRID["pitch_offsets"])
    with multiprocessing.Pool(
        len(offsets),
        initializer=_init,
        initargs=(slope_deg, rise, friction_share, ARM_TORQUE_OVERRIDE),
    ) as pool:
        bases = sorted(
            x
            for part in pool.map(_robust_bases, [(o, feet_state, slope_deg) for o in offsets])
            for x in part
        )
        if not bases:
            return {"slope_deg": slope_deg, "rise_m": rise, "error": "no robust stance"}
        kept = [b for _, b in bases[:: max(1, len(bases) // keep)]][:keep]
        results = pool.map(_best_at_base, [(b, feet_state, points) for b in kept])
    whole = np.zeros((len(points), len(ANGLES_DEG)))
    owner = [[None] * len(ANGLES_DEG) for _ in points]
    for base, per_point in results:
        for i, best in enumerate(per_point):
            for j, value in enumerate(best):
                if value > whole[i, j]:
                    whole[i, j] = value
                    owner[i][j] = base
    nominal = np.zeros((len(points), len(ANGLES_DEG)))
    for i, point in enumerate(points):
        rows = capacity(model, data, bases[0][1], feet_state, point, limits, seeds=ARM_SEEDS[:1])
        if rows:
            nominal[i] = [f or 0.0 for f in rows[0]["forces"]]
    need = needed_force()
    return {
        "slope_deg": slope_deg,
        "rise_m": rise,
        "gap": gap,
        "friction_share": friction_share,
        "needed_n": need,
        "arm_limits": ARM_LIMIT_SET if ARM_TORQUE_OVERRIDE is None else ARM_TORQUE_OVERRIDE,
        "angles_deg": list(ANGLES_DEG),
        "points_m": [np.round(p, 3).tolist() for p in points],
        "robust_stances": len(bases),
        "bases_searched": len(kept),
        "nominal_base": [round(v, 4) for v in bases[0][1]],
        "nominal_n": np.round(nominal, 1).tolist(),
        "whole_body_n": np.round(whole, 1).tolist(),
        "whole_body_base": [
            [None if b is None else [round(v, 4) for v in b] for b in r] for r in owner
        ],
        "share_reaching_need": {
            "nominal": float(np.mean(nominal >= need)),
            "whole_body": float(np.mean(whole >= need)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slope", type=float, default=60.0)
    parser.add_argument("--rise", type=float, default=0.2)
    parser.add_argument("--gap", type=int, default=2)
    parser.add_argument("--friction-share", type=float, default=0.7)
    parser.add_argument("--keep", type=int, default=BASES_KEPT)
    parser.add_argument("--arm-torque", type=float, help="override every arm joint limit (N m)")
    args = parser.parse_args()
    global ARM_TORQUE_OVERRIDE
    ARM_TORQUE_OVERRIDE = args.arm_torque
    args.out.mkdir(parents=True, exist_ok=True)
    result = study(args.slope, args.rise, args.gap, args.friction_share, args.keep)
    share = int(round(args.friction_share * 100))
    arm = "" if args.arm_torque is None else f"_arm{int(args.arm_torque)}"
    name = f"capacity_{int(args.slope)}deg_{int(args.rise * 1000)}mm_mu{share}{arm}.json"
    (args.out / name).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("needed_n", "robust_stances", "share_reaching_need")}))
    whole = np.array(result["whole_body_n"])
    nominal = np.array(result["nominal_n"])
    print("max nominal", nominal.max(), "max whole body", whole.max())
    for i, point in enumerate(result["points_m"]):
        print(point, "nominal max", nominal[i].max(), "whole max", whole[i].max(),
              "best angle", ANGLES_DEG[int(np.argmax(whole[i]))])  # fmt: skip


if __name__ == "__main__":
    main()
