"""Plan the stand-up from four legs to the wall-braced pose and run it both ways (M2).

The plan is quasi-static. One foot moves at a time and the other three hold the robot. For
every swing the base pose is picked from a grid so that the largest joint-limit ratio stays
low along the swing. Load is handed to or from a foot by capping its normal force, which
keeps the static solution continuous through lift-off and touchdown.
"""

import argparse
import itertools
import json
import multiprocessing
from pathlib import Path

import mujoco
import numpy as np

from .gripper_control import FingerDrive, measured_arm_target
from .hold import ARM_GAINS, LEG_GAINS, apply_physics
from .pose_search import FOOT_Y, FRICTION_MARGIN, clear_of_collisions, set_base, solve_leg_ik
from .robot import ARM_JOINTS, LEG_PARTS, LEGS, ROOT, compile_robot, world_geoms
from .statics import FLOOR_NORMAL, WALL_NORMAL, Contact, joint_names, solve_static, torque_limits

REFERENCE = ROOT / "evidence/d1-front-mount/matrix.json"
NOSE_M = 0.343
NOSE_GAP_M = 0.05
STAND_HEIGHT_M = 0.30
TUCK = {"arm_joint2": -0.8, "arm_joint3": -1.2}
LIFT_M = 0.05
IK_TOLERANCE_M = 2e-3
# A horizontal push of this fraction of body weight at the centre of mass must still balance.
# On flat ground this keeps the centre of mass about 0.1 x 0.34 m = 3.4 cm inside the support
# polygon. Without it the plan put the centre of mass 1.6 cm from an edge and the robot tipped.
DISTURBANCE_FRACTION = 0.1
WALL_DISTURBANCE_FRACTION = 0.05
"""Push used when a foot is on the wall. The tipping failure that set 0.1 happened on three floor
feet; with wall contact the failure was sliding, which the wall preload handles, and 0.1 left
no base pose for the last front swing."""
# Parts other than the feet keep this distance from the floor and the wall in planned poses, so
# a tracking error of BASE_TRACK_LIMIT_M cannot bring the body into contact. The M1 pose keeps
# 36.7 mm (calves to wall). Without it the plan put the body 1 mm from the wall.
CLEARANCE_M = 0.03
# Wall heights of the intermediate front-foot steps. The last step goes to the M1 pose.
WALL_STEPS_M = (0.35, 0.55)
SHIFT_S, HANDOFF_S, SWING_S, ARM_S, HOLD_S = 1.5, 0.4, 1.2, 1.5, 1.0
SAMPLE_S = 0.02
GRID = {
    "x": np.round(np.arange(0.22, 0.58, 0.03), 3),
    # Lateral shift must reach about 0.1 m: after one rear foot steps forward, the robust support
    # region for the next rear swing sits near the stepped foot.
    "y": np.round(np.arange(-0.09, 0.10, 0.03), 3),
    "z": np.round(np.arange(0.22, 0.50, 0.03), 3),
    "pitch_deg": range(0, 61, 5),
}
# Pass criteria, fixed before the first run.
FALL_TILT_DEG = 30.0
BASE_TRACK_LIMIT_M = 0.03
RETURN_LIMIT_M = 0.02
STANCE_SLIP_LIMIT_M = 0.01
FRICTION_MIN_NORMAL_N = 5.0

_MODEL: dict = {}


def reference_pose() -> dict:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))["reference_pose"]


def base_of(qpos: np.ndarray) -> tuple[float, float, float, float]:
    """Return x, y, z, pitch of a base that is only pitched, as set_base writes it.

    >>> model, data = compile_robot()
    >>> set_base(model, data, 0.5, np.array([0.1, 0.0, 0.3]))
    >>> [round(v, 6) for v in base_of(data.qpos)]
    [0.1, 0.0, 0.3, 0.5]
    """
    pitch = 2 * np.arctan2(-qpos[5], qpos[3])
    return float(qpos[0]), float(qpos[1]), float(qpos[2]), float(pitch)


def normal_of(surface: str) -> np.ndarray:
    return FLOOR_NORMAL if surface == "floor" else WALL_NORMAL


def tuck_arm() -> dict[str, float]:
    return {name: TUCK.get(name, 0.0) for name in ARM_JOINTS}


def set_arm(model, data, arm: dict[str, float]) -> None:
    for name, value in arm.items():
        data.qpos[model.joint(name).qposadr[0]] = value


def stances(wall_x: float, radius: float) -> tuple[dict, list[tuple[str, dict]]]:
    """Return the start stance and the stance after each swing, ending at the M1 feet."""
    pose = reference_pose()
    model, data = compile_robot(wall_x=wall_x)
    data.qpos[:] = pose["qpos"]
    mujoco.mj_kinematics(model, data)
    final = {leg: data.geom_xpos[model.geom(leg).id].copy() for leg in LEGS}
    centre_x = wall_x - NOSE_M - NOSE_GAP_M
    hip_x = 0.1934
    start = {
        leg: (
            np.array(
                [
                    centre_x + (hip_x if leg[0] == "F" else -hip_x),
                    FOOT_Y if leg[1] == "L" else -FOOT_Y,
                    radius,
                ]
            ),
            "floor",
        )
        for leg in LEGS
    }
    # Rear feet step first, on the floor. With both front feet on the wall and one rear foot
    # on the floor no static solution exists, because the wall carries load only by friction.
    moves = [(leg, (final[leg].copy(), "floor")) for leg in ("RL", "RR")]
    for height in WALL_STEPS_M:
        moves += [
            (leg, (np.r_[wall_x - radius, final[leg][1], height], "wall")) for leg in ("FL", "FR")
        ]
    moves += [(leg, (final[leg].copy(), "wall")) for leg in ("FL", "FR")]
    sequence, current = [], dict(start)
    for leg, target in moves:
        current = dict(current)
        current[leg] = target
        sequence.append((leg, current))
    return start, sequence


def contacts_for(stance: dict, skip: str | None = None) -> list[Contact]:
    mu = 0.6 * FRICTION_MARGIN
    return [
        Contact(leg, normal_of(surface), mu) for leg, (_, surface) in stance.items() if leg != skip
    ]


def swing_point(start: tuple, end: tuple, u: float) -> np.ndarray:
    """Lift along the start normal, travel, and approach along the end normal.

    >>> a = (np.array([0.0, 0.0, 0.0]), "floor")
    >>> b = (np.array([1.0, 0.0, 0.5]), "wall")
    >>> swing_point(a, b, 0.25).round(3).tolist(), swing_point(a, b, 1.0).round(3).tolist()
    ([0.0, 0.0, 0.05], [1.0, 0.0, 0.5])
    """
    first = start[0] + LIFT_M * normal_of(start[1])
    last = end[0] + LIFT_M * normal_of(end[1])
    if u < 0.25:
        return start[0] + smooth(u / 0.25) * (first - start[0])
    if u < 0.75:
        return first + smooth((u - 0.25) / 0.5) * (last - first)
    return last + smooth((u - 0.75) / 0.25) * (end[0] - last)


def smooth(s: float) -> float:
    """Cubic ease with zero speed at both ends.

    >>> smooth(0.0), smooth(0.5), smooth(1.0)
    (0.0, 0.5, 1.0)
    """
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


def place(
    model,
    data,
    base,
    feet: dict[str, np.ndarray],
    arm: dict,
    clearance: float = 0.0,
    warm: np.ndarray | None = None,
) -> bool:
    """Pose the robot: base, arm, and legs by IK. `warm` is a previous qpos to start the IK from."""
    data.qpos[:] = model.qpos0 if warm is None else warm
    yaw = base[4] if len(base) >= 5 else 0.0
    roll = base[5] if len(base) >= 6 else 0.0
    set_base(model, data, base[3], np.array(base[:3]), yaw=yaw, roll=roll)
    set_arm(model, data, arm)
    if solve_leg_ik(model, data, feet, reset=warm is None) > IK_TOLERANCE_M:
        return False
    if not clear_of_collisions(model, data):
        return False
    return clearance <= 0 or min_clearance(model, data, clearance) >= clearance


def min_clearance(model, data, cap: float) -> float:
    """Smallest distance from any collision geom other than a foot to the floor or the wall."""
    feet = {model.geom(leg).id for leg in LEGS}
    world = world_geoms(model)
    fromto = np.zeros(6)
    smallest = cap
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] == 0 or geom in feet:
            continue
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        for other in world:
            smallest = min(smallest, mujoco.mj_geomDistance(model, data, geom, other, cap, fromto))
    return smallest


def evaluate(
    model,
    data,
    base,
    feet,
    contacts,
    arm,
    limits,
    max_normal=None,
    robust=False,
    disturb=None,
    warm=None,
):
    """Return the static solution, or None if unreachable, colliding, or not balanced.

    With robust=True the pose must also balance under the DISTURBANCE_FRACTION pushes and keep
    CLEARANCE_M between the body and the floor or wall. Planning uses robust=True.
    """
    if not place(model, data, base, feet, arm, CLEARANCE_M if robust else 0.0, warm):
        return None
    preload = wall_preload(contacts, max_normal)
    solution = solve_static(
        model, data, contacts, limits, max_normal=max_normal, min_normal=preload
    )
    if solution is None or not (robust if disturb is None else disturb):
        return solution
    on_wall = any(np.allclose(c.normal, WALL_NORMAL) for c in contacts)
    fraction = WALL_DISTURBANCE_FRACTION if on_wall else DISTURBANCE_FRACTION
    push = fraction * float(model.body_subtreemass[1]) * 9.81
    for direction in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)):
        force = push * np.array(direction, dtype=float)
        if (
            solve_static(
                model,
                data,
                contacts,
                limits,
                max_normal=max_normal,
                com_force=force,
                min_normal=preload,
            )
            is None
        ):
            return None
    return solution


def wall_preload(contacts: list[Contact], max_normal: dict | None, value: float | None = None):
    """Return a preload lower bound for wall contacts, never above a contact's cap.

    Plans use PLAN_PRELOAD_N. With 20 N a single wall foot plus the disturbance test left no
    base pose for one front swing, so plans ask for 10 N and the controller tries 20 N first.
    """
    value = PLAN_PRELOAD_N if value is None else value
    caps = max_normal or {}
    return {
        c.leg: min(value, caps.get(c.leg, np.inf))
        for c in contacts
        if np.allclose(c.normal, WALL_NORMAL)
    }


def _init(wall_x: float) -> None:
    _MODEL["pair"] = compile_robot(wall_x=wall_x)
    _MODEL["limits"] = torque_limits()


def _quick(job) -> list[tuple[float, tuple]]:
    """Score every base of one pitch by the static ratio at the top of the swing."""
    pitch_deg, before, after, leg = job
    model, data = _MODEL["pair"]
    feet = {name: item[0] for name, item in after.items()}
    feet[leg] = swing_point(before[leg], after[leg], 0.5)
    contacts = contacts_for(before, skip=leg)
    scored = []
    for x, y, z in itertools.product(GRID["x"], GRID["y"], GRID["z"]):
        base = (float(x), float(y), float(z), float(np.radians(pitch_deg)))
        solution = evaluate(
            model, data, base, feet, contacts, tuck_arm(), _MODEL["limits"], robust=True
        )
        if solution is not None and solution.ratio <= 1.0:
            scored.append((solution.ratio, base))
    return scored


def interpolate_base(a, b, s: float) -> tuple:
    w = smooth(s)
    n = max(len(a), len(b))
    start = tuple(a) + (0.0,) * (n - len(a))
    end = tuple(b) + (0.0,) * (n - len(b))
    return tuple(float((1 - w) * p + w * q) for p, q in zip(start, end, strict=True))


MOVE_PENALTY = (0.3, 0.1)
"""Cost per metre of base travel and per radian of pitch change, to prefer short shifts."""


def path_cost(
    model, data, previous, base, before, after, leg, limits, final=None, *, swing_curve=swing_point
):
    """Return the cost of the shift into `base` and the swing, or None if any sample fails.

    Every sample at the sample() resolution must be reachable, clear by CLEARANCE_M, and
    balanced with the plan preload. The disturbance pushes are checked on every third sample
    (60 ms apart). For the last swing, `final` is the M1 base and the shift there must hold
    too. The cost is the largest static ratio plus a small penalty for base travel.
    """
    worst = 0.0

    def check(base_pose, feet, contacts, index) -> bool:
        nonlocal worst
        solution = evaluate(
            model,
            data,
            base_pose,
            feet,
            contacts,
            tuck_arm(),
            limits,
            robust=True,
            disturb=index % 3 == 0,
        )
        if solution is None:
            return False
        worst = max(worst, solution.ratio)
        return True

    stand = {name: item[0] for name, item in before.items()}
    shift = np.linspace(0, 1, int(round(SHIFT_S / SAMPLE_S)) + 1)
    for index, s in enumerate(shift):
        last_one = index == len(shift) - 1
        if not check(
            interpolate_base(previous, base, s),
            stand,
            contacts_for(before),
            0 if last_one else index,
        ):
            return None
    swing = np.linspace(0, 1, int(round(SWING_S / SAMPLE_S)) + 1)
    for index, u in enumerate(swing):
        feet = {name: item[0] for name, item in after.items()}
        feet[leg] = swing_curve(before[leg], after[leg], u)
        contacts = contacts_for(before if u < 1 else after, skip=leg if u < 1 else None)
        if not check(base, feet, contacts, 0 if index == len(swing) - 1 else index):
            return None
    if final is not None:
        stand = {name: item[0] for name, item in after.items()}
        for index, s in enumerate(shift):
            if not check(interpolate_base(base, final, s), stand, contacts_for(after), index):
                return None
    travel = np.linalg.norm(np.subtract(base[:3], previous[:3]))
    turn = abs(base[3] - previous[3])
    if final is not None:
        travel += np.linalg.norm(np.subtract(final[:3], base[:3]))
        turn += abs(final[3] - base[3])
    return worst + MOVE_PENALTY[0] * travel + MOVE_PENALTY[1] * turn


def _path_job(job) -> float | None:
    previous, base, before, after, leg, final = job
    model, data = _MODEL["pair"]
    return path_cost(model, data, previous, base, before, after, leg, _MODEL["limits"], final)


def plan(wall_x: float = 0.75, limit: int = 60, branch: int = 3) -> dict:
    """Pick a base pose for every swing by depth-first search with backtracking.

    Candidates for each swing come from the quick score, which does not depend on the
    previous base, so all swings are scored once up front. At each swing the best `limit`
    candidates get a full path check in parallel, and up to `branch` of the passing ones are
    tried in order of cost before backing up to the previous swing.
    """
    model, data = compile_robot(wall_x=wall_x)
    radius = model.geom_size[model.geom("FL").id][0]
    limits = torque_limits()
    start, sequence = stances(wall_x, radius)
    start_base = (wall_x - NOSE_M - NOSE_GAP_M, 0.0, STAND_HEIGHT_M, 0.0)
    feet = {name: item[0] for name, item in start.items()}
    if evaluate(model, data, start_base, feet, contacts_for(start), tuck_arm(), limits) is None:
        raise RuntimeError("start stance is not feasible")
    befores = [start] + [after for _, after in sequence[:-1]]
    final_base = base_of(np.array(reference_pose()["qpos"]))
    with multiprocessing.Pool(
        len(GRID["pitch_deg"]), initializer=_init, initargs=(wall_x,)
    ) as pool:
        jobs = [
            (p, befores[k], after, leg)
            for k, (leg, after) in enumerate(sequence)
            for p in GRID["pitch_deg"]
        ]
        parts = pool.map(_quick, jobs)
        width = len(GRID["pitch_deg"])
        scored = [
            sorted(item for part in parts[k * width : (k + 1) * width] for item in part)
            for k in range(len(sequence))
        ]
        tried = {"paths": 0, "backtracks": 0}

        def search(k: int, previous: tuple) -> list | None:
            if k == len(sequence):
                return []
            leg, after = sequence[k]
            last = final_base if k == len(sequence) - 1 else None
            candidates = [c for _, c in scored[k][:limit]]
            costs = pool.map(
                _path_job, [(previous, c, befores[k], after, leg, last) for c in candidates]
            )
            tried["paths"] += len(candidates)
            passing = sorted((cost, c) for cost, c in zip(costs, candidates) if cost is not None)
            for cost, candidate in passing[:branch]:
                rest = search(k + 1, candidate)
                if rest is not None:
                    return [(candidate, cost)] + rest
                tried["backtracks"] += 1
            return None

        found = search(0, start_base)
    if found is None:
        raise RuntimeError("no base pose sequence found")
    steps = [
        {"leg": leg, "base": base, "ratio": cost, "stance": after}
        for (leg, after), (base, cost) in zip(sequence, found)
    ]
    return {
        "wall_x": wall_x,
        "start": start,
        "start_base": start_base,
        "steps": steps,
        "search": tried,
        "candidates": [len(item) for item in scored],
    }


def _stance_json(stance: dict) -> dict:
    return {
        leg: {"position_m": np.round(p, 4).tolist(), "surface": s} for leg, (p, s) in stance.items()
    }


def plan_json(result: dict) -> dict:
    return {
        "wall_x_m": result["wall_x"],
        "start_base": result["start_base"],
        "start": _stance_json(result["start"]),
        "search": result.get("search"),
        "candidates": result.get("candidates"),
        "steps": [
            {
                "leg": step["leg"],
                "base_xyz_pitch": [round(v, 4) for v in step["base"]],
                "path_cost": round(step["ratio"], 4),
                "stance": _stance_json(step["stance"]),
            }
            for step in result["steps"]
        ],
    }


def _stance_from_json(stance: dict) -> dict:
    return {leg: (np.array(v["position_m"]), v["surface"]) for leg, v in stance.items()}


def segments(saved: dict, arm_up: bool = True) -> list[dict]:
    """Turn a saved plan into timed segments that end in the M1 pose.

    With arm_up=False the arm stays tucked at the end, for the lever task to take over.
    """
    reference = reference_pose()
    final_arm = {
        name: reference["qpos"][index]
        for name, index in zip(ARM_JOINTS, _arm_addresses(saved["wall_x_m"]))
    }
    out = [
        {
            "kind": "hold",
            "base": tuple(saved["start_base"]),
            "stance": _stance_from_json(saved["start"]),
            "seconds": HOLD_S,
        }
    ]
    base, before = tuple(saved["start_base"]), out[0]["stance"]
    for step in saved["steps"]:
        target, after, leg = (
            tuple(step["base_xyz_pitch"]),
            _stance_from_json(step["stance"]),
            step["leg"],
        )
        out.append(
            {"kind": "shift", "from": base, "to": target, "stance": before, "seconds": SHIFT_S}
        )
        out.append(
            {"kind": "unload", "base": target, "stance": before, "leg": leg, "seconds": HANDOFF_S}
        )
        out.append(
            {
                "kind": "swing",
                "base": target,
                "before": before,
                "after": after,
                "leg": leg,
                "seconds": SWING_S,
            }
        )
        out.append(
            {"kind": "load", "base": target, "stance": after, "leg": leg, "seconds": HANDOFF_S}
        )
        base, before = target, after
    final_base = base_of(np.array(reference["qpos"]))
    out.append(
        {"kind": "shift", "from": base, "to": final_base, "stance": before, "seconds": SHIFT_S}
    )
    if not arm_up:
        out.append({"kind": "hold", "base": final_base, "stance": before, "seconds": HOLD_S})
        return out
    out.append(
        {"kind": "arm", "base": final_base, "stance": before, "arm": final_arm, "seconds": ARM_S}
    )
    out.append(
        {"kind": "hold", "base": final_base, "stance": before, "arm": final_arm, "seconds": HOLD_S}
    )
    return out


def _arm_addresses(wall_x: float) -> list[int]:
    model, _ = compile_robot(wall_x=wall_x)
    return [model.joint(name).qposadr[0] for name in ARM_JOINTS]


def sample(saved: dict, arm_up: bool = True) -> dict:
    """Sample the plan every SAMPLE_S: joint targets, static torques, and contact flags."""
    model, data = compile_robot(wall_x=saved["wall_x_m"])
    return sample_segments(model, data, segments(saved, arm_up), torque_limits(), tuck_arm())


class InfeasibleSegment(RuntimeError):
    """A planned motion could not be converted into a feasible controller sample."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


def _infeasible_pose(model, data, feet, clock, index, kind, progress):
    mujoco.mj_forward(model, data)
    foot_geoms = {model.geom(leg).id for leg in LEGS}
    world = set(world_geoms(model))
    errors = {leg: float(np.linalg.norm(data.geom(leg).xpos - feet[leg])) for leg in LEGS}
    overlaps = []
    for c in data.contact[: data.ncon]:
        pair = set(map(int, c.geom))
        if c.dist >= -1e-4 or (pair & foot_geoms and pair & world):
            continue
        names = [model.geom(i).name or model.body(model.geom_bodyid[i]).name for i in c.geom]
        overlaps.append({"geoms": names, "penetration_m": float(-c.dist)})
    return {
        "segment": index,
        "kind": kind,
        "progress": progress,
        "planned_time_s": clock,
        "foot_ik_error_m": errors,
        "foot_ik_limit_m": IK_TOLERANCE_M,
        "collision_tolerance_m": 1e-4,
        "unexpected_overlaps": overlaps,
        "static_reference_available": False,
        "planning_qpos": data.qpos.tolist(),
    }


def sample_segments(
    model, data, plan: list[dict], limits: np.ndarray, stow: dict[str, float]
) -> dict:
    """Sample shifts, foot swings and holds with the given robot and torque limits."""
    names = joint_names()
    address = np.array([model.joint(name).qposadr[0] for name in names])
    rows = {
        "time": [],
        "qpos": [],
        "target": [],
        "torque": [],
        "ratio": [],
        "smooth_ratio": [],
        "stance": [],
        "segment": [],
        "normal": [],
        "feet": [],
        "wall": [],
        "cap": [],
        "tip": [],
    }
    clock = 0.0
    for index, seg in enumerate(plan):
        count = int(round(seg["seconds"] / SAMPLE_S))
        handoff = None
        if seg["kind"] in ("unload", "load"):
            stand = {name: item[0] for name, item in seg["stance"].items()}
            full = evaluate(
                model, data, seg["base"], stand, contacts_for(seg["stance"]), stow, limits
            )
            if full is None:
                raise InfeasibleSegment(
                    f"segment {index} {seg['kind']} has no full support pose",
                    _infeasible_pose(model, data, stand, clock, index, seg["kind"], 0),
                )
            handoff = float(full.forces[seg["leg"]] @ normal_of(seg["stance"][seg["leg"]][1]))
        for k in range(0 if index == 0 else 1, count + 1):
            s = k / count
            arm = stow
            cap = None
            if seg["kind"] in ("hold", "unload", "load", "arm"):
                base, stance = seg["base"], seg["stance"]
                feet = {name: item[0] for name, item in stance.items()}
                contacts = contacts_for(stance)
                if seg["kind"] == "unload":
                    cap = {seg["leg"]: (1 - s) * handoff}
                if seg["kind"] == "load":
                    cap = {seg["leg"]: s * handoff}
                if seg["kind"] == "arm":
                    arm = {
                        name: (1 - smooth(s)) * stow[name] + smooth(s) * seg["arm"][name]
                        for name in ARM_JOINTS
                    }
                elif "arm" in seg:
                    arm = seg["arm"]
                flags = [True] * 4
            elif seg["kind"] == "shift":
                base, stance = interpolate_base(seg["from"], seg["to"], s), seg["stance"]
                feet = {name: item[0] for name, item in stance.items()}
                contacts = contacts_for(stance)
                flags = [True] * 4
            else:
                base, leg = seg["base"], seg["leg"]
                bases = seg.get("swing_bases")
                if bases is not None:
                    progress = s * (len(bases) - 1)
                    phase = min(int(progress), len(bases) - 2)
                    base = interpolate_base(bases[phase], bases[phase + 1], progress - phase)
                feet = {name: item[0] for name, item in seg["after"].items()}
                curve = seg.get("swing_curve", swing_point)
                feet[leg] = curve(seg["before"][leg], seg["after"][leg], s)
                stance = seg["before"] if s < 1 else seg["after"]
                contacts = contacts_for(stance, skip=leg if s < 1 else None)
                cap = None if s < 1 else {leg: 0.0}
                flags = [name != leg or s >= 1 for name in LEGS]
            warm = rows["qpos"][-1] if rows["qpos"] else None
            solution = evaluate(
                model, data, base, feet, contacts, arm, limits, max_normal=cap, warm=warm
            )
            if solution is not None:
                planned_ratio = solution.ratio
                solution = solve_static(model, data, contacts, limits, max_normal=cap, smooth=True)
            if solution is None:
                raise InfeasibleSegment(
                    f"segment {index} {seg['kind']} infeasible at s={s:.2f}",
                    _infeasible_pose(
                        model, data, feet, clock + s * seg["seconds"], index, seg["kind"], s
                    ),
                )
            rows["time"].append(clock + s * seg["seconds"])
            rows["qpos"].append(data.qpos.copy())
            rows["target"].append(data.qpos[address].copy())
            rows["torque"].append(solution.torque.copy())
            rows["ratio"].append(planned_ratio)
            rows["smooth_ratio"].append(solution.ratio)
            rows["stance"].append(flags)
            rows["segment"].append(index)
            rows["feet"].append([data.geom_xpos[model.geom(leg).id].copy() for leg in LEGS])
            rows["wall"].append([stance[leg][1] == "wall" for leg in LEGS])
            rows["cap"].append([(cap or {}).get(leg, np.inf) for leg in LEGS])
            rows["tip"].append(np.zeros(3))
            rows["normal"].append(
                [
                    float(solution.forces[leg] @ normal_of(stance[leg][1]))
                    if leg in solution.forces
                    else 0.0
                    for leg in LEGS
                ]
            )
        clock += seg["seconds"]
    return {key: np.array(value) for key, value in rows.items()}


def round_trip(track: dict) -> dict:
    """Append the time-reversed plan so the robot goes back to four legs."""
    out = {}
    end = track["time"][-1]
    for key, value in track.items():
        if key == "time":
            out[key] = np.r_[value, 2 * end - value[-2::-1]]
        else:
            out[key] = np.concatenate([value, value[-2::-1]])
    return out


BASE_GAINS = (1600.0, 340.0)
"""Base position feedback in N/m and N s/m: about 1.5 Hz with damping ratio 1 on 17.9 kg."""
ROT_GAINS = (40.0, 6.0)
"""Base orientation feedback in N m/rad and N m s/rad."""
CONTROL_EVERY = 10
LOG_EVERY = 10
STANCE_DAMPING = 0.3
SWING_FROM_PLAN = True
"""Swing legs follow planned joint angles instead of world foot targets from the measured base."""
LIFT_FADE_S = 0.3
"""Seconds over which a lifted leg moves from its measured joints onto the planned joints."""
SWING_GAIN_SCALE = 1.0
"""Multiplier on LEG_GAINS for a leg in swing, which carries no load."""
COMPENSATION = (1.0, 0.0)
"""Share of the passive joint damping and dry friction that the controller cancels.

Cancelling the damping removed the wall-foot sliding on the way down. Also cancelling the
dry friction, or only half the damping, left 17 to 49 mm of wall-foot sliding."""
FRICTION_SPEED = 0.02
"""rad/s over which the dry-friction compensation ramps in, to avoid chatter at rest."""
SLIDE_DAMPING = 20.0
"""N s/m on the sliding velocity of a stance foot's contact point.

The feet are 22 mm spheres that roll as the legs turn, so their centres move by about 1 cm
without any sliding. An earlier spring that held foot centres at touchdown fought this rolling
with up to 15 N per foot. Damping only the contact-point velocity resists sliding, not rolling.
"""


def contact_velocity(model, data, geom: int, normal: np.ndarray, jacobian: np.ndarray):
    """Tangential velocity of the foot material point at the contact, and its Jacobian.

    For rolling without sliding this velocity is zero even though the sphere centre moves.
    """
    point = data.geom_xpos[geom] - model.geom_size[geom][0] * normal
    mujoco.mj_jac(model, data, jacobian, None, point, model.geom_bodyid[geom])
    speed = jacobian @ data.qvel
    return speed - (speed @ normal) * normal


NOMINAL_MU = 0.6
"""Friction the controller assumes. S8 changes the true friction but not this value."""
WALL_PRELOAD_N = 20.0
PLAN_PRELOAD_N = 10.0
"""Lower bound on the normal force of a loaded wall foot.

The plan leaves the wall feet 6 to 12 N, so small tangential loads from leg motion used up the
friction and the feet slid down the wall. Pressing at least 20 N leaves friction to spare."""
CONTROL_ATTEMPTS = tuple(
    (wall, floor, preload, scale)
    for wall, preload in ((0.5, WALL_PRELOAD_N), (0.5, PLAN_PRELOAD_N), (0.6, PLAN_PRELOAD_N))
    for scale in (1.0, 0.5)
    for floor in (0.5, FRICTION_MARGIN)
) + tuple((FRICTION_MARGIN, FRICTION_MARGIN, 0.0, scale) for scale in (1.0, 0.5, 0.25, 0.0))
"""(wall share of NOMINAL_MU, floor share, wall preload in N, feedback share), tried in order.

Minimising torque drives feet to the friction limit. The controller first asks every foot for
half the friction and the wall feet for a 20 N preload, and relaxes step by step only when no
solution exists. The plan itself uses FRICTION_MARGIN (0.7) and PLAN_PRELOAD_N."""


def leg_ik(model, scratch, leg: str, target: np.ndarray, iterations: int = 30) -> np.ndarray:
    """Solve one leg for a world foot target from the base pose already in scratch.qpos."""
    geom = model.geom(leg).id
    joints = [model.joint(f"{leg}_{part}_joint") for part in LEG_PARTS]
    jacobian = np.zeros((3, model.nv))
    for _ in range(iterations):
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        gap = target - scratch.geom_xpos[geom]
        if np.linalg.norm(gap) < 1e-4:
            break
        mujoco.mj_jacGeom(model, scratch, jacobian, None, geom)
        local = jacobian[:, [item.dofadr[0] for item in joints]]
        step = local.T @ np.linalg.solve(local @ local.T + 1e-4 * np.eye(3), gap)
        for item, delta in zip(joints, step):
            address = item.qposadr[0]
            scratch.qpos[address] = np.clip(
                scratch.qpos[address] + np.clip(delta, -0.3, 0.3), *item.range
            )
    return np.array([scratch.qpos[item.qposadr[0]] for item in joints])


def _feedback_wrench(model, data, track, i, planned, scales=(1.0, 1.0)) -> np.ndarray:
    """Return the generalized force of the base feedback wrench at the measured pose."""
    times = track["time"]
    dt = times[i + 1] - times[i]
    base_speed = (track["qpos"][i + 1][:3] - track["qpos"][i][:3]) / dt
    spin = np.zeros(3)
    mujoco.mju_subQuat(spin, track["qpos"][i + 1][3:7], track["qpos"][i][3:7])
    position_scale, rotation_scale = scales
    force = BASE_GAINS[0] * position_scale * (planned[:3] - data.qpos[:3]) + BASE_GAINS[
        1
    ] * np.sqrt(position_scale) * (base_speed - data.qvel[:3])
    error = np.zeros(3)
    mujoco.mju_subQuat(error, planned[3:7], data.qpos[3:7])
    local = ROT_GAINS[0] * rotation_scale * error + ROT_GAINS[1] * np.sqrt(rotation_scale) * (
        spin / dt - data.qvel[3:6]
    )
    torque = data.xmat[1].reshape(3, 3) @ local
    wrench = np.zeros(model.nv)
    mujoco.mj_applyFT(model, data, force, torque, data.xpos[1], 1, wrench)
    return wrench


def physics_step_count(duration_s: float, timestep_s: float) -> int:
    """Count whole steps, including an endpoint rounded down by trajectory summation."""
    ratio = duration_s / timestep_s
    nearest = round(ratio)
    return nearest if abs(ratio - nearest) <= 1e-8 else int(ratio)


class StanceXYSpring:
    """Optional world-XY foot-centre restraint applied through leg motor torque."""

    def __init__(self, gain_n_per_m: float, start_s: float, end_s: float):
        values = (gain_n_per_m, start_s, end_s)
        if any(not np.isfinite(value) for value in values):
            raise ValueError("Stance XY spring settings must be finite")
        if gain_n_per_m <= 0:
            raise ValueError("Stance XY spring gain must be positive")
        if start_s < 0 or end_s < start_s:
            raise ValueError("Stance XY spring interval must be non-negative and ordered")
        self.gain_n_per_m = gain_n_per_m
        self.start_s = start_s
        self.end_s = end_s
        self.anchors_xy: dict[int, np.ndarray] = {}
        self.jacobian: np.ndarray | None = None
        self.maximum_error_m = 0.0
        self.maximum_force_n = 0.0
        self.maximum_joint_correction_nm = 0.0

    def correction(self, model, data, geom: int, dofs: np.ndarray, time_s: float) -> np.ndarray:
        if time_s < self.start_s - 1e-9 or time_s > self.end_s + 1e-9:
            return np.zeros(len(dofs))
        centre_xy = data.geom_xpos[geom][:2].copy()
        anchor_xy = self.anchors_xy.setdefault(int(geom), centre_xy)
        error_xy = centre_xy - anchor_xy
        force_xy = self.gain_n_per_m * error_xy
        if self.jacobian is None:
            self.jacobian = np.zeros((3, model.nv))
        mujoco.mj_jacGeom(model, data, self.jacobian, None, geom)
        correction = -(self.jacobian[:2, dofs].T @ force_xy)
        self.maximum_error_m = max(self.maximum_error_m, float(np.linalg.norm(error_xy)))
        self.maximum_force_n = max(self.maximum_force_n, float(np.linalg.norm(force_xy)))
        self.maximum_joint_correction_nm = max(
            self.maximum_joint_correction_nm, float(np.max(np.abs(correction)))
        )
        return correction

    def details(self) -> dict:
        return {
            "gain_n_per_m": self.gain_n_per_m,
            "active_s": [self.start_s, self.end_s],
            "axes": "world_xy",
            "application": "physical stance motor command only: -J_xy.T K e_xy",
            "anchor_count": len(self.anchors_xy),
            "maximum_xy_error_m": self.maximum_error_m,
            "maximum_spring_force_n": self.maximum_force_n,
            "maximum_joint_correction_nm": self.maximum_joint_correction_nm,
        }


def execute(
    track: dict,
    wall_x: float | None,
    floor_mu: float = 0.6,
    wall_mu: float = 0.6,
    physics: dict | None = None,
    gains_scale: float = 1.0,
    builder=None,
    watch=None,
    mode: str = "force",
    *,
    limits_nm: np.ndarray | None = None,
    swing_from_plan: bool | None = None,
    log_every_step: bool = False,
    actuator_command=None,
    arm_target_command=None,
    actuation_period_s: float | None = None,
    finger_actuator: str = "RB1_finger_upper",
    task_contact_pairs: tuple[tuple[str, str], ...] = (),
    stance_xy_spring_n_per_m: float = 0.0,
    stance_xy_spring_active_s: tuple[float, float] | None = None,
    stance_slip_spring_n_per_m: float = 0.0,
    stance_slip_spring_active_s: tuple[float, float] | None = None,
    stance_slip_spring_legs: tuple[str, ...] = (),
    walking_control=None,
) -> tuple[dict, dict]:
    """Run the sampled plan in contact physics.

    mode="joint": every leg follows joint targets from inverse kinematics on the planned base,
    with stance feet aimed at where they actually touched down, plus the planned static
    torques. mode="force" is described below.

    Every 0.02 s (CONTROL_EVERY steps of the model's own timestep) the static QP is solved again
    at the measured pose with a base feedback wrench added, and its torques drive the stance
    legs. Swing legs follow the planned world foot position by inverse kinematics from the
    measured base, and the arm follows its planned joints, both with PD on top of the QP
    torques. `builder` may supply another model (the lever scene) and `watch(model, data)` may
    return an extra value to log. `limits_nm` sets the limits for a different robot.
    Set `swing_from_plan=False` to solve swing targets from the measured base pose.
    `actuation_period_s` holds every motor request between servo ticks while physics,
    contact measurements and the state audit continue at each simulation step.
    Use `log_every_step` when verification needs every physics sample.
    `actuator_command(time_s, slide_m, velocity_mps)` supplies a named finger motor's
    force in N. Its state and command audit is added to the controller metrics.
    `arm_target_command(time_s, model, data, nominal_rad)` may read the measured state
    and return six corrected joint targets for the existing torque-limited arm controller.
    `task_contact_pairs` exempts named geom pairs from the unexpected-contact metric;
    the task observer measures those pairs and checks their timing.
    A positive `stance_xy_spring_n_per_m` adds ``-J_xy.T K e_xy`` to grounded leg
    motor commands only during `stance_xy_spring_active_s`. The default zero gain
    leaves the prior controller path unchanged.
    A positive `stance_slip_spring_n_per_m` applies ``-J.T K integrated_slip`` only
    to the named grounded legs and interval. It changes motor commands while leaving
    the measured contact-point slip state untouched.
    A walking controller may replace the eighteen requested motor torques on a servo tick.
    Its distribution status contributes to the same per-step and cumulative solver audit.
    """
    if builder is None:
        model, data = compile_robot(wall_x=wall_x, floor_mu=floor_mu, wall_mu=wall_mu)
    else:
        model, data = builder()
    model_dt = model.opt.timestep
    base_feedback_scales = getattr(walking_control, "base_feedback_scales", None)
    apply_physics(model, physics or {})
    stance_xy_spring = None
    if stance_xy_spring_n_per_m != 0.0:
        if stance_xy_spring_active_s is None:
            raise ValueError("A stance XY spring interval is required for a positive gain")
        stance_xy_spring = StanceXYSpring(stance_xy_spring_n_per_m, *stance_xy_spring_active_s)
    if stance_slip_spring_n_per_m:
        values = (stance_slip_spring_n_per_m, *(stance_slip_spring_active_s or ()))
        if (
            stance_slip_spring_active_s is None
            or any(not np.isfinite(value) for value in values)
            or stance_slip_spring_n_per_m <= 0
            or stance_slip_spring_active_s[0] < 0
            or stance_slip_spring_active_s[1] < stance_slip_spring_active_s[0]
            or not stance_slip_spring_legs
            or not set(stance_slip_spring_legs) <= set(LEGS)
        ):
            raise ValueError("Stance slip spring settings are invalid")
    slip_spring_force_n = 0.0
    slip_spring_joint_nm = 0.0
    actuator_stride = 1
    if actuation_period_s is not None:
        ratio = actuation_period_s / model.opt.timestep
        if not np.isfinite(ratio) or ratio < 1 or abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("Actuation period must be a positive whole number of physics steps")
        actuator_stride = round(ratio)
    # Control and logging stay at the same rate in seconds when the timestep is overridden.
    per_model_step = model_dt / model.opt.timestep
    control_every = max(1, round(CONTROL_EVERY * per_model_step))
    log_every = 1 if log_every_step else max(1, round(LOG_EVERY * per_model_step))
    scratch = mujoco.MjData(model)
    limits = torque_limits() if limits_nm is None else limits_nm
    follow_plan = SWING_FROM_PLAN if swing_from_plan is None else swing_from_plan
    names = joint_names()
    position = np.array([model.joint(name).qposadr[0] for name in names])
    velocity = np.array([model.joint(name).dofadr[0] for name in names])
    columns = {leg: np.arange(3 * k, 3 * k + 3) for k, leg in enumerate(LEGS)}
    passive_damping = model.dof_damping[velocity].copy()
    passive_friction = model.dof_frictionloss[velocity].copy()
    base_kp = np.r_[np.full(12, LEG_GAINS[0]), np.full(6, ARM_GAINS[0])]
    base_kd = np.r_[np.full(12, LEG_GAINS[1]), np.full(6, ARM_GAINS[1])]
    kp, kd = base_kp * gains_scale, base_kd * gains_scale
    gain_schedule = track.get("gain_scale")
    size = track["qpos"].shape[1]
    data.qpos[:size] = track["qpos"][0]
    mujoco.mj_forward(model, data)
    finger = (
        FingerDrive(model, data, finger_actuator, actuator_command) if actuator_command else None
    )
    task_pairs = {
        (model.geom(x).id, model.geom(y).id)
        for a, b in task_contact_pairs
        for x, y in ((a, b), (b, a))
    }
    feet_ids = [model.geom(leg).id for leg in LEGS]
    allowed = set(feet_ids)
    world = set(world_geoms(model))
    times = track["time"]
    steps = physics_step_count(times[-1], model.opt.timestep)
    feed = track["torque"][0].copy()
    swing_target = track["target"][0].copy()
    swing_speed = np.zeros(18)
    was_stance = np.ones(4, dtype=bool)
    offset = np.zeros(18)
    lift_time = [0.0] * 4
    foot_jacobian = np.zeros((3, model.nv))
    anchor = [None] * 4
    max_slip = [0.0] * 4
    slid = [np.zeros(3) for _ in range(4)]
    worst = {"base_track_m": 0.0, "tilt_track_deg": 0.0, "ratio": 0.0}
    saturated, fell, fallback, relaxed = 0, False, 0, 0
    other: dict[str, float] = {}
    friction = [0.0] * 4
    log = {
        "time": [],
        "torque": [],
        "force": [],
        "base": [],
        "feet": [],
        "extra": [],
        "qpos": [],
        "qvel": [],
        "ctrl": [],
        "attempt": [],
        "commanded": [],
    }
    chosen = walking_attempt = 0
    commanded = np.zeros((4, 3))
    if finger:
        log["physics_time_s"] = []
        log["requested_finger_force_n"] = []
        log["unexpected_contact_n"] = []
        log["robot_torque_requested_nm"] = []
        log["base_track_m"] = []
        log["stance_slip_m"] = []
        log["fell"] = []
    if arm_target_command is not None:
        log["nominal_arm_target_rad"] = []
        log["arm_target_rad"] = []
    for step in range(steps):
        t = step * model.opt.timestep
        i = min(int(np.searchsorted(times, t, side="right")) - 1, len(times) - 2)
        w = (t - times[i]) / (times[i + 1] - times[i])
        if gain_schedule is not None:
            gain = (1 - w) * gain_schedule[i] + w * gain_schedule[i + 1]
            kp, kd = base_kp * gain, base_kd * gain
        planned = (1 - w) * track["qpos"][i] + w * track["qpos"][i + 1]
        planned[3:7] /= np.linalg.norm(planned[3:7])
        stance = track["stance"][i] & track["stance"][i + 1]
        if mode == "joint":
            feed = (1 - w) * track["torque"][i] + w * track["torque"][i + 1]
            if step % control_every == 0:
                previous = swing_target.copy()
                ahead = min(i + 1, len(times) - 1)
                for k, leg in enumerate(LEGS):
                    scratch.qpos[:size] = track["qpos"][ahead]
                    goal = (
                        anchor[k]
                        if stance[k] and anchor[k] is not None
                        else track["feet"][ahead][k]
                    )
                    swing_target[columns[leg]] = leg_ik(model, scratch, leg, goal)
                swing_speed = (swing_target - previous) / (control_every * model.opt.timestep)
                if step == 0:
                    swing_speed[:] = 0
        elif step % control_every == 0:
            scales = base_feedback_scales(t) if base_feedback_scales else (1.0, 1.0)
            wrench = _feedback_wrench(model, data, track, i, planned, scales)
            cap = {
                leg: float(track["cap"][i][k])
                for k, leg in enumerate(LEGS)
                if np.isfinite(track["cap"][i][k])
            }
            solution = None
            for attempt, (share, floor_share, preload, scale) in enumerate(CONTROL_ATTEMPTS):
                contacts = [
                    Contact(
                        leg,
                        WALL_NORMAL if track["wall"][i][k] else FLOOR_NORMAL,
                        NOMINAL_MU * (share if track["wall"][i][k] else floor_share),
                    )
                    for k, leg in enumerate(LEGS)
                    if stance[k]
                ]
                scratch.qpos[:] = data.qpos
                solution = solve_static(
                    model,
                    scratch,
                    contacts,
                    limits,
                    tip_force=track["tip"][i] if "tip" in track else None,
                    max_normal=cap,
                    smooth=True,
                    extra_qfrc=-scale * wrench,
                    min_normal=wall_preload(contacts, cap, preload),
                )
                if solution is not None:
                    relaxed += int(attempt > 0)
                    break
            chosen = attempt if solution is not None else -1
            if solution is not None:
                commanded = np.array([solution.forces.get(leg, np.zeros(3)) for leg in LEGS])
            if solution is None:
                fallback += 1
            else:
                feed = solution.torque
            ahead = min(i + 1, len(times) - 1)
            for k, leg in enumerate(LEGS):
                if not stance[k] and was_stance[k]:
                    cols = columns[leg]
                    swing_target[cols] = data.qpos[position][cols]
                    # A force-controlled stance leg ends where its rolled foot put it, which
                    # can differ from the planned joints by 0.2 rad; fade that out.
                    offset[cols] = data.qpos[position][cols] - track["target"][i][cols]
                    lift_time[k] = t
            was_stance = stance.copy()
            previous = swing_target.copy()
            for k, leg in enumerate(LEGS):
                if not stance[k] and follow_plan:
                    cols = columns[leg]
                    fade = 1.0 - smooth((t - lift_time[k]) / LIFT_FADE_S)
                    swing_target[cols] = track["target"][ahead][cols] + fade * offset[cols]
                elif not stance[k]:
                    scratch.qpos[:] = data.qpos
                    goal = track["feet"][ahead][k]
                    swing_target[columns[leg]] = leg_ik(model, scratch, leg, goal)
            swing_speed = (swing_target - previous) / (control_every * model.opt.timestep)
        update_command = step % actuator_stride == 0
        if update_command:
            dt = times[i + 1] - times[i]
            nominal_arm = (1 - w) * track["target"][i][12:] + w * track["target"][i + 1][12:]
            arm_target = nominal_arm
            if arm_target_command is not None:
                arm_target = measured_arm_target(
                    arm_target_command, t, model, data, nominal_arm, finger
                )
            arm_speed = (track["target"][i + 1][12:] - track["target"][i][12:]) / dt
            q, qd = data.qpos[position], data.qvel[velocity]
            command = feed.copy()
            for k, leg in enumerate(LEGS):
                cols = columns[leg]
                if mode == "joint":
                    command[cols] += kp[cols] * (swing_target[cols] - q[cols]) + kd[cols] * (
                        swing_speed[cols] - qd[cols]
                    )
                elif stance[k]:
                    command[cols] -= STANCE_DAMPING * qd[cols]
                    normal = WALL_NORMAL if track["wall"][i][k] else FLOOR_NORMAL
                    slide = contact_velocity(model, data, feet_ids[k], normal, foot_jacobian)
                    command[cols] -= foot_jacobian[:, velocity[cols]].T @ (SLIDE_DAMPING * slide)
                    if stance_xy_spring is not None:
                        command[cols] += stance_xy_spring.correction(
                            model, data, feet_ids[k], velocity[cols], t
                        )
                    if (
                        stance_slip_spring_n_per_m
                        and leg in stance_slip_spring_legs
                        and stance_slip_spring_active_s[0] <= t <= stance_slip_spring_active_s[1]
                    ):
                        force = stance_slip_spring_n_per_m * slid[k]
                        correction = -(foot_jacobian[:, velocity[cols]].T @ force)
                        command[cols] += correction
                        slip_spring_force_n = max(slip_spring_force_n, float(np.linalg.norm(force)))
                        slip_spring_joint_nm = max(
                            slip_spring_joint_nm, float(np.max(np.abs(correction)))
                        )
                else:
                    command[cols] += SWING_GAIN_SCALE * (
                        kp[cols] * (swing_target[cols] - q[cols])
                        + kd[cols] * (swing_speed[cols] - qd[cols])
                    )
            command[12:] += kp[12:] * (arm_target - q[12:]) + kd[12:] * (arm_speed - qd[12:])
            # Compensate passive joint damping and dry friction missing from the static solution.
            command += COMPENSATION[0] * passive_damping * qd
            command += COMPENSATION[1] * passive_friction * np.tanh(qd / FRICTION_SPEED)
            if walking_control is not None:
                before = finger.snapshot(data) if finger else None
                command, walking_attempt = walking_control.command(
                    model, data, track, i, w, stance, commanded, t, command
                )
                if finger:
                    finger.compare(data, before)
                if command.shape != limits.shape or not np.isfinite(command).all():
                    raise ValueError("Walking feedback must return eighteen finite motor torques")
                fallback += int(walking_attempt < 0)
                relaxed += int(walking_attempt > 0)
            data.ctrl[:18] = np.clip(command, -limits, limits)
        saturated += int(np.any(np.abs(command) > limits))
        if finger:
            requested_finger = finger.apply(
                data, t, model.opt.timestep, update_command=update_command
            )
        mujoco.mj_step(model, data)
        track_error = float(np.linalg.norm(data.qpos[:3] - planned[:3]))
        worst["base_track_m"] = max(worst["base_track_m"], track_error)
        difference = np.zeros(3)
        mujoco.mju_subQuat(difference, data.qpos[3:7], planned[3:7])
        tilt = float(np.degrees(np.linalg.norm(difference)))
        worst["tilt_track_deg"] = max(worst["tilt_track_deg"], tilt)
        fell = fell or tilt > FALL_TILT_DEG or data.qpos[2] < 0.12
        for k, geom in enumerate(feet_ids):
            if stance[k]:
                if anchor[k] is None:
                    anchor[k] = data.geom_xpos[geom].copy()
                    slid[k] = np.zeros(3)
                normal = WALL_NORMAL if track["wall"][i][k] else FLOOR_NORMAL
                slide = contact_velocity(model, data, geom, normal, foot_jacobian)
                slid[k] += slide * model.opt.timestep
                max_slip[k] = max(max_slip[k], float(np.linalg.norm(slid[k])))
            else:
                anchor[k] = None
        if step % log_every == 0:
            mujoco.mj_forward(model, data)
            applied = data.actuator_force[:18].copy()
            worst["ratio"] = max(worst["ratio"], float(np.max(np.abs(applied) / limits)))
            sample_contacts = {} if finger else other
            totals = _foot_forces(
                model, data, feet_ids, world, allowed, sample_contacts, task_pairs
            )
            if finger:
                for pair_name, force_n in sample_contacts.items():
                    other[pair_name] = max(other.get(pair_name, 0.0), force_n)
                log["unexpected_contact_n"].append(max(sample_contacts.values(), default=0.0))
                log["robot_torque_requested_nm"].append(command.copy())
                log["base_track_m"].append(track_error)
                log["stance_slip_m"].append(max(max_slip))
                log["fell"].append(fell)
            for k in range(4):
                force = totals[k]
                axis = 0 if track["wall"][i][k] else 2
                normal = abs(force[axis])
                if stance[k] and normal > FRICTION_MIN_NORMAL_N:
                    tangent = np.linalg.norm(np.delete(force, axis))
                    mu = wall_mu if axis == 0 else floor_mu
                    friction[k] = max(friction[k], float(tangent / (mu * normal)))
            log["time"].append(t)
            log["torque"].append(applied)
            log["force"].append(np.array(totals))
            log["base"].append(data.qpos[:7].copy())
            log["feet"].append(np.array([data.geom_xpos[g].copy() for g in feet_ids]))
            log["qpos"].append(data.qpos.copy())
            log["qvel"].append(data.qvel.copy())
            log["ctrl"].append(data.ctrl.copy())
            log["attempt"].append(
                -1 if min(chosen, walking_attempt) < 0 else max(chosen, walking_attempt)
            )
            log["commanded"].append(commanded.copy())
            if arm_target_command is not None:
                log["nominal_arm_target_rad"].append(nominal_arm)
                log["arm_target_rad"].append(arm_target.copy())
            before_observer = finger.snapshot(data) if finger else None
            log["extra"].append(watch(model, data) if watch else 0.0)
            if finger:
                finger.compare(data, before_observer)
                log["physics_time_s"].append(data.time)
                log["requested_finger_force_n"].append(requested_finger)
    final_error = float(np.linalg.norm(data.qpos[:3] - track["qpos"][-1][:3]))
    feet_down = all(abs(data.geom_xpos[g][2] - model.geom_size[g][0]) < 0.01 for g in feet_ids)
    passed = (
        not fell
        and saturated == 0
        and worst["ratio"] <= 1.0
        and worst["base_track_m"] <= BASE_TRACK_LIMIT_M
        and final_error <= RETURN_LIMIT_M
        and feet_down
        and max(max_slip) <= STANCE_SLIP_LIMIT_M
    )
    summary = {
        "passed": bool(passed),
        "fell": bool(fell),
        "saturated_steps": saturated,
        "qp_failed_ticks": fallback,
        "qp_relaxed_ticks": relaxed,
        "max_ratio": worst["ratio"],
        "max_base_track_mm": worst["base_track_m"] * 1e3,
        "max_tilt_track_deg": worst["tilt_track_deg"],
        "return_error_mm": final_error * 1e3,
        "feet_down_at_end": bool(feet_down),
        "max_stance_slip_mm": dict(zip(LEGS, np.round(np.array(max_slip) * 1e3, 3).tolist())),
        "max_friction_use": dict(zip(LEGS, np.round(friction, 3).tolist())),
        "unexpected_contacts_n": other,
        "floor_mu": floor_mu,
        "wall_mu": wall_mu,
        "physics": physics or {},
        "gains_scale": gains_scale,
        "mode": mode,
        "seconds": float(times[-1]),
    }
    if finger:
        summary["gripper_audit"] = finger.audit()
    if walking_control is not None:
        summary["walking_control"] = walking_control.describe()
    if stance_xy_spring is not None:
        summary["stance_xy_spring"] = stance_xy_spring.details()
    if stance_slip_spring_n_per_m:
        summary["stance_slip_spring"] = {
            "gain_n_per_m": stance_slip_spring_n_per_m,
            "active_s": list(stance_slip_spring_active_s),
            "legs": list(stance_slip_spring_legs),
            "application": "physical stance motor command only: -J.T K integrated_slip",
            "maximum_spring_force_n": slip_spring_force_n,
            "maximum_joint_correction_nm": slip_spring_joint_nm,
        }
    return summary, {key: np.array(value) for key, value in log.items()}


def _foot_forces(model, data, feet_ids, world, allowed, other, task_pairs=()) -> list[np.ndarray]:
    totals = [np.zeros(3) for _ in feet_ids]
    wrench = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        mujoco.mj_contactForce(model, data, index, wrench)
        force = contact.frame.reshape(3, 3).T @ wrench[:3]
        pair = (contact.geom1, contact.geom2)
        if pair[1] in allowed and pair[0] in world:
            totals[feet_ids.index(pair[1])] += force
        elif pair[0] in allowed and pair[1] in world:
            totals[feet_ids.index(pair[0])] -= force
        elif pair not in task_pairs:
            names = sorted(model.body(model.geom_bodyid[g]).name for g in pair)
            key = "+".join(names)
            other[key] = max(other.get(key, 0.0), float(np.linalg.norm(force)))
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", type=Path, help="reuse a saved plan.json")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.plan:
        saved = json.loads(args.plan.read_text(encoding="utf-8"))
    else:
        saved = plan_json(plan())
        (args.out / "plan.json").write_text(json.dumps(saved, indent=1), encoding="utf-8")
    track = round_trip(sample(saved))
    summary, log = execute(track, saved["wall_x_m"])
    summary["planned_max_ratio"] = float(track["ratio"].max())
    (args.out / "run.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    np.savez_compressed(args.out / "track.npz", **track, **{f"log_{k}": v for k, v in log.items()})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
