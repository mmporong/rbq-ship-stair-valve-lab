"""Lever on the wall above four-leg reach, operated by the arm from the braced pose (M3).

The handle sticks out of the wall and ends in a crossbar. The arm presses the crossbar down
with the underside of its wrist, which is the lowest part of the gripper when the tip points
upward. Lever resistance is hinge friction, sized from the tip force in the parameter file.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from . import transition as tr
from .pose_search import (
    FOOT_Y,
    HIP_X,
    PLANAR_ARM,
    clear_of_collisions,
    set_base,
    solve_arm_ik,
    solve_leg_ik,
)
from .robot import LEGS, build_robot, compile_robot, load_parameters, set_damping
from .statics import solve_static, torque_limits

PUSHER_BODIES = ("arm_Empty_Link6", "arm_Empty_Link_L", "arm_Empty_Link_R")
BAR_HALF_M = 0.07
FRICTION_TIME_S = 0.004
PUSH_MARGIN = 1.0
"""Static feedforward force as a multiple of the force that just overcomes hinge friction.

The arm PD supplies the rest. With 1.2 the arm compliance let the handle run to 62.9 deg
against a 43 deg plan; with 1.0 it stops at 52.7 deg."""
OVERSHOOT_DEG = 3.0
GAP_M = 0.003
SEEDS = [(1.2, -1.2, 0.3), (1.0, -0.5, 0.0), (0.5, -1.0, 0.0), (0.2, -0.3, 0.5), (1.5, -2.0, 0.8)]
TIMES = {
    "reach": 2.0,
    "approach": 1.0,
    "settle": 0.5,
    "push": 3.0,
    "release": 0.5,
    "lift": 1.0,
    "back": 2.0,
}
"""Seconds per phase. The wrist comes in level from the robot side 12 mm above the bar, then
settles onto it, because there is no reach to come down from above."""
HOVER_M = 0.012
APPROACH_RUN_M = 0.05
HOLD_LIMITS = {"slip_m": 0.005, "base_m": 0.010, "friction": 0.8}
"""During the lever task the body must hold as in M1."""


def lever_parameters() -> dict:
    return dict(load_parameters()["lever_scene"]["value"])


def build_scene(
    wall_x: float = 0.75,
    floor_mu: float = 0.6,
    wall_mu: float = 0.6,
    case: str = "scaled",
    lever: dict | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Return the robot, wall, and lever. The lever joint angle grows as the handle goes down."""
    lever = lever or lever_parameters()
    spec = build_robot(case=case, wall_x=wall_x, floor_mu=floor_mu, wall_mu=wall_mu)
    length, radius = lever["handle_length_m"], lever["handle_radius_m"]
    body = spec.worldbody.add_body(name="lever", pos=[wall_x, 0.0, lever["hinge_height_m"]])
    # The handle is counterweighted. MuJoCo hinge friction is a soft constraint, and the bare
    # handle crept down under its own weight (2.6 deg in the first second, 61 deg in 30 s).
    body.gravcomp = 1.0
    joint = body.add_joint(name="lever_hinge", type=mujoco.mjtJoint.mjJNT_HINGE)
    joint.axis = [0.0, -1.0, 0.0]
    joint.range = [-0.1, 1.3]
    joint.limited = True
    joint.frictionloss = lever["tip_force_n"] * length
    set_damping(joint, 0.05)
    # Stiffest friction constraint allowed at the 2 ms step. With the default 0.02 s time
    # constant a 3 N push (below the 5 N needed) moved the handle 11 deg in 1 s; now 2.3 deg.
    joint.solref_friction = [FRICTION_TIME_S, 1.0]
    joint.armature = 0.001
    stem = body.add_geom(
        name="lever_stem",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0, 0, 0, -length, 0, 0],
        size=[radius, 0, 0],
        rgba=[0.85, 0.2, 0.15, 1],
    )
    stem.mass = lever["handle_mass_kg"] * 0.6
    bar = body.add_geom(
        name="lever_bar",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[-length, -BAR_HALF_M, 0, -length, BAR_HALF_M, 0],
        size=[radius, 0, 0],
        rgba=[0.85, 0.2, 0.15, 1],
    )
    bar.mass = lever["handle_mass_kg"] * 0.4
    spec.add_exclude(bodyname1="world", bodyname2="lever")
    model = spec.compile()
    return model, mujoco.MjData(model)


def bar_top(lever: dict, wall_x: float, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Top point of the crossbar centre and the handle's upward normal at a lever angle.

    >>> top, normal = bar_top({"hinge_height_m": 1.0, "handle_length_m": 0.1,
    ...                        "handle_radius_m": 0.01}, 0.5, 0.0)
    >>> top.round(3).tolist(), normal.round(3).tolist()
    ([0.4, 0.0, 1.01], [-0.0, 0.0, 1.0])
    """
    hinge = np.array([wall_x, 0.0, lever["hinge_height_m"]])
    length = lever["handle_length_m"]
    centre = hinge + np.array([-length * np.cos(angle), 0.0, -length * np.sin(angle)])
    normal = np.array([-np.sin(angle), 0.0, np.cos(angle)])
    return centre + lever["handle_radius_m"] * normal, normal


def pusher_offset(model, data, pitch: float) -> np.ndarray:
    """Lowest point of the wrist and fingers relative to the tip site, for a tip pitch.

    The gripper moves in the sagittal plane, so the offset depends only on the pitch. It is
    measured with upward rays on the current arm pose after aiming the tip at `pitch`.
    """
    shoulder = data.xanchor[model.joint("arm_joint2").id]
    target = shoulder + np.array([0.35, 0.0, 0.25])
    solve_arm_ik(model, data, target, pitch, SEEDS)
    mujoco.mj_forward(model, data)
    site = model.site("arm_tip").id
    tip = data.site_xpos[site].copy()
    bodies = {model.body(name).id for name in PUSHER_BODIES}
    group = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
    hit = np.zeros(1, dtype=np.int32)
    lowest = None
    for x in np.linspace(tip[0] - 0.12, tip[0] + 0.03, 61):
        for y in (-0.04, 0.04):
            start = np.array([x, y, tip[2] - 0.25])
            distance = mujoco.mj_ray(model, data, start, np.array([0, 0, 1.0]), group, 0, -1, hit)
            if distance >= 0 and model.geom_bodyid[hit[0]] in bodies:
                z = start[2] + distance
                if lowest is None or z < lowest[1]:
                    lowest = (x, z)
    if lowest is None:
        raise RuntimeError("no gripper surface found")
    return np.array([lowest[0] - tip[0], 0.0, lowest[1] - tip[2]])


def braced_config(model, data) -> None:
    data.qpos[:] = 0
    data.qpos[: len(tr.reference_pose()["qpos"])] = tr.reference_pose()["qpos"]
    tr.set_arm(model, data, tr.tuck_arm())
    mujoco.mj_kinematics(model, data)


def four_leg_config(model, data, wall_x: float) -> None:
    """Tallest four-leg stance with the nose 1 cm from the wall, as in the M1 reach study."""
    data.qpos[:] = model.qpos0
    radius = model.geom_size[model.geom("FL").id][0]
    centre = np.array([wall_x - 0.343 - 0.01, 0.0, 0.385])
    set_base(model, data, 0.0, centre)
    targets = {
        leg: np.array(
            [
                centre[0] + (HIP_X if leg[0] == "F" else -HIP_X),
                FOOT_Y if leg[1] == "L" else -FOOT_Y,
                radius,
            ]
        )
        for leg in LEGS
    }
    solve_leg_ik(model, data, targets)
    tr.set_arm(model, data, tr.tuck_arm())
    mujoco.mj_kinematics(model, data)


def push_targets(lever: dict, wall_x: float, offset: np.ndarray, angles) -> list[np.ndarray]:
    return [
        bar_top(lever, wall_x, a)[0] + GAP_M * bar_top(lever, wall_x, a)[1] - offset for a in angles
    ]


def arm_path(model, data, targets, pitch: float) -> list[list[float]] | None:
    """Solve the planar arm for each tip target in order, or None if any target fails."""
    joints = [model.joint(name) for name in PLANAR_ARM]
    seeds = list(SEEDS)
    out = []
    for target in targets:
        error = solve_arm_ik(model, data, target, pitch, seeds)
        if error > 1e-3 or not clear_of_collisions(model, data):
            return None
        solution = [float(data.qpos[item.qposadr[0]]) for item in joints]
        seeds = [tuple(solution)] + list(SEEDS)
        out.append(solution)
    return out


def choose_scene(wall_x: float = 0.75, heights=None, pitches=(10, 20, 30, 40)) -> list[dict]:
    """Check the push path from the braced and four-leg poses per hinge height and tip pitch."""
    lever = lever_parameters()
    heights = heights if heights is not None else np.round(np.arange(1.18, 1.36, 0.02), 2)
    model, data = compile_robot(wall_x=wall_x)
    target = np.radians(lever["target_deg"] + OVERSHOOT_DEG)
    angles = np.linspace(0.0, target, 12)
    rows = []
    for pitch in pitches:
        braced_config(model, data)
        offset = pusher_offset(model, data, np.radians(pitch))
        for height in heights:
            trial = dict(lever, hinge_height_m=float(height))
            targets = push_targets(trial, wall_x, offset, angles)
            braced_config(model, data)
            braced = arm_path(model, data, targets, np.radians(pitch)) is not None
            four_leg_config(model, data, wall_x)
            reach = [arm_path(model, data, [t], np.radians(pitch)) is not None for t in targets]
            rows.append(
                {
                    "hinge_height_m": float(height),
                    "tip_pitch_deg": pitch,
                    "offset_m": offset.round(4).tolist(),
                    "braced_full_path": braced,
                    "four_leg_points_reached": int(sum(reach)),
                    "points": len(targets),
                }
            )
    return rows


def four_leg_any_pitch(wall_x: float = 0.75, pitches=range(0, 81, 10)) -> list[dict]:
    """At the chosen hinge height, which push points the four-leg pose reaches at any pitch."""
    lever = lever_parameters()
    model, data = compile_robot(wall_x=wall_x)
    angles = np.linspace(0.0, np.radians(lever["target_deg"] + OVERSHOOT_DEG), 12)
    rows = []
    for pitch in pitches:
        braced_config(model, data)
        offset = pusher_offset(model, data, np.radians(pitch))
        targets = push_targets(lever, wall_x, offset, angles)
        four_leg_config(model, data, wall_x)
        hits = [arm_path(model, data, [t], np.radians(pitch)) is not None for t in targets]
        braced_config(model, data)
        braced = arm_path(model, data, targets, np.radians(pitch)) is not None
        rows.append(
            {
                "tip_pitch_deg": pitch,
                "four_leg_points": [int(h) for h in hits],
                "four_leg_reaches_start": bool(hits[0]),
                "braced_full_path": braced,
            }
        )
    return rows


def lever_rows(saved: dict, lever: dict, pitch_deg: float, force_scale: float = 1.0) -> dict:
    """Sample the lever task from the braced pose with the arm tucked, back to the same pose."""
    wall_x = saved["wall_x_m"]
    model, data = compile_robot(wall_x=wall_x)
    limits = torque_limits()
    names = tr.joint_names()
    address = np.array([model.joint(name).qposadr[0] for name in names])
    pitch = np.radians(pitch_deg)
    braced_config(model, data)
    offset = pusher_offset(model, data, pitch)
    stance = tr.segments(saved, arm_up=False)[-1]["stance"]
    base = tr.base_of(np.array(tr.reference_pose()["qpos"]))
    feet = {name: item[0] for name, item in stance.items()}
    contacts = tr.contacts_for(stance)
    end = np.radians(lever["target_deg"] + OVERSHOOT_DEG)
    hold = lever["tip_force_n"] * lever["handle_length_m"] / lever["contact_from_hinge_m"]
    push = PUSH_MARGIN * hold * force_scale
    first_top, first_normal = bar_top(lever, wall_x, 0.0)
    hover = first_top + HOVER_M * first_normal - offset
    pre = hover - np.array([APPROACH_RUN_M, 0.0, 0.0])
    last_top, last_normal = bar_top(lever, wall_x, end)
    plan = []
    for name in TIMES:
        count = int(round(TIMES[name] / tr.SAMPLE_S))
        for k in range(1, count + 1):
            s = k / count
            w = tr.smooth(s)
            force = np.zeros(3)
            if name == "reach":
                target = ("joints", s)
            elif name == "approach":
                target = ("tip", pre + w * (hover - pre))
            elif name == "settle":
                stop = first_top + GAP_M * first_normal - offset
                target = ("tip", hover + w * (stop - hover))
            elif name == "push":
                angle = w * end
                top, normal = bar_top(lever, wall_x, angle)
                target = ("tip", top + GAP_M * normal - offset)
                force = push * min(s / 0.1, 1.0) * normal
            elif name == "release":
                target = ("tip", last_top + GAP_M * last_normal - offset)
                force = push * (1 - w) * last_normal
            elif name == "lift":
                start = last_top + GAP_M * last_normal - offset
                target = ("tip", start + w * 0.05 * last_normal)
            else:
                target = ("back", s)
            plan.append((name, target, force))
    rows = {key: [] for key in ("qpos", "target", "torque", "ratio", "smooth_ratio", "tip")}
    joints = [model.joint(n) for n in PLANAR_ARM]
    tuck = tr.tuck_arm()
    first_solution, last_solution = None, None
    seeds = list(SEEDS)
    # Solve the tip path first so that "reach" and "back" can interpolate in joint space.
    tip_solutions = []
    tr.place(model, data, base, feet, tuck)
    for name, target, _ in plan:
        if target[0] == "tip":
            error = solve_arm_ik(model, data, target[1], pitch, seeds)
            if error > 1e-3:
                raise RuntimeError(f"lever path unreachable in {name}")
            solution = [float(data.qpos[j.qposadr[0]]) for j in joints]
            seeds = [tuple(solution)] + list(SEEDS)
            tip_solutions.append(solution)
            first_solution = first_solution or solution
            last_solution = solution
        else:
            tip_solutions.append(None)
    for (name, target, force), solution in zip(plan, tip_solutions):
        arm = dict(tuck)
        if target[0] == "joints":
            w = tr.smooth(target[1])
            for j, value in zip(PLANAR_ARM, first_solution):
                arm[j] = (1 - w) * tuck[j] + w * value
        elif target[0] == "back":
            w = tr.smooth(target[1])
            for j, value in zip(PLANAR_ARM, last_solution):
                arm[j] = (1 - w) * value + w * tuck[j]
        else:
            arm.update(dict(zip(PLANAR_ARM, solution)))
        if not tr.place(model, data, base, feet, arm):
            raise RuntimeError(f"lever pose collides in {name}")
        planned = solve_static(model, data, contacts, limits, tip_force=force)
        smooth = solve_static(model, data, contacts, limits, tip_force=force, smooth=True)
        if planned is None or smooth is None:
            raise RuntimeError(f"lever pose not balanced in {name}")
        rows["qpos"].append(data.qpos.copy())
        rows["target"].append(data.qpos[address].copy())
        rows["torque"].append(smooth.torque.copy())
        rows["ratio"].append(planned.ratio)
        rows["smooth_ratio"].append(smooth.ratio)
        rows["tip"].append(force)
    return {key: np.array(value) for key, value in rows.items()}, [p[0] for p in plan]


def mission(saved: dict, lever: dict, pitch_deg: float, force_scale: float = 1.0) -> dict:
    """Climb with the arm tucked, operate the lever, and climb back down."""
    up = tr.sample(saved, arm_up=False)
    task, names = lever_rows(saved, lever, pitch_deg, force_scale)
    count = len(task["qpos"])
    clock = up["time"][-1] + tr.SAMPLE_S * np.arange(1, count + 1)
    last = {key: value[-1] for key, value in up.items()}
    middle = {
        "time": clock,
        "qpos": task["qpos"],
        "target": task["target"],
        "torque": task["torque"],
        "ratio": task["ratio"],
        "smooth_ratio": task["smooth_ratio"],
        "stance": np.repeat(last["stance"][None], count, axis=0),
        "segment": np.array([100 + list(TIMES).index(n) for n in names]),
        "normal": np.repeat(last["normal"][None], count, axis=0),
        "feet": np.repeat(last["feet"][None], count, axis=0),
        "wall": np.repeat(last["wall"][None], count, axis=0),
        "cap": np.repeat(last["cap"][None], count, axis=0),
        "tip": task["tip"],
    }
    track = {key: np.concatenate([up[key], middle[key]]) for key in up}
    down = {key: value[-2::-1] for key, value in up.items()}
    end = track["time"][-1]
    down["time"] = end + (up["time"][-1] - up["time"][-2::-1])
    return {key: np.concatenate([track[key], down[key]]) for key in track}


def lever_angle(model, data) -> float:
    return float(data.qpos[model.joint("lever_hinge").qposadr[0]])


def evaluate_task(
    track: dict, log: dict, summary: dict, lever: dict, wall_mu: float, floor_mu: float
) -> dict:
    """M3 verdict from the lever window of the mission log."""
    window = np.isin(track["segment"], np.arange(100, 100 + len(TIMES)))
    start, stop = track["time"][window][[0, -1]]
    inside = (log["time"] >= start) & (log["time"] <= stop)
    angles = np.degrees(log["extra"])
    after = log["time"] > stop
    base_planned = track["qpos"][window][0][:3]
    base_shift = float(np.linalg.norm(log["base"][inside][:, :3] - base_planned, axis=1).max())
    feet = log["feet"][inside]
    slip = float(np.linalg.norm(feet - feet[0], axis=2).max())
    wall = track["wall"][window][0]
    friction = 0.0
    for k in range(4):
        force = log["force"][inside][:, k]
        axis = 0 if wall[k] else 2
        normal = np.abs(force[:, axis])
        tangent = np.linalg.norm(np.delete(force, axis, axis=1), axis=1)
        mu = wall_mu if wall[k] else floor_mu
        ok = normal > tr.FRICTION_MIN_NORMAL_N
        if ok.any():
            friction = max(friction, float((tangent[ok] / (mu * normal[ok])).max()))
    limits = torque_limits()
    ratio = float((np.abs(log["torque"][inside]) / limits).max())
    reached = float(angles[inside].max())
    before = float(np.abs(angles[log["time"] < start]).max())
    kept = float(angles[after].min()) if after.any() else float(angles[-1])
    target = lever["target_deg"]
    passed = (
        before <= 1.0
        and reached >= target
        and kept >= target - 2.0
        and base_shift <= HOLD_LIMITS["base_m"]
        and slip <= HOLD_LIMITS["slip_m"]
        and friction <= HOLD_LIMITS["friction"]
        and ratio <= 1.0
        and not summary["fell"]
    )
    return {
        "passed": bool(passed),
        "lever_before_task_deg": before,
        "lever_max_deg": reached,
        "lever_after_deg": kept,
        "target_deg": target,
        "max_base_shift_mm": base_shift * 1e3,
        "max_foot_slip_mm": slip * 1e3,
        "max_friction_use": friction,
        "max_ratio": ratio,
        "window_s": [float(start), float(stop)],
    }


def run(
    saved: dict,
    lever: dict | None = None,
    pitch_deg: float = 20.0,
    floor_mu: float = 0.6,
    wall_mu: float = 0.6,
    case: str = "scaled",
    wall_shift: float = 0.0,
    force_scale: float = 1.0,
    physics: dict | None = None,
) -> tuple[dict, dict, dict]:
    """Run the whole mission. wall_shift moves the true wall and lever, not the plan."""
    lever = lever or lever_parameters()
    track = mission(saved, lever, pitch_deg, force_scale)
    wall_x = saved["wall_x_m"]

    def builder():
        return build_scene(wall_x + wall_shift, floor_mu, wall_mu, case, lever)

    summary, log = tr.execute(
        track,
        wall_x,
        floor_mu=floor_mu,
        wall_mu=wall_mu,
        physics=physics,
        builder=builder,
        watch=lever_angle,
    )
    task = evaluate_task(track, log, summary, lever, wall_mu, floor_mu)
    return {"round_trip": summary, "lever": task}, track, log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--choose", action="store_true", help="only scan hinge heights")
    parser.add_argument("--pitch", type=float, default=20.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    saved = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.choose:
        rows = choose_scene(saved["wall_x_m"])
        reach = four_leg_any_pitch(saved["wall_x_m"])
        (args.out / "scene_scan.json").write_text(
            json.dumps({"braced_and_four_leg": rows, "four_leg_any_pitch": reach}, indent=1),
            encoding="utf-8",
        )
        print(reach)
        for row in rows:
            print(row)
        return
    result, track, log = run(saved, pitch_deg=args.pitch)
    (args.out / "mission.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    np.savez_compressed(
        args.out / "mission.npz", **track, **{f"log_{k}": v for k, v in log.items()}
    )
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
