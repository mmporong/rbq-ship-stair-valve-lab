"""Search wall-braced poses, using real arm kinematics and collision checks."""

import argparse
import itertools
import json
import multiprocessing
from pathlib import Path

import mujoco
import numpy as np

from .robot import LEG_PARTS, LEGS, compile_robot, load_parameters, resolve_mount, world_geoms
from .statics import FLOOR_NORMAL, WALL_NORMAL, Contact, joint_names, solve_static, torque_limits

HIP_X = 0.1934
FOOT_Y = 0.142
PLANAR_ARM = ["arm_joint2", "arm_joint3", "arm_joint5"]
TIP_CLEARANCE = 0.01
FRICTION_MARGIN = 0.7
# The body is about 0.33 m long from centre to nose, so skip poses that clearly reach the wall.
PITCHES = range(30, 86, 5)
HIP_HEIGHTS = (0.28, 0.32, 0.36, 0.385)
WALL_OFFSETS = (0.05, 0.12, 0.20, 0.28)
FOOT_HEIGHTS = (-0.15, -0.05, 0.05)
REAR_OFFSETS = (-0.05, 0.05, 0.15)
TIP_PITCHES = (0.0, 20.0, 40.0, 60.0, 80.0)


def set_base(
    model, data, pitch: float, position: np.ndarray, *, yaw: float = 0.0, roll: float = 0.0
) -> None:
    """Set the free base, applying roll about the pitched and yawed body's forward axis."""
    pitch_cos, pitch_sin = np.cos(pitch / 2), np.sin(-pitch / 2)
    yaw_cos, yaw_sin = np.cos(yaw / 2), np.sin(yaw / 2)
    data.qpos[:3] = position
    data.qpos[3:7] = [
        yaw_cos * pitch_cos,
        -yaw_sin * pitch_sin,
        yaw_cos * pitch_sin,
        yaw_sin * pitch_cos,
    ]
    if roll:
        roll_quat = np.array([np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0])
        mujoco.mju_mulQuat(data.qpos[3:7], data.qpos[3:7].copy(), roll_quat)


def set_legs(model, data, values: tuple[float, float, float] = (0.0, 0.9, -1.8)) -> None:
    for leg in LEGS:
        for part, value in zip(LEG_PARTS, values):
            data.qpos[model.joint(f"{leg}_{part}_joint").qposadr[0]] = value


def solve_leg_ik(
    model, data, targets: dict[str, np.ndarray], iterations: int = 60, reset: bool = True
) -> float:
    """Place each foot centre by damped least squares inside the joint limits.

    With reset=False the solve starts from the leg angles already in data, which keeps a
    sequence of poses on one branch.
    """
    if reset:
        set_legs(model, data)
    error = np.inf
    for _ in range(iterations):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        error = 0.0
        for leg, target in targets.items():
            geom = model.geom(leg).id
            gap = target - data.geom_xpos[geom]
            error = max(error, float(np.linalg.norm(gap)))
            jacobian = np.zeros((3, model.nv))
            mujoco.mj_jacGeom(model, data, jacobian, None, geom)
            ids = [model.joint(f"{leg}_{part}_joint") for part in LEG_PARTS]
            local = jacobian[:, [item.dofadr[0] for item in ids]]
            step = local.T @ np.linalg.solve(local @ local.T + 1e-4 * np.eye(3), gap)
            for item, delta in zip(ids, step):
                address = item.qposadr[0]
                value = data.qpos[address] + np.clip(delta, -0.3, 0.3)
                data.qpos[address] = np.clip(value, *item.range)
        if error < 1e-4:
            break
    mujoco.mj_kinematics(model, data)
    return error


def solve_arm_ik(model, data, target: np.ndarray, tip_pitch: float, seeds: list) -> float:
    """Place the tip in the sagittal plane with joints 2, 3, 5, and aim it up by tip_pitch."""
    joints = [model.joint(name) for name in PLANAR_ARM]
    site = model.site("arm_tip").id
    best = (np.inf, None)
    for seed in seeds:
        for item, value in zip(joints, seed):
            data.qpos[item.qposadr[0]] = value
        for _ in range(40):
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            position, rotation = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, position, rotation, site)
            axis = data.site_xmat[site].reshape(3, 3)[:, 0]
            current = np.arctan2(axis[2], axis[0])
            gap = np.r_[(target - data.site_xpos[site])[[0, 2]], tip_pitch - current]
            dofs = [item.dofadr[0] for item in joints]
            local = np.vstack([position[[0, 2]][:, dofs], -rotation[1, dofs]])
            step = local.T @ np.linalg.solve(local @ local.T + 1e-3 * np.eye(3), gap)
            for item, delta in zip(joints, step):
                address = item.qposadr[0]
                data.qpos[address] = np.clip(
                    data.qpos[address] + np.clip(delta, -0.3, 0.3), *item.range
                )
            if np.abs(gap).max() < 5e-4:
                break
        mujoco.mj_kinematics(model, data)
        error = float(np.linalg.norm((target - data.site_xpos[site])[[0, 2]]))
        if error < best[0]:
            best = (error, [data.qpos[item.qposadr[0]] for item in joints])
    if best[1] is None:
        return np.inf
    for item, value in zip(joints, best[1]):
        data.qpos[item.qposadr[0]] = value
    mujoco.mj_kinematics(model, data)
    return best[0]


def clear_of_collisions(model, data, tolerance: float = 1e-4) -> bool:
    """Return True if only the feet touch the floor or the wall."""
    mujoco.mj_forward(model, data)
    allowed = {model.geom(leg).id for leg in LEGS}
    world = set(world_geoms(model))
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.dist > -tolerance:
            continue
        pair = {contact.geom1, contact.geom2}
        if pair & allowed and pair & world:
            continue
        return False
    return True


def braced_contacts(floor_mu: float, wall_mu: float) -> list[Contact]:
    """Plan with FRICTION_MARGIN of the true friction so feet do not sit on the slip limit."""
    floor, wall = floor_mu * FRICTION_MARGIN, wall_mu * FRICTION_MARGIN
    return [
        Contact("FL", WALL_NORMAL, wall),
        Contact("FR", WALL_NORMAL, wall),
        Contact("RL", FLOOR_NORMAL, floor),
        Contact("RR", FLOOR_NORMAL, floor),
    ]


def highest_tip(model, data, wall_x: float, step: float = 0.02) -> tuple[float, float] | None:
    """Scan down from the top of the arm's range and return the first reachable clean height."""
    seeds = [(1.2, -1.2, 0.3), (1.0, -0.5, 0.0), (0.5, -1.0, 0.0)]
    shoulder = data.xanchor[model.joint("arm_joint2").id]
    x = wall_x - TIP_CLEARANCE
    reach = load_parameters()["reach_with_gripper_m"]["value"] + 0.015
    run = x - shoulder[0]
    if run >= reach:
        return None
    top = shoulder[2] + np.sqrt(reach * reach - run * run)
    for height in np.arange(top, shoulder[2] - 0.1, -step):
        for pitch in np.radians(TIP_PITCHES):
            target = np.array([x, 0.0, height])
            if solve_arm_ik(
                model, data, target, float(pitch), seeds
            ) < 1e-3 and clear_of_collisions(model, data):
                return float(height), float(pitch)
    return None


def search_pitch(
    degrees: int,
    case: str,
    floor_mu: float,
    wall_mu: float,
    push: float,
    arm: str,
    wall_x: float,
    mount_x: float | None = None,
) -> list[dict]:
    mount_x = resolve_mount(mount_x)
    model, data = compile_robot(
        case=case, wall_x=wall_x, floor_mu=floor_mu, wall_mu=wall_mu, mount_x=mount_x
    )
    limits = torque_limits(arm)
    contacts = braced_contacts(floor_mu, wall_mu)
    radius = model.geom_size[model.geom("FL").id][0]
    force = np.array([-push, 0.0, 0.0])
    found = []
    for hip, reach, rise, rear in itertools.product(
        HIP_HEIGHTS, WALL_OFFSETS, FOOT_HEIGHTS, REAR_OFFSETS
    ):
        pitch = np.radians(degrees)
        cos, sin = np.cos(pitch), np.sin(pitch)
        front_hip_x = wall_x - reach
        centre = np.array([front_hip_x - HIP_X * cos, 0.0, hip + HIP_X * sin])
        rear_hip_x = centre[0] - HIP_X * cos
        data.qpos[:] = model.qpos0
        set_base(model, data, pitch, centre)
        targets = {
            "FL": np.array([wall_x - radius, FOOT_Y, centre[2] + HIP_X * sin + rise]),
            "FR": np.array([wall_x - radius, -FOOT_Y, centre[2] + HIP_X * sin + rise]),
            "RL": np.array([rear_hip_x + rear, FOOT_Y, radius]),
            "RR": np.array([rear_hip_x + rear, -FOOT_Y, radius]),
        }
        if solve_leg_ik(model, data, targets) > 2e-3:
            continue
        for name, value in (("arm_joint2", -0.8), ("arm_joint3", -1.2)):
            data.qpos[model.joint(name).qposadr[0]] = value
        if not clear_of_collisions(model, data):
            continue
        result = highest_tip(model, data, wall_x)
        if result is None:
            continue
        height, tip_pitch = result
        solution = solve_static(model, data, contacts, limits, force)
        if solution is None:
            continue
        found.append(
            {
                "pitch_deg": degrees,
                "hip_height_m": hip,
                "wall_offset_m": reach,
                "foot_rise_m": rise,
                "rear_offset_m": rear,
                "tip_height_m": height,
                "tip_pitch_rad": tip_pitch,
                "ratio": solution.ratio,
                "residual_n": solution.residual,
                "torque_nm": dict(zip(joint_names(), np.round(solution.torque, 3).tolist())),
                "forces_n": {k: np.round(v, 3).tolist() for k, v in solution.forces.items()},
                "qpos": data.qpos.tolist(),
                "wall_x_m": wall_x,
                "mount_x_m": mount_x,
            }
        )
    return found


def four_leg_reach(case: str, mount_x: float | None = None, push: float = 5.0) -> dict:
    """Return the highest wall point reachable with all four feet on the floor."""
    mount_x = resolve_mount(mount_x)
    results = {}
    contacts = [Contact(leg, FLOOR_NORMAL, 0.6 * FRICTION_MARGIN) for leg in LEGS]
    for name, height in (("nominal_0.33", 0.33), ("tall_0.385", 0.385)):
        wall_x = 0.60
        model, data = compile_robot(case=case, wall_x=wall_x, mount_x=mount_x)
        data.qpos[:] = model.qpos0
        radius = model.geom_size[model.geom("FL").id][0]
        nose = 0.343
        centre = np.array([wall_x - nose - 0.01, 0.0, height])
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
        error = solve_leg_ik(model, data, targets)
        for joint, value in (("arm_joint2", 0.0), ("arm_joint3", -1.2)):
            data.qpos[model.joint(joint).qposadr[0]] = value
        tip = highest_tip(model, data, wall_x)
        solution = (
            solve_static(model, data, contacts, torque_limits(), np.array([-push, 0.0, 0.0]))
            if tip
            else None
        )
        results[name] = {
            "ik_error_m": error,
            "tip_height_m": None if tip is None else tip[0],
            "tip_pitch_rad": None if tip is None else tip[1],
            "ratio": None if solution is None else solution.ratio,
        }
    return results


def search(
    case: str,
    floor_mu: float,
    wall_mu: float,
    push: float,
    arm: str,
    wall_x: float = 0.75,
    mount_x: float | None = None,
) -> list[dict]:
    mount_x = resolve_mount(mount_x)
    jobs = [(degrees, case, floor_mu, wall_mu, push, arm, wall_x, mount_x) for degrees in PITCHES]
    with multiprocessing.Pool(len(jobs)) as pool:
        return [item for part in pool.starmap(search_pitch, jobs) for item in part]


def pick(found: list[dict], cap: float) -> dict | None:
    allowed = [item for item in found if item["ratio"] <= cap]
    return (
        max(allowed, key=lambda item: (item["tip_height_m"], -item["ratio"])) if allowed else None
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="scaled")
    parser.add_argument("--floor-mu", type=float, default=0.6)
    parser.add_argument("--wall-mu", type=float, default=0.6)
    parser.add_argument("--push", type=float, default=5.0)
    parser.add_argument("--arm", default="derived")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    found = search(args.case, args.floor_mu, args.wall_mu, args.push, args.arm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {key: str(value) for key, value in vars(args).items()},
        "solutions": len(found),
        "four_leg_reach": four_leg_reach(args.case),
        "best_by_cap": {str(cap): pick(found, cap) for cap in (0.3, 0.5, 0.7, 1.0)},
    }
    args.out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("solutions", len(found), summary["four_leg_reach"])
    for cap, item in summary["best_by_cap"].items():
        if item is None:
            print(cap, "none")
            continue
        print(
            cap,
            {
                key: item[key]
                for key in ("tip_height_m", "pitch_deg", "hip_height_m", "ratio", "tip_pitch_rad")
            },
        )


if __name__ == "__main__":
    main()
