"""Hold a wall-braced pose in MuJoCo contact physics and record loads."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .pose_search import braced_contacts
from .robot import LEGS, compile_robot
from .statics import joint_names, solve_static, torque_limits

# Gains are assumptions for a stiff pose hold. Their effect is compared before results are final.
LEG_GAINS = (60.0, 2.0)
ARM_GAINS = (30.0, 1.0)
SLIP_LIMIT_M = 0.005
BASE_LIMIT_M = 0.010
FRICTION_USE_LIMIT = 0.8


def apply_physics(model: mujoco.MjModel, physics: dict) -> None:
    """Override simulator settings to check that results do not depend on one setting.

    Keys: integrator (name such as "discrete"), timestep, impratio, noslip_iterations,
    contact_solref (time constant applied to every geom). The defaults come from the Go2
    Menagerie model.
    """
    if "timestep" in physics:
        model.opt.timestep = physics["timestep"]
    if "integrator" in physics:
        model.opt.integrator = getattr(
            mujoco.mjtIntegrator, f"mjINT_{physics['integrator'].upper()}"
        )
    if "impratio" in physics:
        model.opt.impratio = physics["impratio"]
    if "noslip_iterations" in physics:
        model.opt.noslip_iterations = physics["noslip_iterations"]
    if "contact_solref" in physics:
        model.geom_solref[:, 0] = physics["contact_solref"]


def contact_forces(model, data) -> dict[str, np.ndarray]:
    """Sum world-frame contact force on each foot from floor and wall contacts."""
    feet = {model.geom(leg).id: leg for leg in LEGS}
    totals = {leg: np.zeros(3) for leg in LEGS}
    other = {}
    wrench = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        mujoco.mj_contactForce(model, data, index, wrench)
        force = contact.frame.reshape(3, 3).T @ wrench[:3]
        for geom, sign in ((contact.geom2, 1.0), (contact.geom1, -1.0)):
            if geom in feet:
                totals[feet[geom]] += sign * force
                break
        else:
            names = sorted(
                model.body(model.geom_bodyid[g]).name for g in (contact.geom1, contact.geom2)
            )
            key = "+".join(names)
            other[key] = max(other.get(key, 0.0), float(np.linalg.norm(force)))
    return totals, other


def run(
    pose: dict,
    case: str,
    floor_mu: float,
    wall_mu: float,
    push: float,
    arm: str,
    seconds: float,
    gains_scale: float = 1.0,
    physics: dict | None = None,
) -> dict:
    model, data = compile_robot(
        case=case,
        wall_x=pose["wall_x_m"],
        floor_mu=floor_mu,
        wall_mu=wall_mu,
        mount_x=pose.get("mount_x_m"),
    )
    apply_physics(model, physics or {})
    limits = torque_limits(arm)
    qpos = np.array(pose["qpos"])
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    tip_force = np.array([-push, 0.0, 0.0])
    static = solve_static(model, data, braced_contacts(floor_mu, wall_mu), limits, tip_force)
    if static is None:
        return {
            "passed": False,
            "infeasible": True,
            "case": case,
            "arm_limit_set": arm,
            "floor_mu": floor_mu,
            "wall_mu": wall_mu,
            "push_n": push,
        }
    data.qpos[:] = qpos
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    names = joint_names()
    position = np.array([model.joint(name).qposadr[0] for name in names])
    velocity = np.array([model.joint(name).dofadr[0] for name in names])
    actuator = np.r_[np.arange(12), np.arange(12, 18)]
    target = qpos[position]
    kp = np.r_[np.full(12, LEG_GAINS[0]), np.full(6, ARM_GAINS[0])] * gains_scale
    kd = np.r_[np.full(12, LEG_GAINS[1]), np.full(6, ARM_GAINS[1])] * gains_scale
    site = model.site("arm_tip").id
    wrist = model.site_bodyid[site]
    foot_start = {leg: data.geom_xpos[model.geom(leg).id].copy() for leg in LEGS}
    base_start = data.qpos[:3].copy()
    tilt_start = data.qpos[3:7].copy()
    steps = int(seconds / model.opt.timestep)
    torques, forces, ratios = [], {leg: [] for leg in LEGS}, []
    max_slip, max_base, max_tilt, saturated = 0.0, 0.0, 0.0, 0
    other_contacts: dict[str, float] = {}
    for step in range(steps):
        error = target - data.qpos[position]
        command = static.torque + kp * error - kd * data.qvel[velocity]
        command = np.clip(command, -limits, limits)
        saturated += int(np.any(np.abs(command) >= limits - 1e-9))
        data.ctrl[actuator] = command
        data.qfrc_applied[:] = 0
        mujoco.mj_applyFT(
            model, data, tip_force, np.zeros(3), data.site_xpos[site], wrist, data.qfrc_applied
        )
        mujoco.mj_step(model, data)
        for leg in LEGS:
            shift = data.geom_xpos[model.geom(leg).id] - foot_start[leg]
            tangent = shift[[0, 1]] if leg[0] == "R" else shift[[1, 2]]
            max_slip = max(max_slip, float(np.linalg.norm(tangent)))
        max_base = max(max_base, float(np.linalg.norm(data.qpos[:3] - base_start)))
        difference = np.zeros(3)
        mujoco.mju_subQuat(difference, data.qpos[3:7], tilt_start)
        max_tilt = max(max_tilt, float(np.linalg.norm(difference)))
        if step % 10 == 0:
            mujoco.mj_forward(model, data)
            applied = data.actuator_force[actuator].copy()
            torques.append(applied)
            ratios.append(np.abs(applied) / limits)
            totals, other = contact_forces(model, data)
            for leg in LEGS:
                forces[leg].append(totals[leg])
            for key, value in other.items():
                other_contacts[key] = max(other_contacts.get(key, 0.0), value)
    torque = np.array(torques)
    settle = len(torque) // 5
    final = {leg: np.array(values)[settle:] for leg, values in forces.items()}
    mean_force = {leg: value.mean(axis=0) for leg, value in final.items()}
    vertical = sum(item[2] for item in mean_force.values())
    weight = float(model.body_subtreemass[1] * 9.81)
    friction = {}
    for leg, value in final.items():
        normal = np.abs(value[:, 0]) if leg[0] == "F" else value[:, 2]
        tangent = np.linalg.norm(value[:, [1, 2]] if leg[0] == "F" else value[:, [0, 1]], axis=1)
        mu = wall_mu if leg[0] == "F" else floor_mu
        friction[leg] = float(np.max(tangent / np.maximum(mu * normal, 1e-6)))
    fell = max_base > 0.1
    wall_contact = min(np.abs(v[:, 0]).min() for k, v in final.items() if k[0] == "F") > 0
    passed = (
        max_slip <= SLIP_LIMIT_M
        and max_base <= BASE_LIMIT_M
        and not fell
        and saturated == 0
        and wall_contact
        and max(friction.values()) <= FRICTION_USE_LIMIT
    )
    return {
        "passed": bool(passed),
        "case": case,
        "arm_limit_set": arm,
        "floor_mu": floor_mu,
        "wall_mu": wall_mu,
        "push_n": push,
        "seconds": seconds,
        "gains_scale": gains_scale,
        "physics": physics or {},
        "tip_height_m": pose["tip_height_m"],
        "tip_height_measured_m": float(data.site_xpos[site][2]),
        "max_foot_slip_mm": max_slip * 1e3,
        "max_base_shift_mm": max_base * 1e3,
        "max_tilt_deg": float(np.degrees(max_tilt)),
        "saturated_steps": saturated,
        "weight_n": weight,
        "mean_vertical_support_n": float(vertical),
        "static_ratio": static.ratio,
        "max_ratio": float(np.array(ratios).max()),
        "joint": {
            name: {
                "mean_abs_nm": float(np.abs(torque[settle:, i]).mean()),
                "max_abs_nm": float(np.abs(torque[settle:, i]).max()),
                "limit_nm": float(limits[i]),
                "max_ratio": float(np.abs(torque[settle:, i]).max() / limits[i]),
                "static_nm": float(static.torque[i]),
            }
            for i, name in enumerate(names)
        },
        "mean_contact_force_n": {k: np.round(v, 3).tolist() for k, v in mean_force.items()},
        "max_friction_use": friction,
        "friction_limit": FRICTION_USE_LIMIT,
        "unexpected_contacts_n": other_contacts,
    }


def max_static_push(pose: dict, case: str, floor_mu: float, wall_mu: float, arm: str) -> dict:
    """Find the largest wall push the fixed pose can balance, and the joint that limits it."""
    model, data = compile_robot(
        case=case,
        wall_x=pose["wall_x_m"],
        floor_mu=floor_mu,
        wall_mu=wall_mu,
        mount_x=pose.get("mount_x_m"),
    )
    limits = torque_limits(arm)
    contacts = braced_contacts(floor_mu, wall_mu)
    data.qpos[:] = np.array(pose["qpos"])

    def solve(push: float):
        return solve_static(model, data, contacts, limits, np.array([-push, 0.0, 0.0]))

    low, high = 0.0, 200.0
    if solve(low) is None:
        return {"max_push_n": None}
    for _ in range(40):
        middle = (low + high) / 2
        solution = solve(middle)
        low, high = (middle, high) if solution and solution.ratio <= 1.0 else (low, middle)
    solution = solve(low)
    names = joint_names()
    index = int(np.argmax(np.abs(solution.torque) / limits))
    return {
        "max_push_n": low,
        "limiting_joint": names[index],
        "limiting_torque_nm": float(solution.torque[index]),
    }


def render(pose: dict, case: str, path: Path) -> None:
    model, data = compile_robot(case=case, wall_x=pose["wall_x_m"], mount_x=pose.get("mount_x_m"))
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 960
    data.qpos[:] = np.array(pose["qpos"])
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 960, 1280)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.45, 0, 0.55]
    camera.distance, camera.azimuth, camera.elevation = 2.2, 90, -8
    renderer.update_scene(data, camera)
    pixels = renderer.render()
    renderer.close()
    try:
        from PIL import Image

        Image.fromarray(pixels).save(path)
    except ImportError:
        import struct
        import zlib

        height, width, _ = pixels.shape
        raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))

        def chunk(kind: bytes, payload: bytes) -> bytes:
            body = kind + payload
            return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

        path.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--cap", default="0.5")
    parser.add_argument("--case", default="scaled")
    parser.add_argument("--floor-mu", type=float, default=0.6)
    parser.add_argument("--wall-mu", type=float, default=0.6)
    parser.add_argument("--push", type=float, default=5.0)
    parser.add_argument("--arm", default="derived")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--gains-scale", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    pose = json.loads(args.pose.read_text(encoding="utf-8"))["best_by_cap"][args.cap]
    result = run(
        pose,
        args.case,
        args.floor_mu,
        args.wall_mu,
        args.push,
        args.arm,
        args.seconds,
        args.gains_scale,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    if args.image:
        render(pose, args.case, args.image)
    keys = (
        "passed",
        "tip_height_measured_m",
        "max_foot_slip_mm",
        "max_base_shift_mm",
        "max_tilt_deg",
        "saturated_steps",
        "weight_n",
        "mean_vertical_support_n",
        "static_ratio",
        "max_ratio",
    )
    print({key: result[key] for key in keys})
    print("contacts", result["mean_contact_force_n"], "friction", result["max_friction_use"])
    print("unexpected", result["unexpected_contacts_n"])


if __name__ == "__main__":
    main()
