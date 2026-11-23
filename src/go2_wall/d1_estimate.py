"""Estimate D1 joint torque capacity from the official rated payload and reach.

The official page gives no joint torque. A joint must at least hold the rated payload at full
reach, so the gravity torque in that pose is a lower bound on its capacity.
"""

import json
import sys

import mujoco
import numpy as np

from .robot import ARM_JOINTS, build_arm_only, load_parameters

GRAVITY = np.array([0.0, 0.0, -9.81])
# At zero the upper arm points up and the forearm forward. J2 = +90 deg and J3 = -90 deg lay
# both links out in a straight horizontal line.
RATED_POSE = {"arm_joint2": np.pi / 2, "arm_joint3": -np.pi / 2}


def horizontal_reach(case: str = "scaled") -> dict:
    """Measure the kinematic reach from the J2 axis in the rated pose."""
    model, data = build_arm_only(case)
    for name, value in RATED_POSE.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    shoulder = data.xanchor[model.joint("arm_joint2").id]
    wrist = data.xanchor[model.joint("arm_joint6").id]
    tip = data.site("arm_tip").xpos
    return {
        "to_joint6_m": float(np.linalg.norm((wrist - shoulder)[[0, 2]])),
        "to_tip_m": float(np.linalg.norm((tip - shoulder)[[0, 2]])),
        "tip_height_drop_m": float(abs(tip[2] - shoulder[2])),
    }


def gravity_torques(case: str, payload_kg: float) -> np.ndarray:
    """Return the holding torque of joints 1 to 6 in the rated pose."""
    model, data = build_arm_only(case)
    for name, value in RATED_POSE.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    jacobian = np.zeros((3, model.nv))
    site = model.site("arm_payload").id
    mujoco.mj_jacSite(model, data, jacobian, None, site)
    torque = data.qfrc_bias - jacobian.T @ (payload_kg * GRAVITY)
    return np.array([torque[model.joint(name).dofadr[0]] for name in ARM_JOINTS])


def estimate() -> dict:
    params = load_parameters()
    payload = params["rated_payload_kg"]["value"]
    cases = {case: gravity_torques(case, payload) for case in ("scaled", "servo_lumped")}
    needed = np.max([np.abs(value) for value in cases.values()], axis=0)
    shoulder_class = float(max(needed[:3]))
    wrist_class = float(max(needed[3:]))
    public = params["joint_torque_limit_public_nm"]
    return {
        "rated_hold_torque_nm": {key: np.round(value, 4).tolist() for key, value in cases.items()},
        "torque_lower_bound_nm": [shoulder_class] * 3 + [wrist_class] * 3,
        "method": "Largest rated-pose holding torque over both mass cases, grouped by the two "
        "torque classes of the URDF (joints 1-3, joints 4-6)",
        "public_limits_nm": {"urdf": public["urdf"], "osu": public["osu"]},
        "public_limit_ratio_to_lower_bound": {
            key: round(public[key][1] / shoulder_class, 3) for key in ("urdf", "osu")
        },
        "reach": horizontal_reach(),
    }


if __name__ == "__main__":
    result = estimate()
    text = json.dumps(result, indent=2)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)
