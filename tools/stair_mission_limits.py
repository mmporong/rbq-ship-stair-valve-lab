"""Named 18-axis torque limits for the RBQ-10 + RB1 stair mission."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np

from go2_wall.robot import ARM_JOINTS, RB1_JOINTS
from go2_wall.statics import joint_names, torque_limits

NOMINAL_LIMIT_SET = "rb1"
LIMS_EX_LIMIT_SET = "lims-ex"

RB1_SOURCE_JOINTS = (
    "RB1_Base_Yaw_joint",
    "RB1_Shoulder_Pitch_joint",
    "RB1_Elbow_Pitch_joint",
    "RB1_Elbow_Yaw_joint",
    "RB1_Wrist_Pitch_joint",
    "RB1_Wrist_Yaw_joint",
)
LIMS_EX_NAMED_LIMITS_NM = (
    ("LIMS_EX_Shoulder_Yaw_Joint", 92.5),
    ("LIMS_EX_Shoulder_Pitch_Joint", 85.0),
    ("LIMS_EX_Elbow_Joint_1", 42.5),
    ("LIMS_EX_Wrist_Pitch_Joint", 15.2),
    ("LIMS_EX_Wrist_Roll_Joint", 7.6),
    ("LIMS_EX_Wrist_Yaw_Joint", 15.2),
)


def mission_torque_limits(limit_set: str = NOMINAL_LIMIT_SET) -> np.ndarray:
    """Return the 18 controller-axis limits, optionally replacing the six arm axes."""
    nominal = torque_limits("rb1").astype(float, copy=True)
    if limit_set == NOMINAL_LIMIT_SET:
        return nominal
    if limit_set == LIMS_EX_LIMIT_SET:
        nominal[-len(ARM_JOINTS) :] = [limit for _, limit in LIMS_EX_NAMED_LIMITS_NM]
        return nominal
    raise ValueError(f"unknown arm limit set: {limit_set!r}")


def assert_rb1_motor_order(model: mujoco.MjModel) -> None:
    """Assert source-name and compiled 18-axis motor ordering before applying limits."""
    if tuple(RB1_JOINTS) != RB1_SOURCE_JOINTS:
        raise AssertionError("robot.RB1_JOINTS no longer matches the pinned RB1 source order")
    expected_joints = tuple(joint_names())
    if model.nu < len(expected_joints):
        raise AssertionError(
            f"model has {model.nu} actuators; expected at least {len(expected_joints)}"
        )
    actual_joints = tuple(model.joint(model.actuator(i).trnid[0]).name for i in range(18))
    if actual_joints != expected_joints:
        raise AssertionError(f"18-axis motor order differs: {actual_joints!r}")
    expected_motors = tuple(name.replace("_joint", "_motor") for name in expected_joints)
    actual_motors = tuple(model.actuator(i).name for i in range(18))
    if actual_motors != expected_motors:
        raise AssertionError(f"18-axis motor names differ: {actual_motors!r}")


def apply_motor_torque_ranges(
    model: mujoco.MjModel, limits_nm: Sequence[float]
) -> np.ndarray:
    """Apply symmetric torque control ranges to the ordered 18 robot motors."""
    limits = np.asarray(limits_nm, dtype=float)
    if limits.shape != (18,) or not np.all(np.isfinite(limits)) or np.any(limits <= 0):
        raise ValueError("limits_nm must contain 18 finite positive values")
    assert_rb1_motor_order(model)
    for actuator_id, limit in enumerate(limits):
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            raise AssertionError(f"actuator {actuator_id} is not joint-transmitted")
        gear = model.actuator_gear[actuator_id, 0]
        if not np.isclose(gear, 1.0):
            raise AssertionError(f"actuator {actuator_id} gear is {gear}, expected 1")
        model.actuator_ctrlrange[actuator_id] = [-limit, limit]
        model.actuator_ctrllimited[actuator_id] = mujoco.mjtLimited.mjLIMITED_TRUE
    return limits.copy()


def configure_mission_torque_limits(
    model: mujoco.MjModel, limit_set: str = NOMINAL_LIMIT_SET
) -> np.ndarray:
    """Resolve a named limit set and apply it to the compiled mission model."""
    return apply_motor_torque_ranges(model, mission_torque_limits(limit_set))
