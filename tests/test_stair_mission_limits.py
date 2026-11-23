import importlib.util
from pathlib import Path

import numpy as np
import pytest

from go2_wall.robot import ARM_JOINTS, RB1_JOINTS, compile_robot
from go2_wall.statics import joint_names, torque_limits

MODULE_PATH = Path(__file__).parents[1] / "tools/stair_mission_limits.py"
SPEC = importlib.util.spec_from_file_location("stair_mission_limits", MODULE_PATH)
assert SPEC and SPEC.loader
limits_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(limits_module)

LIMS_EX_NAMED_LIMITS_NM = limits_module.LIMS_EX_NAMED_LIMITS_NM
RB1_SOURCE_JOINTS = limits_module.RB1_SOURCE_JOINTS
apply_motor_torque_ranges = limits_module.apply_motor_torque_ranges
assert_rb1_motor_order = limits_module.assert_rb1_motor_order
configure_mission_torque_limits = limits_module.configure_mission_torque_limits
mission_torque_limits = limits_module.mission_torque_limits


@pytest.fixture(scope="module")
def rbq_model():
    model, _ = compile_robot(wall_x=None, arm="rb1")
    return model


def test_nominal_limits_are_the_existing_18_axis_rbq_limits():
    limits = mission_torque_limits("rb1")

    assert limits.shape == (18,)
    np.testing.assert_array_equal(limits, torque_limits("rb1"))


def test_lims_ex_replaces_only_the_six_arm_axes_in_named_source_order():
    nominal = mission_torque_limits("rb1")
    limits = mission_torque_limits("lims-ex")

    np.testing.assert_array_equal(limits[:12], nominal[:12])
    np.testing.assert_array_equal(
        limits[12:], [92.5, 85.0, 42.5, 15.2, 7.6, 15.2]
    )
    assert tuple(name for name, _ in LIMS_EX_NAMED_LIMITS_NM) == (
        "LIMS_EX_Shoulder_Yaw_Joint",
        "LIMS_EX_Shoulder_Pitch_Joint",
        "LIMS_EX_Elbow_Joint_1",
        "LIMS_EX_Wrist_Pitch_Joint",
        "LIMS_EX_Wrist_Roll_Joint",
        "LIMS_EX_Wrist_Yaw_Joint",
    )


def test_pinned_rb1_source_names_map_to_compiled_arm_order(rbq_model):
    assert tuple(RB1_JOINTS) == RB1_SOURCE_JOINTS
    assert tuple(joint_names()[-6:]) == tuple(ARM_JOINTS)
    assert_rb1_motor_order(rbq_model)


def test_configure_applies_actual_motor_control_ranges(rbq_model):
    limits = configure_mission_torque_limits(rbq_model, "lims-ex")

    np.testing.assert_array_equal(rbq_model.actuator_ctrlrange[:18, 0], -limits)
    np.testing.assert_array_equal(rbq_model.actuator_ctrlrange[:18, 1], limits)
    assert np.all(rbq_model.actuator_ctrllimited[:18])


def test_invalid_limit_set_and_vector_are_rejected(rbq_model):
    with pytest.raises(ValueError, match="unknown arm limit set"):
        mission_torque_limits("unknown")
    with pytest.raises(ValueError, match="18 finite positive"):
        apply_motor_torque_ranges(rbq_model, np.ones(17))
