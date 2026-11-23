from copy import deepcopy

import numpy as np
import pytest

from go2_wall import grasp_motion
from go2_wall.open_valve_grasp import opening_candidate

CLOSING = {
    "base_pose": [0.5, 0.03, 1.0625, 0.61],
    "start_deg": 270.0,
    "sense": -1,
    "centre_m": [0.9, 0.0, 1.58],
    "handle_length_m": 0.12,
    "sample_angles_deg": [0.0, 45.0, 91.4],
    "arm_joint_targets_rad": [[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]],
    "peak_static_ratio": 0.4,
}


def test_opening_candidate_reverses_the_same_world_arc_without_mutation():
    original = deepcopy(CLOSING)

    opening = opening_candidate(CLOSING)

    assert CLOSING == original
    assert opening["start_deg"] == pytest.approx(180.0)
    assert opening["sense"] == 1
    assert opening["operation"] == "open"
    assert opening["gripper_tangent_sign"] == -1
    assert opening["grip_lead_s"] == pytest.approx(0.4)
    assert opening["release_open_before_clear"] is True
    assert opening["release_open_s"] == pytest.approx(0.5)
    assert opening["release_duration_s"] == pytest.approx(3.0)
    assert opening["world_closed_angle_deg"] == pytest.approx(180.0)
    assert opening["world_open_angle_deg"] == pytest.approx(270.0)
    np.testing.assert_allclose(opening["base_pose"], [0.58, 0.03, 1.1225, 0.61])
    np.testing.assert_allclose(opening["opening_base_offset_m"], [0.08, 0.0, 0.06])
    closing_open, _, _ = grasp_motion.handle_frame(CLOSING, 0.0)
    closing_closed, _, _ = grasp_motion.handle_frame(CLOSING, 90.0)
    opening_closed, _, _ = grasp_motion.handle_frame(opening, 0.0)
    opening_open, _, _ = grasp_motion.handle_frame(opening, 90.0)
    np.testing.assert_allclose(opening_closed, closing_closed, atol=1e-12)
    np.testing.assert_allclose(opening_open, closing_open, atol=1e-12)
    _, closing_closed_rotation = grasp_motion.grasp_site_target(CLOSING, 90.0)
    _, opening_closed_rotation = grasp_motion.grasp_site_target(opening, 0.0)
    _, closing_open_rotation = grasp_motion.grasp_site_target(CLOSING, 0.0)
    _, opening_open_rotation = grasp_motion.grasp_site_target(opening, 90.0)
    np.testing.assert_allclose(opening_closed_rotation, closing_closed_rotation, atol=1e-12)
    np.testing.assert_allclose(opening_open_rotation, closing_open_rotation, atol=1e-12)


def test_opening_candidate_keeps_only_an_interpolated_closed_pose_seed():
    opening = opening_candidate(CLOSING)

    assert "sample_angles_deg" not in opening
    assert "arm_joint_targets_rad" not in opening
    assert "peak_static_ratio" not in opening
    np.testing.assert_allclose(
        opening["initial_arm_target_rad"],
        [
            np.interp(90.0, CLOSING["sample_angles_deg"], column)
            for column in np.asarray(CLOSING["arm_joint_targets_rad"]).T
        ],
    )


def test_opening_release_opens_fingers_before_the_arm_moves_clear():
    opening = opening_candidate(CLOSING)
    boundaries = grasp_motion._boundaries(90.0, 9.0, grasp_motion.DEFAULT_DURATIONS_S)
    boundaries["release_open_before_clear"] = True
    boundaries["release_force_end"] = boundaries["hold"] + 1.0
    boundaries["release_clear_start"] = boundaries["release_force_end"]
    boundaries["release_clear_end"] = boundaries["release"] - 0.5
    command = grasp_motion.GraspArmTargetCommand(opening, boundaries, 90.0, 9.0)
    halfway_open = 0.5 * (boundaries["hold"] + boundaries["release_force_end"])
    halfway_clear = 0.5 * (
        boundaries["release_clear_start"] + boundaries["release_clear_end"]
    )

    grasp_position, _ = grasp_motion.grasp_site_target(opening, 90.0)
    early_position, _ = command._target(halfway_open)
    later_position, _ = command._target(halfway_clear)
    finger_force = grasp_motion.finger_command(
        halfway_open,
        grasp_motion.CLOSED_FINGER_M,
        0.0,
        boundaries=boundaries,
        force_limit_n=100.0,
    )

    np.testing.assert_allclose(early_position, grasp_position)
    assert -98.0 < finger_force < 0.0
    assert later_position[1] > grasp_position[1]


@pytest.mark.parametrize("change", [{"sense": 0}, {"sample_angles_deg": [0.0, 80.0]}])
def test_opening_candidate_rejects_an_invalid_axis_or_missing_closed_seed(change):
    candidate = {**CLOSING, **change}
    if "sample_angles_deg" in change:
        candidate["arm_joint_targets_rad"] = [[0.0, 1.0], [0.5, 1.5]]
    with pytest.raises(ValueError):
        opening_candidate(candidate)
