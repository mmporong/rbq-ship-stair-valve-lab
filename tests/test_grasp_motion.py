import mujoco
import numpy as np
import pytest

from go2_wall import grasp_motion, stair_valve

SAVED = {
    "centre_m": [0.9, 0.0, 1.58],
    "start_deg": 270.0,
    "sense": -1,
    "handle_length_m": 0.12,
}


def test_rated_tangent_load_relief_preserves_turn_and_releases_after_hold():
    legacy = {"operation_name": "operate"}
    schedule = {**legacy, "unload_start": 8.0, "unload_end": 9.0, "hold_load_fraction": 0.2}
    for stage, at, expected in (
        ("contact", 2.0, 0.0),
        ("operate", 7.0, 1.0),
        ("settle", 8.0, 1.0),
        ("settle", 8.5, 0.6),
        ("hold", 10.0, 0.2),
        ("release", 15.0, 0.0),
    ):
        assert grasp_motion.turn_load_fraction(at, stage, schedule) == pytest.approx(expected)
    assert grasp_motion.turn_load_fraction(10.0, "hold", legacy) == 1.0


def test_rated_terminal_overtravel_unwinds_before_five_second_hold(monkeypatch):
    from go2_wall import stairs

    monkeypatch.syspath_prepend("tools")
    from stair_mission_route import route

    geometry, _, study, _, _, _ = route(50.0, 0.19)
    monkeypatch.setattr(stairs, "ARM", "rb1")
    model, data = stairs.build_stairs(50.0, 0.19)
    track, plan = grasp_motion.make_grasp_track(
        model,
        data,
        {**study, "arm_stow_rad": geometry["arm_stow_rad"]},
        study["candidates"][4],
        target_deg=90.0,
        motion_s=9.0,
        resistance_nm=10.0,
        jaw_load_bias_m=0.0005,
        terminal_overtravel_deg=1.2,
    )
    assert plan["target_deg"] == 90.0
    assert plan["command_target_deg"] == pytest.approx(92.6)
    assert plan["hold_target_deg"] == pytest.approx(91.4)
    borders = plan["boundaries_s"]
    assert borders["hold"] - borders["settle"] == pytest.approx(5.0)
    held = np.flatnonzero(track["stage"] == "hold")
    assert held.size > 1
    np.testing.assert_allclose(track["target"][held[-1], 12:], track["target"][held[0], 12:])
    assert np.max(abs(track["target"][held[0], 12:] - track["target"][held[0] - 1, 12:])) < 0.001


def test_grasp_frame_centres_jaws_and_tracks_valve_tangent():
    point, tangent, radial = grasp_motion.handle_frame(SAVED, 20.0)
    tip, rotation = grasp_motion.grasp_site_target(SAVED, 20.0)

    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12)
    assert np.allclose(rotation[:, 0], tangent)
    assert np.allclose(
        tip - grasp_motion.PAD_CENTRE_BEHIND_TIP_M * rotation[:, 2],
        point,
    )
    assert np.isclose(np.linalg.norm(point - SAVED["centre_m"]), SAVED["handle_length_m"])
    assert np.isclose(tangent @ radial, 0.0)


def test_jaw_load_bias_is_explicit_and_does_not_change_default_target():
    default_tip, default_rotation = grasp_motion.grasp_site_target(SAVED, 20.0)
    biased_tip, biased_rotation = grasp_motion.grasp_site_target(SAVED, 20.0, 0.0005)
    repeated_default_tip, _ = grasp_motion.grasp_site_target(SAVED, 20.0)

    np.testing.assert_allclose(biased_tip - default_tip, 0.0005 * default_rotation[:, 0])
    np.testing.assert_array_equal(biased_rotation, default_rotation)
    np.testing.assert_array_equal(repeated_default_tip, default_tip)


def test_sample_times_preserves_grid_runs_and_appends_exact_non_grid_endpoint():
    grid_duration = 29.0
    expected_grid = np.arange(int(round(grid_duration / 0.02)) + 1) * 0.02

    np.testing.assert_array_equal(grasp_motion._sample_times(grid_duration), expected_grid)
    non_grid = grasp_motion._sample_times(29.25)
    assert non_grid[-2] == pytest.approx(29.24)
    assert non_grid[-1] == 29.25


def test_finger_profile_closes_during_contact_and_releases_smoothly():
    durations = grasp_motion.DEFAULT_DURATIONS_S
    boundaries = grasp_motion._boundaries(20.0, 4.0, durations)

    before_grip = grasp_motion.finger_command(
        boundaries["grip_start"] - 0.1,
        grasp_motion.OPEN_FINGER_M,
        0.0,
        boundaries=boundaries,
        force_limit_n=100.0,
    )
    after_grip = grasp_motion.finger_command(
        boundaries["contact"],
        grasp_motion.OPEN_FINGER_M,
        0.0,
        boundaries=boundaries,
        force_limit_n=100.0,
    )
    release_start = grasp_motion.finger_command(
        boundaries["hold"],
        grasp_motion.CLOSED_FINGER_M,
        0.0,
        boundaries=boundaries,
        force_limit_n=100.0,
    )
    release_end = grasp_motion.finger_command(
        0.5 * (boundaries["release_clear_end"] + boundaries["release"]),
        grasp_motion.CLOSED_FINGER_M,
        0.0,
        boundaries=boundaries,
        force_limit_n=100.0,
    )

    assert before_grip == 0.0
    assert -100.0 < after_grip < 0.0
    assert release_start == -98.0
    assert release_end == pytest.approx(100.0 * np.tanh(1.0))


def test_arm_feedback_is_read_only_and_slew_limited():
    model, data = stair_valve.build_grasp_scene(
        50.0, 0.19, SAVED["centre_m"], SAVED["start_deg"], SAVED["sense"], 0.12, 1.0
    )
    mujoco.mj_forward(model, data)
    boundaries = grasp_motion._boundaries(20.0, 4.0, grasp_motion.DEFAULT_DURATIONS_S)
    command = grasp_motion.make_arm_target_command(SAVED, boundaries, 20.0, 4.0)
    nominal = np.zeros(6)
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    ctrl = data.ctrl.copy()

    applied = command(boundaries["approach"] + 0.1, model, data, nominal)

    assert applied.shape == (6,)
    assert np.max(np.abs(applied - nominal)) <= grasp_motion.ARM_TARGET_STEP_LIMIT_RAD
    assert np.array_equal(data.qpos, qpos)
    assert np.array_equal(data.qvel, qvel)
    assert np.array_equal(data.ctrl, ctrl)
    assert command.details()["samples"] == 1


def test_open_hand_control_never_requests_force_outside_actuator_limit():
    for position in np.linspace(0.0, grasp_motion.OPEN_FINGER_M, 21):
        requested = grasp_motion.open_finger_command(position, 0.0, force_limit_n=80.0)
        assert 0.0 <= requested < 80.0
