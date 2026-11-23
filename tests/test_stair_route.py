from pathlib import Path

import mujoco
import numpy as np
import pytest

from go2_wall import stairs, transition
from go2_wall.pose_search import set_base


@pytest.mark.parametrize("ascending", [True, False])
def test_swing_clears_the_second_tread_nose(ascending):
    slope, rise, radius = 50.0, 0.19, 0.03
    depth = stairs.stair_parameters()["tread_depth_m"]
    lower = (stairs.tread_centre(1, slope, rise, depth) + [0, 0, radius], "floor")
    upper = (stairs.tread_centre(2, slope, rise, depth) + [0, 0, radius], "floor")
    start, end = (lower, upper) if ascending else (upper, lower)
    nose_x = stairs.going_of(slope, rise)
    second_tread_z = 2 * rise

    def bottom_near_nose(curve):
        points = np.array([curve(start, end, u) for u in np.linspace(0, 1, 101)])
        np.testing.assert_array_equal(points[[0, -1]], [start[0], end[0]])
        near = np.abs(points[:, 0] - nose_x) <= radius
        assert np.any(near)
        return points[near, 2] - radius

    assert np.any(bottom_near_nose(transition.swing_point) < second_tread_z)
    assert np.all(bottom_near_nose(stairs.swing_over_tread) >= second_tread_z + 0.049)


def test_base_shift_preserves_pitch_while_turning_the_heading():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    start = (2.0, 0.0, 2.6, np.pi / 6)
    midpoint = transition.interpolate_base(start, (*start, np.pi), 0.5)
    set_base(model, data, midpoint[3], np.array(midpoint[:3]), yaw=midpoint[4])
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, data.qpos[3:7])
    np.testing.assert_allclose(rot.reshape(3, 3)[:, 0], [0, np.sqrt(3) / 2, 0.5], atol=1e-12)


def test_base_shift_adds_roll_about_the_body_forward_axis():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    start = (2.0, 0.0, 2.6, np.pi / 6, np.pi / 2)
    goal = (*start, np.pi / 3)
    midpoint = transition.interpolate_base(start, goal, 0.5)
    assert midpoint == (*start, np.pi / 6)
    assert transition.interpolate_base(goal, start, 1.0) == (*start, 0.0)

    set_base(model, data, start[3], np.array(start[:3]), yaw=start[4])
    unrolled = np.empty(9)
    mujoco.mju_quat2Mat(unrolled, data.qpos[3:7])
    set_base(model, data, midpoint[3], np.array(midpoint[:3]), yaw=midpoint[4], roll=midpoint[5])
    rolled = np.empty(9)
    mujoco.mju_quat2Mat(rolled, data.qpos[3:7])
    original = unrolled.reshape(3, 3)
    np.testing.assert_allclose(rolled.reshape(3, 3)[:, 0], original[:, 0], atol=1e-12)
    expected_side = np.sqrt(3) / 2 * original[:, 1] + 0.5 * original[:, 2]
    np.testing.assert_allclose(rolled.reshape(3, 3)[:, 1], expected_side, atol=1e-12)


def test_round_trip_keeps_floor_return_and_course_verdicts_distinct(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from replay_stair_sequence import round_trip_summary

    pose = np.zeros((100, 7))
    pose[:, 3] = 1.0
    loads = np.zeros((100, 4, 12))
    loads[:, :, 0] = 115.0
    log = {
        "time": np.arange(100) / 100,
        "qpos": pose,
        "qvel": np.zeros((100, 6)),
        "extra": loads,
    }
    platform = {"treads": (11,) * 4, "all_expected_feet_ge_1n": True}
    result = {
        "start_support": {"treads": (0,) * 4},
        "controller_metrics": {"seconds": 1.0, "passed": True},
        "verified_sequence": True,
        "milestones": [
            {"support": platform, "base_heading_deg": -0.4, "motion_end_s": 0.25},
            {"support": platform, "base_heading_deg": 179.5, "motion_end_s": 0.5},
        ],
    }
    report = round_trip_summary(result, log)
    assert report["physical_round_trip_verified"]
    assert report["first_all_platform_s"] == 0.25
    assert report["first_turn_on_platform_s"] == 0.5
    assert report["stable_floor_return"]["sample_count"] == 50

    rejected = round_trip_summary({**result, "verified_sequence": False}, log)
    assert rejected["stable_floor_return"]["verified"]
    assert not rejected["physical_round_trip_verified"]
    log["qvel"][-1, 0] = 0.051
    assert not round_trip_summary(result, log)["stable_floor_return"]["verified"]


def test_swing_body_waypoints_hand_off_at_the_settled_pose(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from replay_stair_sequence import segments_and_milestones

    start = [0.0, 0.0, 0.5, 0.0]
    lifted = [0.02, 0.0, 0.5, 0.0]
    settled = [0.05, 0.0, 0.48, 0.05]
    initial = {
        "base": start,
        "feet_xyz_m": {
            "FL": [0.20, 0.15, 0.03],
            "FR": [0.20, -0.15, 0.03],
            "RL": [-0.20, 0.15, 0.03],
            "RR": [-0.20, -0.15, 0.03],
        },
    }
    moves = [
        {
            "leg": "RR",
            "target_xyz_m": [-0.10, -0.15, 0.03],
            "swing_s": 2.0,
            "path": {"start_base": start, "swing_base": lifted, "swing_bases": [lifted, settled]},
        },
        {
            "leg": "RL",
            "target_xyz_m": [-0.10, 0.15, 0.03],
            "swing_s": 1.2,
            "path": {"start_base": settled, "swing_base": start},
        },
    ]
    segments, _ = segments_and_milestones({"foot_radius_m": 0.03}, None, moves, initial=initial)
    _, shift, unload, swing, load, hold, next_shift, *_ = segments
    assert shift["to"] == unload["base"] == tuple(lifted)
    assert swing["swing_bases"] == [lifted, settled]
    assert load["base"] == hold["base"] == next_shift["from"] == tuple(settled)


@pytest.fixture
def route_evaluator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "train"))
    import eval_target_stairs

    return eval_target_stairs


def test_route_keeps_footprints_on_stair_and_landing(route_evaluator):
    landing = np.array([2.2, 0.0])
    half_size = np.array([0.6, 0.6])
    width = 0.305
    points = np.array([[-0.2, 0.2], [0.3, -0.2], [1.8, 0.5]])
    check = route_evaluator.inside_route
    assert check(points, width, landing, half_size)
    points[1, 1] = -0.31
    assert not check(points, width, landing, half_size)

    feet_on_top = np.array([[2.1, -0.3], [2.1, 0.3], [2.3, -0.3], [2.3, 0.3]])
    assert check(feet_on_top, width, landing, half_size, turning=True)
    feet_on_top[2, 0] = 1.5
    assert not check(feet_on_top, width, landing, half_size, turning=True)


def test_return_requires_four_loaded_feet_and_low_speed(route_evaluator):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="base" pos="-1.1 0 .5">'
        '<freejoint/><geom type="sphere" size=".01"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base = model.body("base").id
    feet = {1, 2, 3, 4}
    stable = route_evaluator.stable_return
    assert stable(data, base, feet, feet)
    assert not stable(data, base, {1, 3}, feet)
    data.qvel[0] = 0.08
    assert not stable(data, base, feet, feet)
    data.qvel[0] = 0.0
    data.qvel[5] = 0.2
    assert not stable(data, base, feet, feet)
