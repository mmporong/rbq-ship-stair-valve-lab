import mujoco
import numpy as np
import pytest

from go2_wall import stair_valve, transition
from go2_wall.gripper_control import FingerDrive, measured_arm_target
from go2_wall.robot import LEGS
from go2_wall.statics import joint_names, torque_limits


@pytest.fixture
def finger():
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody><body><joint name="slide" type="slide"/>
        <geom type="sphere" size=".01"/></body></worldbody>
        <actuator><motor name="hand" joint="slide" ctrllimited="true" ctrlrange="-10 10"/>
        </actuator></mujoco>""")
    return model, mujoco.MjData(model)


def test_records_finger_force_clipping_before_integration(finger):
    model, data = finger
    drive = FingerDrive(model, data, "hand", lambda time_s, pos, vel: 15.0)
    assert drive.apply(data, 0.0, model.opt.timestep) == 15.0
    assert data.ctrl[0] == 10.0
    assert drive.audit()["finger_clipped_steps"] == 1
    with pytest.raises(ValueError, match="clock discontinuity"):
        drive.apply(data, 0.0, model.opt.timestep)


def test_observer_cannot_set_live_slide_position(finger):
    model, data = finger
    drive = FingerDrive(model, data, "hand", lambda time_s, pos, vel: 0.0)
    before = drive.snapshot(data)
    data.qpos[0] = 0.02
    with pytest.raises(ValueError, match="live physics field qpos"):
        drive.compare(data, before)
    assert drive.audit()["qpos_write_count_after_start"] == 1


def test_external_force_is_visible_to_mission_verdict(finger):
    model, data = finger
    drive = FingerDrive(model, data, "hand", lambda time_s, pos, vel: 1.0)
    data.qfrc_applied[:] = -4
    drive.apply(data, 0.0, model.opt.timestep)
    mujoco.mj_step(model, data)
    drive.apply(data, model.opt.timestep, model.opt.timestep)
    assert np.isclose(drive.audit()["max_qfrc_applied_n"], 4.0)
    assert drive.audit()["physics_steps"] == 2


def test_held_finger_request_preserves_physics_audit(finger):
    model, data = finger
    calls = []

    def command(t, pos, vel):
        calls.append(t)
        return 12.0 if len(calls) == 1 else 4.0

    drive = FingerDrive(model, data, "hand", command)
    dt = model.opt.timestep
    applied = []
    for step in range(4):
        drive.apply(data, step * dt, dt, update_command=step % 2 == 0)
        applied.append(data.ctrl[0])
        mujoco.mj_step(model, data)
    assert calls == [0.0, 2 * dt]
    assert applied == [10.0, 10.0, 4.0, 4.0]
    assert drive.audit()["physics_steps"] == 4
    assert drive.audit()["finger_clipped_steps"] == 2


@pytest.mark.parametrize(("physics_dt", "stride"), [(0.005, 1), (0.0025, 2)])
def test_all_motor_commands_hold_between_five_ms_ticks(physics_dt, stride):
    model, data = stair_valve.build_grasp_scene(50.0, 0.19, [0.9, 0.0, 1.58], 270, -1, 0.12, 10.0)
    mujoco.mj_forward(model, data)
    addresses = [model.joint(name).qposadr[0] for name in joint_names()]
    track = {
        "time": np.array([0.0, 0.02]),
        "qpos": np.tile(data.qpos, (2, 1)),
        "target": np.tile(data.qpos[addresses], (2, 1)),
        "feet": np.tile([data.geom(leg).xpos.copy() for leg in LEGS], (2, 1, 1)),
        "torque": np.zeros((2, 18)),
        "wall": np.zeros((2, 4), dtype=bool),
        "stance": np.ones((2, 4), dtype=bool),
    }
    calls = []

    def arm(t, model, data, nominal):
        calls.append(t)
        return nominal + 0.01 + t

    metrics, log = transition.execute(
        track,
        None,
        physics={"timestep": physics_dt},
        builder=lambda: (model, data),
        mode="joint",
        limits_nm=torque_limits("rb1"),
        log_every_step=True,
        actuator_command=lambda t, pos, vel: 1.0 + 10.0 * t,
        arm_target_command=arm,
        actuation_period_s=0.005,
    )
    assert calls == pytest.approx([0.0, 0.005, 0.01, 0.015])
    assert metrics["gripper_audit"]["physics_steps"] == 4 * stride
    assert log["ctrl"].shape == (4 * stride, 19)
    assert np.any(np.diff(log["ctrl"][::stride, :18], axis=0))
    assert np.all(np.diff(log["ctrl"][::stride, -1]) > 0)
    for field in (
        "ctrl",
        "robot_torque_requested_nm",
        "requested_finger_force_n",
        "arm_target_rad",
    ):
        values = log[field]
        np.testing.assert_array_equal(values, np.repeat(values[::stride], stride, axis=0))


def test_arm_feedback_returns_motor_targets_without_setting_live_joints(finger):
    model, data = finger
    drive = FingerDrive(model, data, "hand", lambda *args: 0.0)
    nominal = np.zeros(6)
    target = measured_arm_target(lambda t, m, d, q: q + 0.1, 0.0, model, data, nominal, drive)
    np.testing.assert_array_equal(nominal, np.zeros(6))
    np.testing.assert_allclose(target, np.full(6, 0.1))
    with pytest.raises(ValueError, match="six finite joint targets"):
        measured_arm_target(lambda *args: [np.nan] * 6, 0.0, model, data, nominal, drive)

    def mutating_command(t, m, d, q):
        d.qpos[0] = 0.01
        return q

    with pytest.raises(ValueError, match="live physics field qpos"):
        measured_arm_target(mutating_command, 0.0, model, data, nominal, drive)


def test_stance_xy_spring_uses_physical_foot_centre_and_motor_jacobian():
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody><body><joint type="slide" axis="1 0 0"/>
        <joint type="slide" axis="0 1 0"/><geom name="foot" type="sphere" size=".01"/>
        </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    data.qpos[:] = [0.01, -0.02]
    mujoco.mj_forward(model, data)
    spring = transition.StanceXYSpring(1000.0, 1.0, 2.0)
    dofs = np.array([0, 1])
    geom = model.geom("foot").id

    np.testing.assert_array_equal(spring.correction(model, data, geom, dofs, 0.5), [0, 0])
    np.testing.assert_array_equal(spring.correction(model, data, geom, dofs, 1.0), [0, 0])
    data.qpos[:] = [0.013, -0.016]
    mujoco.mj_forward(model, data)

    np.testing.assert_allclose(spring.correction(model, data, geom, dofs, 1.5), [-3, -4])
    assert spring.details()["anchor_count"] == 1
    assert spring.details()["maximum_xy_error_m"] == pytest.approx(0.005)


@pytest.mark.parametrize(
    ("gain", "start", "end"),
    [(0.0, 1.0, 2.0), (1.0, -1.0, 2.0), (1.0, 2.0, 1.0), (np.nan, 1.0, 2.0)],
)
def test_stance_xy_spring_rejects_invalid_settings(gain, start, end):
    with pytest.raises(ValueError):
        transition.StanceXYSpring(gain, start, end)
