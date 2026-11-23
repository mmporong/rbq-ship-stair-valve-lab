import numpy as np

from go2_wall import transition
from go2_wall.robot import LEGS, compile_robot
from go2_wall.stair_walking_control import distributed_forces, friction_constraints
from go2_wall.statics import joint_names


def test_force_distribution_obeys_normal_caps_and_horizontal_friction():
    # One newton per lateral component consumes the same pyramid budget.
    balance = np.zeros((6, 9))
    for k, (x, y) in enumerate(((1, 0), (-0.5, 1), (-0.5, -1))):
        block = slice(k * 3, k * 3 + 3)
        balance[:3, block] = np.eye(3)
        balance[3:, block] = [[0, 0, y], [0, 0, -x], [-y, x, 0]]
    constraints = friction_constraints(3, 0.42)
    bounds = np.tile([0.0, 0, 0, 0, -1, 15], 3)
    target = np.array([2, 3, 30, 0, 0, 0])
    ref = np.tile([0.0, 0, 10], 3)
    forces = distributed_forces(balance, target, ref, constraints, bounds)
    np.testing.assert_allclose(balance @ forces, target, atol=1e-8)
    assert np.all(constraints @ forces <= bounds + 1e-8)
    excessive = np.array([17, 0, 30, 0, 0, 0])
    assert distributed_forces(balance, excessive, ref, constraints, bounds) is None


def test_walking_solver_status_is_held_between_servo_commands():
    model, data = compile_robot(wall_x=0.75)
    pose = data.qpos.copy()
    cols = [model.joint(j).qposadr[0] for j in joint_names()]
    feet = np.array([data.geom(leg).xpos for leg in LEGS])
    track = {
        "time": np.array([0.0, 0.01]),
        "qpos": np.tile(pose, (2, 1)),
        "target": np.tile(pose[cols], (2, 1)),
        "torque": np.zeros((2, 18)),
        "feet": np.tile(feet, (2, 1, 1)),
        "stance": np.ones((2, 4), dtype=bool),
        "wall": np.zeros((2, 4), dtype=bool),
        "cap": np.full((2, 4), np.inf),
    }

    class RecordedFeedback:
        def command(self, model, data, track, row, weight, stance, nominal, now, original):
            return original, 1 if now < 0.005 else -1

        def describe(self):
            return {}

    metrics, raw = transition.execute(
        track,
        None,
        builder=lambda: (model, data),
        physics={"timestep": 0.001},
        mode="joint",
        actuator_command=None,
        walking_control=RecordedFeedback(),
        actuation_period_s=0.005,
        log_every_step=True,
    )
    assert metrics["qp_relaxed_ticks"] == metrics["qp_failed_ticks"] == 1
    np.testing.assert_array_equal(raw["attempt"], [1] * 5 + [-1] * 5)
