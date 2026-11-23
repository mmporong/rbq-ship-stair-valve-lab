from types import SimpleNamespace

import numpy as np
import pytest

from go2_wall.stair_open_balance import OpeningStanceControl
from go2_wall.statics import joint_names


def test_opening_feedback_uses_measured_angles_and_changes_only_leg_torques():
    names = joint_names()
    model = SimpleNamespace(
        joint=lambda name: SimpleNamespace(
            qposadr=[7 + names.index(name)], dofadr=[6 + names.index(name)]
        )
    )
    data = SimpleNamespace(qpos=np.zeros(25), qvel=np.zeros(24))
    torque = np.arange(18, dtype=float)
    controller = OpeningStanceControl(model, 2.0, 5.0)

    def command(now, stance=(True,) * 4):
        return controller.command(model, data, None, 0, 0, stance, None, now, torque)[0]

    assert controller.base_feedback_scales(1.9) == (1, 1)
    assert controller.base_feedback_scales(2.0) == (2.0, 2.0)
    assert command(4.9) is torque
    assert command(5.0, (False, True, True, True)) is torque
    data.qpos[7] = 0.5
    np.testing.assert_array_equal(command(5.0), torque)
    data.qpos[7] += 0.01
    data.qvel[6] = 0.1
    revised = command(5.1)
    assert revised[0] == pytest.approx(-1.3)
    assert data.qpos[7] == 0.51
    np.testing.assert_array_equal(revised[1:], torque[1:])
    np.testing.assert_array_equal(torque, np.arange(18))
    assert controller.describe()["peak_leg_correction_nm"] == pytest.approx(1.3)


def test_centered_work_target_preserves_the_installed_pose():
    pose = [0.58, 0.03, 1.12, 0.61]
    assert OpeningStanceControl.work_pose(pose) == pytest.approx([0.58, 0.014, 1.115, 0.61])
    assert pose == [0.58, 0.03, 1.12, 0.61]


def test_rear_retargeting_changes_only_the_scheduled_rear_landing(monkeypatch):
    monkeypatch.syspath_prepend("tools")
    from stair_open_approach import rear_landing_stance

    feet = {"RL": [0.382, 0.19, 0.60], "RR": [0.382, -0.19, 0.60]}
    stance = {
        "FL": ([0.701, 0.19, 0.98], "floor"),
        "RL": (feet["RL"], "floor"),
        "RR": ([0.223, -0.19, 0.41], "floor"),
    }
    shifted = rear_landing_stance(stance, feet, 0.04)
    np.testing.assert_allclose(shifted["RL"][0], [0.342, 0.19, 0.60])
    for leg in ("FL", "RR"):
        np.testing.assert_array_equal(shifted[leg][0], stance[leg][0])
    assert feet["RL"][0] == 0.382
