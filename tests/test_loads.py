import mujoco
import numpy as np

from go2_wall import loads
from go2_wall.robot import build_arm_only


def test_mount_reaction_equals_arm_weight_at_rest():
    model, data = build_arm_only("scaled")
    data.qpos[:] = [0.3, 1.2, -1.0, 0.2, 0.4, 0.1][: model.nq]
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0
    mujoco.mj_rnePostConstraint(model, data)
    mount = model.body("arm_mount").id
    root = model.body_rootid[mount]
    force, moment = loads.moment_about(data, mount, data.xpos[mount], root)
    mass = model.body_subtreemass[mount]
    gravity = np.array([0, 0, -9.81 * mass])
    assert np.allclose(force, -gravity, atol=1e-6)
    lever_arm = data.subtree_com[mount] - data.xpos[mount]
    assert np.allclose(moment, -np.cross(lever_arm, gravity), atol=1e-6)


def test_bracket_stress_matches_hand_calculation():
    out = loads.bracket_stress(np.array([0, 0, -23.25]), np.zeros(3), np.eye(3), 0.15)
    assert abs(out["bending_vertical_mpa"] - 23.25 * 0.15 / (0.08 * 0.006**2 / 6) / 1e6) < 1e-9
