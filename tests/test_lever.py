import mujoco
import numpy as np

from go2_wall import lever as lv


def test_untouched_lever_stays_put():
    model, data = lv.build_scene()
    lv.braced_config(model, data)
    pose = data.qpos[: model.nq - 1].copy()
    hinge = model.joint("lever_hinge").qposadr[0]
    for _ in range(1000):
        data.qpos[: model.nq - 1] = pose
        data.qvel[: model.nv - 1] = 0
        mujoco.mj_step(model, data)
    assert abs(np.degrees(data.qpos[hinge])) < 0.1


def test_hinge_friction_matches_the_tip_force():
    lever = lv.lever_parameters()
    model, _ = lv.build_scene(lever=lever)
    dof = model.joint("lever_hinge").dofadr[0]
    expected = lever["tip_force_n"] * lever["handle_length_m"]
    assert abs(model.dof_frictionloss[dof] - expected) < 1e-12
