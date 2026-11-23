import multiprocessing

import mujoco
import numpy as np

from go2_wall import stairs as st
from go2_wall.robot import RBQ_FOOT_Y, compile_robot
from go2_wall.statics import joint_names, torque_limits


def test_rbq_rb1_model_follows_the_planner_conventions():
    model, data = compile_robot(arm="rb1")
    motors = [model.actuator(i) for i in range(model.nu)]
    assert [model.joint(m.trnid[0]).name for m in motors] == joint_names()
    assert np.allclose([m.ctrlrange[1] for m in motors], torque_limits("rb1"))
    arm = [b for b in range(model.nbody) if model.body(b).name.startswith("RB1_")]
    assert abs(sum(model.body_mass[b] for b in arm) - 7.993) < 0.01
    mujoco.mj_forward(model, data)
    assert abs(data.geom_xpos[model.geom("FL").id][1] - RBQ_FOOT_Y) < 1e-6


def test_stair_helpers_switch_body_geometry():
    try:
        st.ARM = "rb1"
        st.use_stow()
        assert st.GRID["dz"][0] == 0.24
        assert abs(st.hip_span() - 0.62436) < 1e-6
        feet = st.stance_on(2, 5, 50.0, 0.2, 0.127, 0.03)
        assert abs(feet["FL"][0][1] - RBQ_FOOT_Y) < 1e-9
    finally:
        st.ARM = "d1"
        st.use_stow()
    assert st.GRID["dz"][0] == 0.15


def stair_worker_parameters():
    model, _ = st._STATE["pair"]
    return (
        st.ARM,
        model.body_subtreemass[model.body("base").id],
        st.GRID["dz"][0],
        st.tr.tuck_arm(),
        st._STATE["limits"],
    )


def test_stair_planner_preserves_rbq_configuration_in_spawned_workers():
    with multiprocessing.get_context("spawn").Pool(
        1, initializer=st._init, initargs=(50.0, 0.19, "rb1")
    ) as pool:
        arm, mass, min_height, stow, limits = pool.apply(stair_worker_parameters)
    assert arm == "rb1"
    assert 48.9 < mass < 49.0
    assert min_height == 0.24
    assert {name: stow[name] for name in st.STOW_RB1} == st.STOW_RB1
    np.testing.assert_array_equal(limits, torque_limits("rb1"))
