import json

import numpy as np

from go2_wall.hold import max_static_push
from go2_wall.pose_search import braced_contacts
from go2_wall.robot import ROOT, compile_robot
from go2_wall.statics import max_tip_force, torque_limits


def test_lp_matches_the_bisection_push_limit():
    pose = json.loads((ROOT / "evidence/d1-front-mount/matrix.json").read_text())["reference_pose"]
    bisection = max_static_push(pose, "scaled", 0.6, 0.6, "derived")["max_push_n"]
    model, data = compile_robot(wall_x=pose["wall_x_m"])
    data.qpos[:] = pose["qpos"]
    import mujoco

    mujoco.mj_forward(model, data)
    site = model.site("arm_tip").id
    lp = max_tip_force(
        model,
        data,
        braced_contacts(0.6, 0.6),
        torque_limits("derived"),
        np.array([1.0, 0.0, 0.0]),
        data.site_xpos[site].copy(),
        model.site_bodyid[site],
    )
    assert abs(lp - bisection) < 0.05
