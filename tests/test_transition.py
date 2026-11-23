import numpy as np

from go2_wall import transition as tr
from go2_wall.robot import LEGS, compile_robot
from go2_wall.statics import solve_static, torque_limits


def test_physics_steps_include_rounded_end_but_never_run_a_partial_step():
    assert tr.physics_step_count(767.9999999999973, 0.005) == 153600
    assert tr.physics_step_count(767.999, 0.005) == 153599
    assert tr.physics_step_count(768, 0.0025) == 307200


def start_pose():
    model, data = compile_robot(wall_x=0.75)
    start, sequence = tr.stances(0.75, model.geom_size[model.geom("FL").id][0])
    base = (0.75 - tr.NOSE_M - tr.NOSE_GAP_M, 0.0, tr.STAND_HEIGHT_M, 0.0)
    feet = {leg: item[0] for leg, item in start.items()}
    assert tr.place(model, data, base, feet, tr.tuck_arm())
    return model, data, start, sequence


def test_rear_feet_step_first_and_the_last_stance_is_the_m1_pose():
    model, data, _, sequence = start_pose()
    assert [leg for leg, _ in sequence[:2]] == ["RL", "RR"]
    final = sequence[-1][1]
    data.qpos[:] = tr.reference_pose()["qpos"]
    import mujoco

    mujoco.mj_kinematics(model, data)
    for leg in LEGS:
        assert np.allclose(final[leg][0], data.geom_xpos[model.geom(leg).id], atol=1e-9)
    assert [final[leg][1] for leg in LEGS] == ["wall", "wall", "floor", "floor"]


def test_smooth_solution_spreads_load_and_keeps_the_lp_ratio():
    model, data, start, _ = start_pose()
    contacts = tr.contacts_for(start)
    lp = solve_static(model, data, contacts, torque_limits())
    qp = solve_static(model, data, contacts, torque_limits(), smooth=True)
    normals = [qp.forces[leg][2] for leg in LEGS]
    assert abs(qp.ratio - lp.ratio) < 1e-6
    assert min(normals) > 0.2 * max(normals)
    assert qp.residual < 1e-6


def test_normal_force_bounds_hold():
    model, data, start, _ = start_pose()
    contacts = tr.contacts_for(start)
    capped = solve_static(model, data, contacts, torque_limits(), max_normal={"FL": 5.0})
    assert capped.forces["FL"][2] <= 5.0 + 1e-6
    floored = solve_static(model, data, contacts, torque_limits(), min_normal={"FL": 60.0})
    assert floored.forces["FL"][2] >= 60.0 - 1e-6


def test_contact_point_does_not_slide_when_still():
    model, data, start, _ = start_pose()
    jacobian = np.zeros((3, model.nv))
    data.qvel[:] = 0
    slide = tr.contact_velocity(model, data, model.geom("FL").id, tr.FLOOR_NORMAL, jacobian)
    assert np.allclose(slide, 0.0)


def test_wall_preload_never_exceeds_a_cap():
    contacts = [tr.Contact("FL", tr.WALL_NORMAL, 0.4), tr.Contact("RL", tr.FLOOR_NORMAL, 0.4)]
    assert tr.wall_preload(contacts, {"FL": 3.0}, 20.0) == {"FL": 3.0}
    assert tr.wall_preload(contacts, None, 20.0) == {"FL": 20.0}
