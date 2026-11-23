import mujoco
import numpy as np
import pytest

from go2_wall import pose_search
from go2_wall.robot import LEGS, compile_robot
from go2_wall.statics import FLOOR_NORMAL, Contact, joint_names, solve_static, torque_limits


def standing():
    model, data = compile_robot(case="scaled", wall_x=1.5)
    data.qpos[:] = model.qpos0
    pose_search.set_base(model, data, 0.0, np.array([0.0, 0.0, 0.33]))
    radius = model.geom_size[model.geom("FL").id][0]
    targets = {
        leg: np.array(
            [
                pose_search.HIP_X if leg[0] == "F" else -pose_search.HIP_X,
                pose_search.FOOT_Y if leg[1] == "L" else -pose_search.FOOT_Y,
                radius,
            ]
        )
        for leg in LEGS
    }
    assert pose_search.solve_leg_ik(model, data, targets) < 1e-3
    return model, data


def test_contact_forces_carry_the_whole_weight():
    model, data = standing()
    contacts = [Contact(leg, FLOOR_NORMAL, 0.6) for leg in LEGS]
    solution = solve_static(model, data, contacts, torque_limits())
    assert solution is not None
    weight = model.body_subtreemass[1] * 9.81
    assert sum(force[2] for force in solution.forces.values()) == pytest.approx(weight, rel=1e-6)
    assert solution.residual < 1e-6


def test_solution_satisfies_joint_equilibrium():
    model, data = standing()
    contacts = [Contact(leg, FLOOR_NORMAL, 0.6) for leg in LEGS]
    solution = solve_static(model, data, contacts, torque_limits())
    generalized = data.qfrc_bias.copy()
    for contact in contacts:
        geom = model.geom(contact.leg).id
        point = data.geom_xpos[geom] - model.geom_size[geom][0] * contact.normal
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, point, model.geom_bodyid[geom])
        generalized -= jacobian.T @ solution.forces[contact.leg]
    dofs = [model.joint(name).dofadr[0] for name in joint_names()]
    assert np.abs(generalized[dofs] - solution.torque).max() < 1e-9
    assert np.abs(generalized[:6]).max() < 1e-6


def test_friction_pyramid_is_respected():
    model, data = standing()
    contacts = [Contact(leg, FLOOR_NORMAL, 0.3) for leg in LEGS]
    solution = solve_static(model, data, contacts, torque_limits())
    for force in solution.forces.values():
        assert abs(force[0]) + abs(force[1]) <= 0.3 * force[2] + 1e-9


def test_reported_ratio_is_the_largest_joint_ratio():
    model, data = standing()
    contacts = [Contact(leg, FLOOR_NORMAL, 0.6) for leg in LEGS]
    limits = torque_limits()
    solution = solve_static(model, data, contacts, limits)
    assert np.max(np.abs(solution.torque) / limits) == pytest.approx(solution.ratio, abs=1e-6)
