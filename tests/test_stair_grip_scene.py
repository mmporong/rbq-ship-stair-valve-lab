import mujoco
import numpy as np
import pytest

from go2_wall import stair_valve as sv
from go2_wall.statics import joint_names


@pytest.fixture(scope="module")
def grasp_scene():
    return sv.build_grasp_scene(
        50.0,
        0.19,
        [0.8, 0.0, 1.5],
        30.0,
        1,
        0.12,
        10.0,
        floor_mu=0.5,
        pad_mu=0.7,
        finger_force_n=80.0,
        slide_loss_n=15.0,
    )


def test_grasp_scene_appends_vendor_finger_actuator_without_pin(grasp_scene):
    model, _ = grasp_scene
    finger = model.actuator(sv.GRIPPER_ACTUATOR)
    upper = model.joint(sv.GRIPPER_JOINTS[0])

    assert model.nu == len(joint_names()) + 1
    assert [model.actuator(i).name for i in range(18)] == [
        name.replace("_joint", "_motor") for name in joint_names()
    ]
    assert finger.id == 18
    assert model.actuator_trnid[finger.id, 0] == upper.id
    assert np.allclose(model.actuator_ctrlrange[finger.id], [-80.0, 80.0])
    assert model.neq == 1
    assert model.eq_type[0] == mujoco.mjtEq.mjEQ_JOINT
    assert not any(model.equality(i).name == sv.VALVE_PIN for i in range(model.neq))

    for joint_name in sv.GRIPPER_JOINTS:
        joint = model.joint(joint_name)
        assert model.dof_frictionloss[joint.dofadr[0]] == pytest.approx(15.0)
    for geom_name in sv.GRIPPER_PADS + (sv.VALVE_LEVER,):
        geom = model.geom(geom_name)
        assert model.geom_contype[geom.id] == 1
        assert model.geom_conaffinity[geom.id] == 1
        assert model.geom_condim[geom.id] == 4
        assert model.geom_friction[geom.id, 0] == pytest.approx(0.7)


def _contacting_scene(sense):
    # At the vendor zero pose the midpoint of the pad boxes is (0.238, 0, 1.6627).
    # A vertical lever ending there intersects both pads and the wrist collision cylinder.
    model, data = sv.build_grasp_scene(50.0, 0.19, [0.238, 0.0, 1.5427], 90.0, sense, 0.12, 10.0)
    mujoco.mj_forward(model, data)
    return model, data


def test_grasp_observation_uses_contact_force_and_closing_direction():
    positive_model, positive_data = _contacting_scene(1)
    negative_model, negative_data = _contacting_scene(-1)

    positive = sv.observe_grasp(positive_model, positive_data)
    negative = sv.observe_grasp(negative_model, negative_data)

    assert tuple(positive) == sv.GRASP_OBSERVATION_FIELDS + ("unexpected_lever_contacts_n",)
    assert positive["pad_normal_force_n"].shape == (2,)
    assert positive["pad_surface_gap_m"].shape == (2,)
    assert np.all(positive["pad_surface_gap_m"] < 0.0)
    assert positive["pad_normal_force_n"].sum() > 0.0
    assert positive["pad_closing_torque_nm"] == pytest.approx(-negative["pad_closing_torque_nm"])
    assert positive["pad_closing_tangent_force_n"] == pytest.approx(
        -negative["pad_closing_tangent_force_n"]
    )
    assert positive["handle_axis_offset_m"] == pytest.approx(0.0, abs=1e-12)
    assert positive["handle_axial_position_m"] == pytest.approx(0.12, abs=1e-12)
    assert positive["max_unexpected_lever_contact_n"] > 0.0
    assert any("RB1_Wrist_Yaw" in name for name in positive["unexpected_lever_contacts_n"])


def test_grasp_observation_reports_only_closing_resistance_as_positive():
    model, data = _contacting_scene(1)
    hinge = model.joint(sv.VALVE_HINGE)
    data.qvel[hinge.dofadr[0]] = 1.0
    mujoco.mj_forward(model, data)

    observation = sv.observe_grasp(model, data)

    assert observation["valve_speed_rad_s"] == pytest.approx(1.0)
    assert observation["hinge_friction_torque_nm"] == pytest.approx(10.0, abs=1e-8)
    assert observation["signed_hinge_friction_torque_nm"] == pytest.approx(-10.0, abs=1e-8)

    data.qvel[hinge.dofadr[0]] = -1.0
    mujoco.mj_forward(model, data)
    observation = sv.observe_grasp(model, data)
    assert observation["hinge_friction_torque_nm"] == pytest.approx(-10.0, abs=1e-8)
    assert observation["signed_hinge_friction_torque_nm"] == pytest.approx(10.0, abs=1e-8)


def test_pad_torque_includes_contact_wrench_moment():
    model, data = _contacting_scene(1)
    hinge = model.joint(sv.VALVE_HINGE)
    # Align this test joint with the contact-normal torque and spin the free base so the
    # condim=4 contacts generate a nonzero rotational wrench.
    model.jnt_axis[hinge.id] = [1.0, 0.0, 0.0]
    data.qvel[3] = 10.0
    mujoco.mj_forward(model, data)
    lever = model.geom(sv.VALVE_LEVER).id
    pads = {model.geom(name).id for name in sv.GRIPPER_PADS}
    axis, anchor = data.xaxis[hinge.id], data.xanchor[hinge.id]
    force_only = total = 0.0
    wrench = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        if lever not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom1 if contact.geom2 == lever else contact.geom2
        if other not in pads:
            continue
        mujoco.mj_contactForce(model, data, index, wrench)
        frame = contact.frame.reshape(3, 3)
        sign = 1.0 if contact.geom2 == lever else -1.0
        force = sign * (frame.T @ wrench[:3])
        contact_torque = sign * (frame.T @ wrench[3:])
        force_moment = np.cross(contact.pos - anchor, force)
        force_only += float(force_moment @ axis)
        total += float((force_moment + contact_torque) @ axis)

    assert abs(total - force_only) > 1e-6
    assert sv.observe_grasp(model, data)["pad_closing_torque_nm"] == pytest.approx(total)


@pytest.mark.parametrize(
    "change",
    [
        {"floor_mu": 0.0},
        {"pad_mu": 0.0},
        {"finger_force_n": 0.0},
        {"slide_loss_n": -1.0},
    ],
)
def test_grasp_scene_rejects_nonphysical_contact_inputs(change):
    with pytest.raises(ValueError):
        sv.build_grasp_scene(50.0, 0.19, [0.8, 0.0, 1.5], 0.0, 1, 0.12, 10.0, **change)
