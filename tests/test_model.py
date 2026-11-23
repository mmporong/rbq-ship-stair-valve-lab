import mujoco
import pytest

from go2_wall import d1_estimate
from go2_wall.robot import (
    bracket_plate,
    build_arm_only,
    compile_robot,
    load_parameters,
    read_urdf,
    resolve_mount,
)


def arm_mass(model: mujoco.MjModel) -> float:
    return sum(
        model.body_mass[index]
        for index in range(model.nbody)
        if model.body(index).name.startswith("arm_")
    )


@pytest.mark.parametrize("case", ["scaled", "servo_lumped"])
def test_arm_mass_matches_official_value(case):
    model, _ = build_arm_only(case)
    assert arm_mass(model) == pytest.approx(2.37, abs=1e-9)


@pytest.mark.parametrize("case", ["scaled", "servo_lumped"])
def test_combined_model_adds_only_the_arm(case):
    model, _ = compile_robot(case=case)
    assert model.nu == 18
    assert model.njnt == 1 + 12 + 6
    assert arm_mass(model) == pytest.approx(2.37, abs=1e-9)


def test_front_bracket_adds_only_the_plate_mass():
    stock, _ = compile_robot(mount_x=0.0)
    braced, _ = compile_robot()
    plate = bracket_plate(resolve_mount(None))
    assert plate["mass_kg"] == pytest.approx(2700 * 0.225 * 0.08 * 0.006, rel=1e-12)
    assert braced.body("bracket").mass[0] == pytest.approx(plate["mass_kg"], rel=1e-9)
    assert braced.body_subtreemass[1] == pytest.approx(
        stock.body_subtreemass[1] + plate["mass_kg"], rel=1e-9
    )


def test_front_bracket_moves_the_arm_forward_and_up_by_the_plate():
    stock, _ = compile_robot(mount_x=0.0)
    braced, _ = compile_robot()
    shift = braced.body("arm_mount").pos - stock.body("arm_mount").pos
    assert shift == pytest.approx([0.15, 0.0, 0.006], abs=1e-12)


def test_front_bracket_stays_on_the_trunk():
    plate = bracket_plate(resolve_mount(None))
    model, _ = compile_robot()
    trunk_half_length = max(
        model.geom_size[g][0]
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == model.body("base").id
        and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
    )
    assert plate["center"][0] + plate["size"][0] / 2 <= trunk_half_length


def test_kinematic_reach_matches_official_values():
    reach = d1_estimate.horizontal_reach()
    parameters = load_parameters()
    assert reach["to_joint6_m"] == pytest.approx(
        parameters["reach_without_gripper_m"]["value"], abs=0.01
    )
    assert reach["to_tip_m"] == pytest.approx(
        parameters["reach_with_gripper_m"]["value"], abs=0.015
    )


def test_rated_torque_matches_a_hand_moment_sum():
    # All links and the payload lie forward of the J2 axis, so sum m * g * horizontal distance.
    model, data = build_arm_only("scaled")
    for name, value in d1_estimate.RATED_POSE.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    shoulder = data.xanchor[model.joint("arm_joint2").id]
    first = model.body("arm_Empty_Link2").id
    moment = 0.0
    for index in range(first, model.nbody):
        moment += model.body_mass[index] * (data.xipos[index][0] - shoulder[0])
    moment += 0.5 * (data.site("arm_payload").xpos[0] - shoulder[0])
    computed = d1_estimate.gravity_torques("scaled", 0.5)[1]
    assert abs(computed) == pytest.approx(moment * 9.81, rel=1e-9)


def test_servo_lumped_keeps_inertia_physical():
    model, _ = build_arm_only("servo_lumped")
    for index in range(model.nbody):
        if model.body_mass[index] > 0:
            a, b, c = sorted(model.body_inertia[index])
            assert a + b >= c - 1e-9


def test_public_torque_limits_cannot_hold_rated_payload():
    # Documents why the public limits are not used as capacity. See docs/parameters.md.
    result = d1_estimate.estimate()
    assert result["torque_lower_bound_nm"][1] > max(
        result["public_limits_nm"]["urdf"][1], result["public_limits_nm"]["osu"][1]
    )


def test_recorded_torque_bound_matches_the_estimate():
    recorded = load_parameters()["arm_torque_lower_bound_nm"]["value"]
    assert recorded == pytest.approx(d1_estimate.estimate()["torque_lower_bound_nm"], abs=1e-3)


def test_urdf_has_nine_links():
    assert len(read_urdf()) == 9
