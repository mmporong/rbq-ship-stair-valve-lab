from pathlib import Path

import mujoco
import numpy as np
import pytest

from go2_wall import stair_valve as sv


@pytest.fixture(scope="module")
def scene():
    return sv.build_scene(50.0, 0.19, [0.8, 0.0, 1.5], 30.0, 1, 0.12, 10.0)


def test_scene_has_passive_quarter_turn_valve_and_ideal_pin(scene):
    model, data = scene
    hinge = model.joint(sv.VALVE_HINGE)
    dof = hinge.dofadr[0]

    assert model.nu == 18
    assert sum(model.geom(i).name.startswith("tread") for i in range(model.ngeom)) == sv.TREADS
    assert [model.actuator(i).name for i in range(model.nu)][-6:] == [
        f"arm_motor{i}" for i in range(1, 7)
    ]
    assert np.allclose(model.jnt_axis[hinge.id], [0.0, -1.0, 0.0])
    assert np.allclose(model.jnt_range[hinge.id], [0.0, np.pi / 2])
    assert model.dof_frictionloss[dof] == pytest.approx(10.0)
    assert model.equality(sv.VALVE_PIN).type == mujoco.mjtEq.mjEQ_CONNECT
    assert model.eq_objtype[model.equality(sv.VALVE_PIN).id] == mujoco.mjtObj.mjOBJ_SITE

    mujoco.mj_forward(model, data)
    expected = np.array([0.8, 0.0, 1.5]) + 0.12 * np.array(
        [np.cos(np.radians(30.0)), 0.0, np.sin(np.radians(30.0))]
    )
    assert np.allclose(data.site(sv.VALVE_HANDLE).xpos, expected)


@pytest.mark.parametrize(("sense", "expected_z_sign"), [(1, 1), (-1, -1)])
def test_positive_hinge_position_follows_sense(sense, expected_z_sign):
    model, data = sv.build_scene(50.0, 0.19, [0.8, 0.0, 1.5], 0.0, sense, 0.12, 10.0)
    hinge = model.joint(sv.VALVE_HINGE)
    data.qpos[hinge.qposadr[0]] = 0.2
    mujoco.mj_forward(model, data)
    radial = data.site(sv.VALVE_HANDLE).xpos - np.array([0.8, 0.0, 1.5])
    assert np.sign(radial[2]) == expected_z_sign


def test_observe_separates_signed_friction_and_pin_torque(scene):
    model, data = scene
    hinge = model.joint(sv.VALVE_HINGE)
    dof = hinge.dofadr[0]
    data.qpos[hinge.qposadr[0]] = 0.2
    data.qvel[dof] = 1.0
    mujoco.mj_forward(model, data)

    observation = sv.observe(model, data)

    assert observation.shape == (len(sv.OBSERVATION_FIELDS),)
    assert np.all(np.isfinite(observation))
    assert observation[0] == pytest.approx(0.2)
    assert observation[1] == pytest.approx(1.0)
    assert observation[2] == pytest.approx(-10.0, abs=1e-8)
    assert observation[3] == pytest.approx(
        np.linalg.norm(data.site(sv.VALVE_HANDLE).xpos - data.site("arm_tip").xpos)
    )
    assert abs(observation[4]) > 0.0

    pin = model.equality(sv.VALVE_PIN).id
    rows = (data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (
        data.efc_id[: data.nefc] == pin
    )
    jacobian = data.efc_J.reshape(data.nefc, model.nv)[rows]
    assert np.any(jacobian[:, dof] != 0.0)
    assert np.any(jacobian[:, :dof] != 0.0)


def test_valve_pass_requires_measured_resistance_and_tread_support(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from replay_stair_valve import HOLD_S, INITIAL_HOLD_S, evaluate

    motion_s, settle_s = 2.0, 1.0
    times = np.arange(round((INITIAL_HOLD_S + motion_s + settle_s + HOLD_S) * 100)) / 100
    fraction = np.clip((times - INITIAL_HOLD_S) / motion_s, 0.0, 1.0)
    support_start = len(sv.OBSERVATION_FIELDS)
    observed = np.zeros((times.size, support_start + 4))
    observed[:, 0] = fraction * np.pi / 2
    observed[:, 1] = np.gradient(observed[:, 0], times)
    observed[:, 2] = -10.0
    observed[:, support_start:] = 120.0
    foot_force = np.zeros((times.size, 4, 3))
    foot_force[..., 2] = 120.0
    log = {
        "time": times,
        "extra": observed,
        "force": foot_force,
        "base": np.zeros((times.size, 7)),
        "feet": np.zeros_like(foot_force),
    }
    metrics = {"max_ratio": 0.4, "saturated_steps": 0, "fell": False, "unexpected_contacts_n": {}}
    assert evaluate(log, metrics, motion_s, settle_s, 10.0)["verified"]

    midstroke = np.flatnonzero((fraction > 0.3) & (fraction < 0.7))[0]
    observed[midstroke, 2] = -7.0
    assert not evaluate(log, metrics, motion_s, settle_s, 10.0)["verified"]
    observed[midstroke, 2] = -10.0
    observed[midstroke, support_start] = 0.0
    assert not evaluate(log, metrics, motion_s, settle_s, 10.0)["verified"]
