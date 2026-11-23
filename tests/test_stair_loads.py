import mujoco
import numpy as np
import pytest

from go2_wall import stair_valve as sv
from go2_wall.robot import LEGS
from go2_wall.stair_loads import (
    FOOT_NAMES,
    JOINT_NAMES,
    MOUNT_WRENCH_FIELDS,
    SURFACE_NAMES,
    StairLoads,
)


def loaded_floor_scene():
    model, data = sv.build_grasp_scene(50.0, 0.19, [0.9, 0.0, 1.58], 270.0, -1, 0.12, 10.0)
    mujoco.mj_forward(model, data)
    foot_bottom = [
        data.geom_xpos[model.geom(name).id, 2] - model.geom_size[model.geom(name).id, 0]
        for name in LEGS
    ]
    data.qpos[2] -= min(foot_bottom) + 0.002
    mujoco.mj_forward(model, data)
    return model, data


def test_walking_caption_contract_detects_missing_and_truncated_channels(monkeypatch):
    monkeypatch.syspath_prepend("tools")
    from verify_stair_artifacts import check_walking_captions

    times = np.array([0.001, 0.002])
    data = {
        "stance_slip_m": np.array([0.008, 0.010]),
        "unexpected_contact_force_n": np.array([0, 12]),
    }
    slip = "지금까지 최대 지지발 미끄러짐 8.000mm\n지금까지 최대 지지발 미끄러짐 10.000mm"
    contact = "\n비허용 접촉력 0.00N\n비허용 접촉력 12.00N"
    assert check_walking_captions("", times, times, data) == {}
    assert check_walking_captions(slip + contact, times, times, data, required=True) == {
        "stance_slip_m": 2,
        "unexpected_contact_force_n": 2,
    }
    for incomplete in ("", slip, slip + "\n비허용 접촉력 0.00N"):
        with pytest.raises(ValueError, match="Walking captions differ"):
            check_walking_captions(incomplete, times, times, data, required=True)


def test_stair_loads_shapes_order_and_off_axis_projection():
    model, data = loaded_floor_scene()
    qpos, qvel, ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()

    sample = StairLoads(model).sample(data)

    assert FOOT_NAMES == tuple(LEGS)
    assert SURFACE_NAMES == ("floor", *(f"tread{k}" for k in range(1, 11)), "platform")
    assert len(JOINT_NAMES) == 18
    assert len(MOUNT_WRENCH_FIELDS) == 6
    assert sample["joint_force_world_n"].shape == (18, 3)
    assert sample["joint_moment_world_nm"].shape == (18, 3)
    assert sample["joint_off_axis_bending_nm"].shape == (18,)
    assert sample["joint_reaction_n"].shape == (18,)
    assert sample["rb1_mount_wrench_world"].shape == (6,)
    assert sample["foot_surface_force_world_n"].shape == (4, 12, 3)
    assert all(np.all(np.isfinite(value)) for value in sample.values())

    axes = np.array([data.xaxis[model.joint(name).id] for name in JOINT_NAMES])
    moments = sample["joint_moment_world_nm"]
    off_axis = moments - np.sum(moments * axes, axis=1)[:, None] * axes
    assert np.allclose(sample["joint_off_axis_bending_nm"], np.linalg.norm(off_axis, axis=1))
    assert np.allclose(
        sample["joint_reaction_n"], np.linalg.norm(sample["joint_force_world_n"], axis=1)
    )
    assert np.array_equal(data.qpos, qpos)
    assert np.array_equal(data.qvel, qvel)
    assert np.array_equal(data.ctrl, ctrl)


def test_foot_surface_reactions_are_forces_on_feet_in_world_axes():
    model, data = loaded_floor_scene()

    reactions = StairLoads(model).sample(data)["foot_surface_force_world_n"]

    assert np.allclose(reactions[:, 1:], 0.0)
    assert np.all(reactions[:, 0, 2] > 0.0)
    expected = np.zeros_like(reactions)
    feet = {model.geom(name).id: index for index, name in enumerate(FOOT_NAMES)}
    surfaces = {model.geom(name).id: index for index, name in enumerate(SURFACE_NAMES)}
    wrench = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.geom1 in surfaces and contact.geom2 in feet:
            surface, foot, sign = contact.geom1, contact.geom2, 1.0
        elif contact.geom2 in surfaces and contact.geom1 in feet:
            surface, foot, sign = contact.geom2, contact.geom1, -1.0
        else:
            continue
        mujoco.mj_contactForce(model, data, index, wrench)
        world = contact.frame.reshape(3, 3).T @ wrench[:3]
        expected[feet[foot], surfaces[surface]] += sign * world
    assert np.allclose(reactions, expected)


def test_render_summary_uses_exact_peak_samples_and_window(monkeypatch):
    monkeypatch.syspath_prepend("tools")
    import render_stair_mission as render

    count = 3
    mission = {
        "time_s": np.array([0.0, 0.5, 1.0]),
        "stage": np.array(["approach", "contact", "operate"]),
        "torque_nm": np.zeros((count, 18)),
        "joint_off_axis_bending_nm": np.zeros((count, 18)),
        "joint_reaction_n": np.zeros((count, 18)),
        "rb1_mount_wrench_world": np.zeros((count, 6)),
        "foot_surface_force_world_n": np.zeros((count, 4, 12, 3)),
        "tread_load_n": np.zeros((count, 4, 12)),
        "hinge_friction_torque_nm": np.zeros(count),
        "pad_closing_torque_nm": np.zeros(count),
        "pad_normal_force_n": np.zeros((count, 2)),
    }
    mission["torque_nm"][1, 2] = -20.0
    mission["joint_off_axis_bending_nm"][2, 3] = 7.0
    mission["rb1_mount_wrench_world"][1, 0] = 9.0
    mission["foot_surface_force_world_n"][2, 1, 4, 2] = 11.0

    manifest = {"timestep_s": 0.5}
    summary = render.peak_summary(manifest, mission)

    assert summary["mount_reference"].startswith("RB1_Base origin")
    assert summary["metrics"]["joint_torque_nm"]["value"] == 20.0
    assert summary["metrics"]["joint_torque_ratio"]["sample_index"] == 1
    assert summary["metrics"]["joint_torque_ratio"]["item"] == JOINT_NAMES[2]
    assert summary["metrics"]["foot_surface_force_n"]["item"] == (
        f"{FOOT_NAMES[1]}/{SURFACE_NAMES[4]}"
    )
    window = render.mission_window(mission, 0.4, 0.6)
    assert window["time_s"] == pytest.approx([0.5])
    assert window["torque_nm"].shape == (1, 18)
    page = render.heatmap_page(manifest, mission)
    assert "RB1_Base" in page
    assert "관절 토크 비율" in page
    mission["rb1_mount_wrench_world"][2, 4] = -11.0
    peaks = {
        (metric, name): (value, t, stage)
        for metric, name, unit, value, t, stage in render.load_rows(manifest, mission)
    }
    assert peaks["mount_moment", "moment_y_nm"] == (-11.0, 1.0, "operate")
    assert peaks["foot_surface_reaction", f"{FOOT_NAMES[1]}/{SURFACE_NAMES[4]}"] == (
        11.0,
        1.0,
        "operate",
    )
    _, binned = render._binned(mission["time_s"], mission["joint_off_axis_bending_nm"])
    assert binned[3, -1] == 7.0


def test_render_numeric_stage_requires_manifest_names(monkeypatch):
    monkeypatch.syspath_prepend("tools")
    import render_stair_mission as render

    assert render.stage_labels(np.array([0, 2]), {"stage_names": ["a", "b", "c"]}).tolist() == [
        "a",
        "c",
    ]
    with pytest.raises(ValueError, match="stage_names"):
        render.stage_labels(np.array([0]), {})
