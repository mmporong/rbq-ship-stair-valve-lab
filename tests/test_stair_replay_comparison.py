import json

import numpy as np
import pytest


def test_rejudge_uses_the_recorded_track_end_for_off_grid_durations(monkeypatch):
    monkeypatch.syspath_prepend("tools")
    from verify_stair_replay import recorded_task_duration

    motion = {"boundaries_s": {"duration": 29.25}}
    assert recorded_task_duration(motion) == 1462 * 0.02
    assert recorded_task_duration({**motion, "track_duration_s": 29.25}) == 29.25
    assert recorded_task_duration({"boundaries_s": {"duration": 26.0}}) == 26.0


def test_compare_preserves_signed_zero_and_effective_servo_period(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend("tools")
    from compare_stair_replays import compare

    first, second = tmp_path / "first", tmp_path / "second"
    scene = {"timestep_s": 0.005, "source_sha256": {}}
    for folder in (first, second):
        folder.mkdir()
        (folder / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
        np.savez_compressed(folder / "mission.npz", time_s=[0.005], force=[0.0])
    assert compare(first, second)["exact_raw_file_match"]

    changed_source = {**scene, "source_sha256": {"controller": "changed"}}
    (second / "scene.json").write_text(json.dumps(changed_source), encoding="utf-8")
    instrumented = compare(first, second)
    assert instrumented["same_scene_and_common_samples"]
    assert not instrumented["same_source_scene_and_common_samples"]

    scene["actuation_period_s"] = 0.010
    (second / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    np.savez_compressed(second / "mission.npz", time_s=[0.005], force=[-0.0])
    result = compare(first, second)
    assert result["raw_differences"] == {"force": {"first_sample": 0}}
    assert result["scene_field_differences"] == ["actuation_period_s"]
    assert not result["same_scene_and_common_samples"]

    scene.update(
        jaw_load_bias_m=0.0005,
        stance_spring_gain_n_per_m=2500,
        stance_spring_active_s=[5.0, 21.25],
        extra_settle_s=2.0,
    )
    (second / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    assert set(compare(first, second)["scene_field_differences"]) == {
        "actuation_period_s",
        "jaw_load_bias_m",
        "stance_spring_gain_n_per_m",
        "stance_spring_active_s",
        "extra_settle_s",
    }


def test_compare_requires_complete_optional_load_capture_and_equal_walking_input(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend("tools")
    from compare_stair_replays import LOAD_CHANNELS, compare

    a, b = tmp_path / "a", tmp_path / "b"
    scene = {"timestep_s": 0.001, "source_sha256": {}, "walking_reference": {"sha256": "abc"}}
    for directory in (a, b):
        directory.mkdir()
        (directory / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
        np.savez(directory / "mission.npz", time_s=[0.001])
    channels = {k: [2.0] for k in LOAD_CHANNELS}
    np.savez(a / "mission.npz", time_s=[0.001], **channels)
    assert compare(a, b, allow_load_observer=True)["same_scene_and_common_samples"]
    with pytest.raises(ValueError, match="Unmatched raw"):
        compare(a, b)
    scene["walking_reference"]["sha256"] = "different"
    (b / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    assert compare(a, b, allow_load_observer=True)["scene_field_differences"] == [
        "walking_reference"
    ]
    channels.pop("joint_moment_world_nm")
    np.savez(a / "mission.npz", time_s=[0.001], **channels)
    with pytest.raises(ValueError, match="incomplete"):
        compare(a, b, allow_load_observer=True)
