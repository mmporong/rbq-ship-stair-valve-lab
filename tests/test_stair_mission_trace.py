from types import SimpleNamespace

import numpy as np

from go2_wall.robot import LEGS


def test_replant_failure_order_for_live_and_frozen_recordings(monkeypatch):
    monkeypatch.syspath_prepend("tools")
    from replay_stair_mission import first_component_failure
    from stair_mission_control_profile import RL_REPLANT_PROFILE, with_failed_replant
    from verify_stair_replay import failed_replant_verdicts

    slip = {"metric": "stance_slip_m", "time_s": 4.197}
    contact = {"metric": "unexpected_contact_force_n", "time_s": 51.747}
    unknown = {"metric": "missing_input", "time_s": None}
    walking = {"verified": False, "first_failure": slip, "failures": [slip, unknown]}
    valve = {"verified": False, "first_failure": contact, "failures": [contact]}
    writer = "511f2ba2595fd6fd0691799b0999139a3eb2fdcaa1f5b566176b3f4697adcb79"
    manifest = {
        "control_profile": {"name": RL_REPLANT_PROFILE},
        "source_sha256": {"tools/replay_stair_mission.py": writer},
    }
    observed, archived = failed_replant_verdicts(walking, 81.6, manifest)
    assert observed == with_failed_replant(walking, 81.6)
    assert [fault["time_s"] for fault in observed["failures"]] == [4.197, 81.6, None]
    assert [fault["time_s"] for fault in archived["failures"]] == [4.197, None, 81.6]
    assert archived["first_failure"]["metric"] == "physical_replant"
    assert first_component_failure([archived, valve]) == slip
    assert first_component_failure([archived, valve], first_fields_only=True) == contact
    assert walking["failures"] == [slip, unknown]
    for profile, digest in (("other-profile", writer), (RL_REPLANT_PROFILE, "other-writer")):
        manifest["control_profile"]["name"] = profile
        manifest["source_sha256"]["tools/replay_stair_mission.py"] = digest
        current, recorded = failed_replant_verdicts(walking, 81.6, manifest)
        assert recorded == current == observed and recorded != archived

    clear = {"verified": True, "first_failure": None, "failures": []}
    assert with_failed_replant(clear, 81.6)["first_failure"]["time_s"] == 81.6
    assert first_component_failure([clear]) is None


def test_walk_uses_first_failed_expected_support_with_physical_time(monkeypatch):
    monkeypatch.syspath_prepend("tools")
    from stair_mission_trace import measured_walk

    time = np.arange(1, 121) * 0.005
    n = len(time)
    geometry = {
        "rise_m": 0.19,
        "foot_radius_m": 0.03,
        "stair_geometry": {"clear_width_m": 0.61, "tread_depth_m": 0.127},
    }
    model = SimpleNamespace(
        geom=lambda name: SimpleNamespace(
            pos=np.array([2.0, 0.0, 1.0]), size=np.array([0.5, 0.5, 0.01])
        )
    )
    qpos = np.zeros((n, 27))
    qpos[:, 3] = 1.0
    log = {
        "time": time - 0.005,
        "qpos": qpos,
        "qvel": np.zeros((n, 26)),
        "feet": np.zeros((n, 4, 3)),
        "base": qpos[:, :7].copy(),
    }
    loads = np.zeros((n, 4, 12))
    loads[:, :, 0] = 20
    trace = {
        "time_s": time,
        "stage": np.repeat("prewalk", n),
        "fell": np.zeros(n),
        "robot_torque_requested_nm": np.zeros((n, 18)),
        "qp_attempt": np.zeros(n),
        "base_track_m": np.zeros(n),
        "stance_slip_m": np.zeros(n),
        "pad_normal_force_n": np.zeros((n, 2)),
        "unexpected_contact_force_n": np.zeros(n),
    }
    stance = {leg: (np.array([0.0, 0.0, 0.03]), "floor") for leg in LEGS}
    args = (geometry, model, [(0.6, stance, "FL")], log, loads, trace, np.ones(18))
    clean = measured_walk(*args)
    assert clean["verified"] and clean["landings_evaluated"] == 1
    loads[95, 1, 0] = 0.4
    loads[100, 1, 0] = 0.0
    fault = measured_walk(*args)["first_failure"]
    assert fault["time_s"] == time[95] and fault["measured"] == 0.4
    assert fault["metric"] == "landing_support_n" and fault["landing"] == 1
    full = measured_walk(*args, full=True)
    assert {f["metric"] for f in full["failures"]} >= {
        "platform_turn_error_deg",
        "landing_count",
        "return_floor_support_n",
    }
    for key, values in (
        ("qpos", qpos),
        ("qvel", log["qvel"]),
        ("robot_torque_requested_nm", trace["robot_torque_requested_nm"]),
    ):
        saved = values[4, 3]
        values[4, 3] = np.nan
        bad = measured_walk(*args, full=True)
        assert not bad["verified"] and bad["first_failure"]["metric"] == "nonfinite_" + key
        assert bad["first_failure"]["time_s"] == time[4]
        values[4, 3] = saved
