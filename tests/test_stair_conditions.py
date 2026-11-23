import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "tools/assess_stair_conditions.py"
SPEC = importlib.util.spec_from_file_location("assess_stair_conditions", MODULE_PATH)
assert SPEC and SPEC.loader
conditions = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(conditions)


def _protocol():
    return {
        "plan_commit": "frozen",
        "single_factor_cases": {"nominal": {}, "slope_60": {"slope_deg": 60}},
        "sensitivity": {"lever_m": [0.1, 0.14], "arm_limits": ["named limits"]},
        "boundary_cases": {"high_rise": {"rise_m": 0.21}},
    }


def _write(path: Path, payload: dict, settings=None, *, scene_extra=None, source=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    planner = {"planning": "test-revision"}
    payload = {**payload, "planner_source_sha256": planner}
    if path.name == "result.json":
        scene = {
            "extent": payload.get("extent"),
            "settings": settings or {},
            "source_sha256": source or {"grasp": "test-revision"},
            **(scene_extra or {}),
        }
        scene["planner_source_sha256"] = planner
        raw = json.dumps(scene).encode()
        path.with_name("mission.npz").write_bytes(b"fixture trace")
        payload["raw_sha256"] = hashlib.sha256(b"fixture trace").hexdigest()
        path.with_name("scene.json").write_bytes(raw)
        payload = {
            **payload,
            "settings": scene["settings"],
            "scene_sha256": hashlib.sha256(raw).hexdigest(),
        }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _frozen_profile():
    return {
        "name": "rl-mid-replant-k3075-v1",
        "replant_after_landing_index": 15,
        "replant_trigger_s": 78.8,
        "replant_duration_s": 2.8,
        "replant_leg": "RL",
        "slip_spring_gain_n_per_m": 3075,
        "slip_spring_active_s": [116.65, 129.4],
        "stance_spring_gain_n_per_m": 2500,
        "terminal_overtravel_deg": 1.3,
    }


def _recorded_profile(*, verified=True):
    return {
        "name": "rl-mid-replant-k3075-v1",
        "trigger_landing_index": 15,
        "trigger_time_s": 78.8,
        "duration_s": 2.8,
        "replant_leg": "RL",
        "slip_spring_gain_n_per_m": 3075.0,
        "slip_spring_active_s": [116.65, 129.4],
        "stance_spring_gain_n_per_m": 2500.0,
        "terminal_overtravel_deg": 1.3,
        "physical_replant": {
            "original_scheduled_landings": 156,
            "auxiliary_replant_landings": 1,
            "total_scheduled_landings": 157,
            "minimum_total_replant_leg_load_n": 0.0,
            "maximum_foot_lift_m": 0.05,
            "released_from_support": True,
            "relanding": {
                "moved_leg": "RL",
                "terminal_foot_xy_error_m": 0.001,
                "support": {"all_expected_feet_ge_1n": True},
            },
            "verified": verified,
        },
    }


def test_protocol_rows_expand_every_declared_group():
    rows = conditions.protocol_rows(_protocol())

    assert [row["group"] for row in rows] == [
        "single_factor",
        "single_factor",
        "sensitivity",
        "sensitivity",
        "sensitivity",
        "boundary",
    ]
    assert rows[2]["changes"] == {"lever_m": 0.1}
    assert rows[-1]["id"] == "high_rise"


def test_frozen_protocol_has_eight_single_factor_and_nine_sensitivity_rows():
    protocol_path = Path(__file__).parents[1] / "evidence/stair-mission-20261121/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rows = conditions.protocol_rows(protocol)

    assert sum(row["group"] == "single_factor" for row in rows) == 8
    assert sum(row["group"] == "sensitivity" for row in rows) == 9


def test_aggregate_uses_real_files_and_leaves_missing_rows_pending(tmp_path):
    failure = _write(
        tmp_path / "condition-60.failure.json",
        {
            "outcome": "planning_failed",
            "slope_deg": 60,
            "first_failure": {
                "segment": 16,
                "kind": "shift",
                "planned_time_s": 13.6,
                "foot_ik_error_m": {"FL": 0.001, "RR": 0.003},
                "foot_ik_limit_m": 0.002,
            },
        },
    )

    report = conditions.aggregate(_protocol(), [failure], relative_to=tmp_path)
    slope = next(row for row in report["conditions"] if row["id"] == "slope_60")

    assert slope["outcome"] == "planning_failed"
    assert slope["evidence_path"] == "condition-60.failure.json"
    assert slope["first_failure"] == {
        "stage": "shift",
        "time_s": 13.6,
        "metric": "max_foot_ik_error_m",
        "measured": 0.003,
        "limit": 0.002,
    }
    assert report["counts"] == {
        "success": 0,
        "planning_failed": 1,
        "physical_failed": 0,
        "pending": 5,
    }


def test_aggregate_classifies_physical_failure_and_success(tmp_path):
    physical = _write(
        tmp_path / "nominal" / "result.json",
        {
            "verified": False,
            "first_failure": {
                "stage": "hold",
                "time_s": 8.0,
                "metric": "angle_error_deg",
                "measured": 0.4,
                "limit": 0.25,
            },
        },
    )
    success = _write(
        tmp_path / "sensitivity_lever_m_0_1" / "result.json",
        {"extent": "full", "mission": {"verified": True}},
        {"handle_length_m": 0.1},
    )

    report = conditions.aggregate(_protocol(), [physical, success], relative_to=tmp_path)
    by_id = {row["id"]: row for row in report["conditions"]}

    assert by_id["nominal"]["outcome"] == "physical_failed"
    assert by_id["nominal"]["first_failure"]["stage"] == "hold"
    assert by_id["sensitivity_lever_m_0_1"]["outcome"] == "success"
    assert by_id["sensitivity_lever_m_0_1"]["first_failure"] is None


def test_condition_uses_earliest_timed_component_failure():
    def fault(metric, time_s):
        return {"stage": "prewalk", "metric": metric, "time_s": time_s}

    slip = fault("stance_slip_m", 4.197)
    contact = fault("unexpected_contact_force_n", 51.747)
    replant = fault("physical_replant", 81.6)
    untimed = fault("missing_input", None)
    invalid = fault("invalid_time", float("nan"))
    case = {
        "first_failure": contact,
        "walking": {"first_failure": replant, "failures": [untimed, replant, invalid, slip]},
        "valve": {"first_failure": contact},
    }
    first = conditions._first_failure_summary(case)
    assert first["metric"] == "stance_slip_m" and first["time_s"] == 4.197
    case["first_failure"] = fault("tied_first", 4.197)
    assert conditions._first_failure_summary(case)["metric"] == "tied_first"
    assert conditions._first_failure_summary({"first_failure": invalid})["time_s"] is None
    fallback = conditions._first_failure_summary({"first_failure": untimed})
    assert fallback["metric"] == "missing_input"


def test_through_task_failure_is_physical_but_partial_pass_stays_pending(tmp_path):
    failed = _write(
        tmp_path / "failed" / "result.json",
        {
            "extent": "through-task",
            "valve": {
                "verified": False,
                "first_failure": {
                    "stage": "contact",
                    "time_s": 3.2,
                    "metric": "pad_force_n",
                    "measured": 0.2,
                    "limit": 1.0,
                },
            },
        },
    )
    passed = _write(
        tmp_path / "passed" / "result.json",
        {"extent": "through-task", "valve": {"verified": True}},
        {"slope_deg": 60},
    )

    report = conditions.aggregate(
        _protocol(),
        [failed, passed],
        relative_to=tmp_path,
        assigned_ids={failed: "nominal", passed: "slope_60"},
    )
    by_id = {row["id"]: row for row in report["conditions"]}

    assert by_id["nominal"]["outcome"] == "physical_failed"
    assert by_id["nominal"]["first_failure"]["stage"] == "contact"
    assert by_id["slope_60"]["outcome"] == "pending"


def test_partial_top_level_success_is_not_whole_mission_success(tmp_path):
    result = _write(
        tmp_path / "result.json",
        {"extent": "approach", "outcome": "success"},
    )

    report = conditions.aggregate(
        _protocol(),
        [result],
        relative_to=tmp_path,
        assigned_ids={result: "nominal"},
    )

    assert report["conditions"][0]["outcome"] == "pending"


def test_through_task_walk_and_valve_pass_is_success(tmp_path):
    result = _write(
        tmp_path / "result.json",
        {
            "extent": "through-task",
            "outcome": "success",
            "walking": {"verified": True},
            "valve": {"verified": True},
        },
    )

    report = conditions.aggregate(
        _protocol(),
        [result],
        relative_to=tmp_path,
        assigned_ids={result: "nominal"},
    )

    assert report["conditions"][0]["outcome"] == "success"


@pytest.mark.parametrize("physical", [None, {"verified": False}])
def test_declared_profile_requires_a_verified_physical_replant(physical):
    profile = _recorded_profile()
    if physical is None:
        profile.pop("physical_replant")
    else:
        profile["physical_replant"] = physical
    payload = {
        "extent": "through-task",
        "outcome": "success",
        "walking": {"verified": True},
        "valve": {"verified": True},
        "control_profile": profile,
    }

    assert conditions._outcome(payload) == "physical_failed"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_scheduled_landings", 156),
        ("released_from_support", False),
        ("maximum_foot_lift_m", 0.0),
        ("minimum_total_replant_leg_load_n", 1.0),
    ],
)
def test_declared_profile_rejects_inconsistent_replant_measurements(field, value):
    profile = _recorded_profile()
    profile["physical_replant"][field] = value
    payload = {
        "extent": "through-task",
        "walking": {"verified": True},
        "valve": {"verified": True},
        "control_profile": profile,
    }

    assert conditions._outcome(payload) == "physical_failed"


def test_frozen_profile_and_source_certify_scoped_through_task_success(tmp_path):
    profile = _recorded_profile()
    source = {"controller": "frozen"}
    protocol = {
        **_protocol(),
        "control_profile": _frozen_profile(),
        "source_sha256": source,
    }
    result = _write(
        tmp_path / "nominal" / "result.json",
        {
            "extent": "through-task",
            "walking": {"verified": True},
            "valve": {"verified": True},
            "control_profile": profile,
        },
        scene_extra={
            "control_profile": {
                key: value for key, value in profile.items() if key != "physical_replant"
            }
        },
        source=source,
    )

    report = conditions.aggregate(protocol, [result], relative_to=tmp_path)

    assert report["conditions"][0]["outcome"] == "success"


@pytest.mark.parametrize("mismatch", ["source", "scene_profile", "result_profile"])
def test_frozen_profile_integrity_mismatch_is_rejected(tmp_path, mismatch):
    source = {"controller": "frozen"}
    profile = _recorded_profile()
    scene_profile = {k: v for k, v in profile.items() if k != "physical_replant"}
    protocol = {
        **_protocol(),
        "control_profile": _frozen_profile(),
        "source_sha256": source,
    }
    observed_source = {"controller": "changed"} if mismatch == "source" else source
    if mismatch == "scene_profile":
        scene_profile["duration_s"] = 3.0
    if mismatch == "result_profile":
        profile["duration_s"] = 3.0
    result = _write(
        tmp_path / "nominal" / "result.json",
        {
            "extent": "through-task",
            "walking": {"verified": True},
            "valve": {"verified": True},
            "control_profile": profile,
        },
        scene_extra={"control_profile": scene_profile},
        source=observed_source,
    )

    with pytest.raises(ValueError, match="source hashes|control profile|control profiles"):
        conditions.aggregate(protocol, [result], relative_to=tmp_path)


def test_run_assignment_binds_category_to_one_result(tmp_path):
    result = _write(tmp_path / "arbitrary-name" / "result.json", {"outcome": "success"})

    case_id, path = conditions.run_assignment(f"rise_200={result.parent}")

    assert case_id == "rise_200"
    assert path == result.resolve()


def test_assignment_cannot_change_the_measured_condition_or_override_failure(tmp_path):
    result = _write(
        tmp_path / "nominal" / "result.json",
        {"extent": "full", "outcome": "success", "mission": {"verified": False}},
        {"slope_deg": 50},
    )
    with pytest.raises(ValueError, match="expected slope_deg=60"):
        conditions.aggregate(
            _protocol(), [result], relative_to=tmp_path, assigned_ids={result: "slope_60"}
        )
    report = conditions.aggregate(_protocol(), [result], relative_to=tmp_path)
    assert report["conditions"][0]["outcome"] == "physical_failed"


def test_multiple_matching_results_are_rejected(tmp_path):
    first = _write(tmp_path / "condition-60.failure.json", {"outcome": "planning_failed"})
    second = _write(
        tmp_path / "other.failure.json",
        {"condition_id": "slope_60", "outcome": "planning_failed"},
    )

    with pytest.raises(ValueError, match="multiple results match slope_60"):
        conditions.aggregate(_protocol(), [first, second], relative_to=tmp_path)


def test_discovery_ignores_unrecognized_json(tmp_path):
    result = _write(tmp_path / "case" / "result.json", {"outcome": "success"})
    failure = _write(tmp_path / "case.failure.json", {"outcome": "planning_failed"})
    _write(tmp_path / "notes.json", {"outcome": "success"})

    assert conditions.discover_artifacts([tmp_path]) == sorted(
        [result.resolve(), failure.resolve()]
    )


def test_planning_failure_cannot_be_assigned_to_a_grip_sensitivity(tmp_path):
    failed = _write(
        tmp_path / "geometry.failure.json", {"outcome": "planning_failed", "slope_deg": 50}
    )
    with pytest.raises(ValueError, match="geometry planning failure"):
        conditions.aggregate(
            _protocol(),
            [failed],
            relative_to=tmp_path,
            assigned_ids={failed: "sensitivity_lever_m_0_1"},
        )
