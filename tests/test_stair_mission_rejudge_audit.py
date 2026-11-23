import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def audit_module():
    path = Path(__file__).resolve().parents[1] / "evidence/stair-mission-20261121/audit/rejudge.py"
    spec = importlib.util.spec_from_file_location("stair_mission_rejudge_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_byte_comparison_distinguishes_signed_zero_and_dtype(audit_module):
    positive_zero = np.array([0.0], dtype=np.float64)
    negative_zero = np.array([-0.0], dtype=np.float64)

    assert not audit_module.array_bytes_equal(positive_zero, negative_zero)
    assert not audit_module.array_bytes_equal(positive_zero, positive_zero.astype(np.float32))
    assert audit_module.array_bytes_equal(positive_zero, positive_zero.copy())


def test_recorded_task_duration_preserves_frozen_track_boundary(audit_module):
    assert (
        audit_module.recorded_task_duration(
            {"track_duration_s": 29.22, "boundaries_s": {"duration": 29.25}}
        )
        == 29.22
    )
    assert audit_module.recorded_task_duration({"boundaries_s": {"duration": 29.25}}) == 1462 * 0.02
    assert audit_module.recorded_task_duration({"boundaries_s": {"duration": 26.0}}) == 26.0


def test_same_raw_path_is_not_an_independent_repeat(audit_module, tmp_path):
    np.savez(tmp_path / "mission.npz", time_s=np.array([0.005]), value=np.array([1]))
    run = {
        "folder": str(tmp_path),
        "raw_sha256": "same-raw",
        "scene_sha256": "same-scene",
    }

    comparisons = audit_module.compare_runs([run, run])
    gate = audit_module.comparison_gate([run, run], comparisons, strict=True)

    assert comparisons[0]["distinct_raw_files"] is False
    assert comparisons[0]["strict_reproduction_passed"] is False
    assert gate["passed"] is False


def test_side_evidence_mismatch_fails_gate(audit_module):
    gate = audit_module.side_evidence_gate({"passed": True}, {"passed": False})

    assert gate == {
        "low_load_positive_negative_passed": True,
        "physical_b_17_conditions_passed": False,
        "passed": False,
    }
