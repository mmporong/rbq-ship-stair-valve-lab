import importlib.util
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1] / "tools" / "triage_rbq_host_ab.py"
SPEC = importlib.util.spec_from_file_location("triage_rbq_host_ab", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_case_commitment_is_canonical_and_order_sensitive():
    cases = [
        {
            "case_id": "x",
            "lateral_m": 0.0,
            "yaw_deg": 0.0,
            "joint_seed": 1,
            "joint_noise_rad": 0.0,
        }
    ]
    assert MODULE.canonical_sha256(cases) == MODULE.canonical_sha256(MODULE.validate_cases(cases))
    assert MODULE.canonical_sha256(cases) != MODULE.canonical_sha256(cases + cases)


def test_case_validator_rejects_hidden_extra_fields():
    with pytest.raises(ValueError, match="fields"):
        MODULE.validate_cases(
            [
                {
                    "case_id": "x",
                    "lateral_m": 0.0,
                    "yaw_deg": 0.0,
                    "joint_seed": 1,
                    "joint_noise_rad": 0.0,
                    "candidate": "phase_25",
                }
            ]
        )


def test_summary_reports_rates_and_means():
    trials = [
        {
            "work_stance_reached": False,
            "maximum_work_stance_s": 0.1,
            "highest_supported_tread": 3,
            "body_collision": True,
            "maximum_forward_x_m": -0.1,
            "maximum_abs_pitch_deg": 20.0,
        },
        {
            "work_stance_reached": True,
            "maximum_work_stance_s": 0.5,
            "highest_supported_tread": 5,
            "body_collision": False,
            "maximum_forward_x_m": 0.1,
            "maximum_abs_pitch_deg": 10.0,
        },
    ]
    result = MODULE.summarize(trials)
    assert result["work_stance_rate"] == 0.5
    assert result["mean_highest_supported_tread"] == 4.0
    assert result["body_collision_rate"] == 0.5
