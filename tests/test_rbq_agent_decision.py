import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "rbq_agent_decision", Path(__file__).parents[1] / "tools" / "rbq_agent_decision.py"
)
assert SPEC and SPEC.loader
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


def development():
    def group(candidate_id, work, collision, tread, pitch, simulated):
        trials = [
            {
                "case_id": f"dev-{index}",
                "candidate_id": candidate_id,
                "termination": "illegal_contact",
                "simulated_s": seconds,
            }
            for index, seconds in enumerate(simulated, 1)
        ]
        return {
            "summary": {
                "n": len(trials),
                "work_stance_rate": work,
                "mean_highest_supported_tread": tread,
                "body_collision_rate": collision,
                "mean_maximum_forward_x_m": -0.1,
                "mean_maximum_abs_pitch_deg": pitch,
            },
            "trials": trials,
        }

    return {
        "host_contract": "rbq_stair_intervention_v1",
        "registered_nvidia_task": False,
        "cpu_only": True,
        "checkpoint": 9100,
        "checkpoint_sha256": "a" * 64,
        "groups": {
            "no_intervention": group("no_intervention", 0.0, 1.0, 1.0, 20.0, [2.0, 3.0]),
            "phase_25": group("phase_25", 0.5, 0.0, 3.0, 10.0, [19.96, 19.96]),
        },
    }


def catalog():
    return {
        "schema": "rbq_intervention_catalog_v1",
        "candidates": [
            {
                "candidate_id": "no_intervention",
                "description": "unchanged",
                "executable": True,
                "mechanics": {"changes_policy": False},
            },
            {
                "candidate_id": "phase_25",
                "description": "phase shift",
                "executable": True,
                "mechanics": {"changes_gait_phase": True},
            },
        ],
    }


def valid_decision(phase):
    ranking = ["termination", "reward", "actuator", "exploration", "optimizer", "physics"]
    if phase == 0:
        return {
            "ranking": ranking,
            "reason": "Measure the observed early termination.",
            "claims": [{"id": "baseline:premature_termination_count", "value": 2}],
        }
    if phase == 1:
        return {
            "ranking": ranking,
            "reason": "No time-limit termination was observed, so the native probe is unknown.",
            "claims": [{"id": "baseline:premature_termination_count", "value": 2}],
        }
    return {
        "ranking": ranking,
        "reason": "No observation is available.",
        "claims": [{"id": "baseline:premature_termination_count", "value": 2}],
    }


def test_run_uses_original_validation_and_source_bound_observation(tmp_path):
    development_path = tmp_path / "development.json"
    catalog_path = tmp_path / "catalog.json"
    development_path.write_text(json.dumps(development()), encoding="utf-8")
    catalog_path.write_text(json.dumps(catalog()), encoding="utf-8")
    calls = []

    def chat(messages, schema):
        calls.append((messages, schema))
        if len(calls) <= 3:
            return {"model": "fixture", "decision": valid_decision(len(calls) - 1)}
        if len(calls) == 4:
            return {
                "model": "fixture",
                "decision": {
                    "candidate_id": "phase_25",
                    "claims": [
                        "receipt:rbq_host_development_failure:body_collision_rate",
                        "candidate:phase_25:mechanics.changes_gait_phase",
                    ],
                    "reason": "It changes mechanics after an observed failure.",
                },
            }
        return {
            "model": "fixture",
            "decision": {
                "candidate_id": "phase_25",
                "claims": [
                    "baseline:premature_termination_count",
                    "candidate:phase_25:mechanics.changes_gait_phase",
                ],
                "reason": "The baseline facts leave the same proposal plausible.",
            },
        }

    result = D.run(development_path, catalog_path, "unused", chat=chat)

    assert len(calls) == 5
    assert result["evidence_state"]["hypotheses"] == list(D.probe_loop.MECHANISMS)
    assert result["evidence_state"]["status"] == "unidentifiable"
    assert result["contrast"]["latest_receipt_cited"] is False
    assert result["contrast"]["safe_confirm"] is False
    assert result["model_runs"]["after"]["decision"]["action"] == "hold"
    assert result["model_runs"]["without_observation"]["decision"]["action"] == "hold"
    assert result["model_runs"]["intervention_selection"]["decision"]["candidate_id"] == "phase_25"
    assert result["selection_contrast"] == {
        "candidate_changed_with_observation": False,
        "observed_candidate": "phase_25",
        "without_observation_candidate": "phase_25",
        "same_selector_contract": True,
        "observed_prompt_sha256": result["selection_contrast"]["observed_prompt_sha256"],
        "without_observation_prompt_sha256": result["selection_contrast"][
            "without_observation_prompt_sha256"
        ],
    }
    assert result["boundary"]["execution_authority"] is False
    assert result["selection_is_efficacy_evidence"] is False


def test_catalog_rejects_outcome_fields_or_non_executable_candidates():
    invalid = catalog()
    invalid["candidates"][1]["mechanics"] = {"success_rate": 1.0}
    with pytest.raises(ValueError, match="outcome-like"):
        D.validate_catalog(invalid, development())

    invalid = catalog()
    invalid["candidates"][1]["executable"] = False
    with pytest.raises(ValueError, match="not executable"):
        D.validate_catalog(invalid, development())


def test_selection_rejects_uncited_or_out_of_catalog_choice():
    facts = {
        "candidate:a:mechanics.changes_policy": False,
        "candidate:b:mechanics.changes_policy": True,
    }
    with pytest.raises(ValueError, match="outside"):
        D.validate_selection(
            {"candidate_id": "c", "claims": list(facts), "reason": "bad"}, ["a", "b"], facts
        )
    with pytest.raises(ValueError, match="exact development fact"):
        D.validate_selection(
            {"candidate_id": "b", "claims": ["made-up"], "reason": "bad"}, ["a", "b"], facts
        )

    observed = {
        **facts,
        "receipt:rbq_host_development_failure:body_collision_rate": 1.0,
    }
    with pytest.raises(ValueError, match="source-bound receipt"):
        D.validate_selection(
            {
                "candidate_id": "b",
                "claims": [
                    "candidate:b:mechanics.changes_policy",
                    "candidate:a:mechanics.changes_policy",
                ],
                "reason": "missing receipt",
            },
            ["a", "b"],
            observed,
        )


def test_episode_probe_never_treats_safety_termination_as_timeout_bug():
    receipt = D.episode_receipt(development())
    assert receipt["outcome"] == "unknown"
    assert receipt["measurement"]["timeout_trial_count"] == 0
    assert receipt["measurement"]["timeout_ratio"] is None
    host = D.rbq_host_receipt(development())
    assert host["probe"] == "rbq_work_stance_and_contact_outcomes"
    assert host["outcome"] == "abnormal"
