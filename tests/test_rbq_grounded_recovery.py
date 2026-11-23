import importlib.util
import json
import os
from pathlib import Path

import pytest

os.environ["RBQ_TRIAGE_SRC"] = str(
    Path(__file__).parents[1] / ".omx/rbq-agent-body-dev/rl-triage-worktree/src"
)
SPEC = importlib.util.spec_from_file_location(
    "rbq_grounded_recovery", Path(__file__).parents[1] / "tools" / "rbq_grounded_recovery.py"
)
assert SPEC and SPEC.loader
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def candidate(candidate_id, mechanics):
    return {
        "candidate_id": candidate_id,
        "description": candidate_id,
        "executable": True,
        "mechanics": mechanics,
    }


def catalog():
    return {
        "schema": "rbq_intervention_catalog_v1",
        "candidates": [
            candidate("no_intervention", {"changes_policy": False}),
            candidate("phase_25", {"phase_offset": 25}),
            candidate("contact_gated_reference", {"uses_contact": True}),
        ],
    }


def trial(case_id, candidate_id, success, collision):
    return {
        "case_id": case_id,
        "candidate_id": candidate_id,
        "termination": "horizon" if success else "terrain_body_contact",
        "simulated_s": 10.0,
        "body_collision": collision,
        "max_nonfoot_terrain_force_n": 0.0 if not collision else 10.0,
        "highest_supported_tread": 5 if success else 0,
        "maximum_work_stance_s": 1.0 if success else 0.0,
        "work_stance_reached": success,
        "maximum_valve_angle_deg": 90.0 if success else 0.0,
        "final_valve_angle_deg": 90.0 if success else 0.0,
        "state_sha256": R.canonical_sha256([case_id, candidate_id]),
    }


def write_inputs(tmp_path):
    sources = {}
    for source_id in ("scene", "reference", "control_source", "simulation_source"):
        path = tmp_path / f"{source_id}.bin"
        path.write_text(source_id, encoding="utf-8")
        sources[source_id] = path
    development = {
        "case_set_sha256": "a" * 64,
        "reference_sha256": R.file_sha256(sources["reference"]),
        "scene_sha256": R.file_sha256(sources["scene"]),
        "control_source_sha256": R.file_sha256(sources["control_source"]),
        "simulation_source_sha256": R.file_sha256(sources["simulation_source"]),
        "model": "fixture",
        "clock_s": {"physics": 0.001},
        "trials": [],
    }
    for candidate_id in ("no_intervention", "phase_25", "contact_gated_reference"):
        for index in range(3):
            success = candidate_id == "contact_gated_reference" or (
                candidate_id == "phase_25" and index == 0
            )
            development["trials"].append(
                trial(f"dev-{index + 1}", candidate_id, success, not success)
            )
    development_path = tmp_path / "development.json"
    catalog_path = tmp_path / "catalog.json"
    development_path.write_text(json.dumps(development), encoding="utf-8")
    catalog_path.write_text(json.dumps(catalog()), encoding="utf-8")
    return development_path, catalog_path, sources


def proposal(candidate_id, schema):
    branch = next(
        row
        for row in schema["oneOf"]
        if row["properties"]["candidate_id"]["const"] == candidate_id
    )
    properties = branch["properties"]
    mechanics = properties["mechanics_fact_id"]["enum"][0]
    observation = properties["observation_fact_id"]["enum"][0]
    return {
        "model": "fixture",
        "message": {
            "content": json.dumps(
                {
                    "candidate_id": candidate_id,
                    "mechanics_fact_id": mechanics,
                    "observation_fact_id": observation,
                    "reason": "fixture",
                }
            )
        },
    }


def test_loop_reveals_only_proposed_results_and_adopts_second_candidate(tmp_path):
    development_path, catalog_path, sources = write_inputs(tmp_path)
    calls = []

    def chat(messages, schema):
        calls.append(messages)
        choices = ["no_intervention", "phase_25", "contact_gated_reference"]
        candidate_id = choices[len(calls) - 1]
        return proposal(candidate_id, schema)

    out = tmp_path / "out"
    result = R.run(development_path, catalog_path, sources, out, "unused", chat=chat)

    assert len(calls) == 3
    assert result["no_observation_ablation"]["proposal"]["candidate_id"] == "no_intervention"
    assert result["attempted_interventions"] == ["phase_25", "contact_gated_reference"]
    assert result["adopted_candidate"] == "contact_gated_reference"
    assert result["trace"][0]["adoption_threshold_met"] is False
    assert result["trace"][1]["adoption_threshold_met"] is True
    assert not (out / "step-02-physics-receipt.json").exists()
    assert json.loads((out / "summary.json").read_text())["efficacy_claim_supported"] is False
    assert all(
        step["proposal"]["native_diagnosis"]["action"] == "hold"
        for step in result["trace"]
    )
    first_prompt = calls[1][0]["content"]
    assert "work_stance_success_count" in first_prompt
    assert "adoption_threshold_met" not in first_prompt


def test_id_citation_binds_exact_source_values_and_rejects_unknown_ids():
    facts = {
        "candidate:phase_25:mechanics.phase_offset": 25,
        "receipt:baseline:body_collision_count": 3,
    }
    raw = {
        "candidate_id": "phase_25",
        "mechanics_fact_id": "candidate:phase_25:mechanics.phase_offset",
        "observation_fact_id": "receipt:baseline:body_collision_count",
        "reason": "source grounded",
    }
    candidate_facts = {"phase_25": ["candidate:phase_25:mechanics.phase_offset"]}
    observations = ["receipt:baseline:body_collision_count"]
    bound = R.recovery.bind_typed_proposal(
        raw, facts, ["phase_25"], candidate_facts, observations
    )
    assert [citation["value"] for citation in bound["citations"]] == [25, 3]
    assert bound["native_diagnosis"]["action"] == "hold"
    raw["observation_fact_id"] = "missing"
    with pytest.raises(ValueError, match="not an authorized"):
        R.recovery.bind_typed_proposal(
            raw, facts, ["phase_25"], candidate_facts, observations
        )


def test_candidate_receipt_uses_work_stance_and_collision_only_for_adoption():
    rows = [trial(f"dev-{index}", "candidate", index < 2, index == 2) for index in range(3)]
    receipt = R.candidate_receipt("candidate", rows, baseline_collision_count=1)
    assert receipt["measurement"]["adoption_threshold_met"] is True
    assert receipt["measurement"]["valve_terminal_indicator_count"] == 2


def test_source_hash_mismatch_fails_before_model_call(tmp_path):
    development_path, catalog_path, sources = write_inputs(tmp_path)
    sources["scene"].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        R.run(
            development_path, catalog_path, sources, tmp_path / "out", "unused", chat=lambda *_: {}
        )


def test_invalid_model_response_is_persisted_before_validation(tmp_path):
    development_path, catalog_path, sources = write_inputs(tmp_path)
    out = tmp_path / "out"
    provider = {"model": "fixture", "message": {"content": json.dumps({"bad": True})}}
    result = R.run(development_path, catalog_path, sources, out, "unused", chat=lambda *_: provider)
    stored = json.loads((out / "ablation-no-observation-provider.json").read_text())
    assert stored == provider
    assert result["no_observation_ablation"]["status"] == "invalid_model_proposal"
    assert result["termination"] == "observed_proposal_invalid_hold"


def test_invalid_ablation_does_not_block_valid_observed_arm(tmp_path):
    development_path, catalog_path, sources = write_inputs(tmp_path)
    calls = 0

    def chat(messages, schema):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"model": "fixture", "message": {"content": json.dumps({"bad": True})}}
        candidate_id = "contact_gated_reference"
        return proposal(candidate_id, schema)

    result = R.run(
        development_path, catalog_path, sources, tmp_path / "out", "unused", chat=chat
    )
    assert result["no_observation_ablation"]["status"] == "invalid_model_proposal"
    assert result["adopted_candidate"] == "contact_gated_reference"
