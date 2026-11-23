from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import rbq_residual_decision as decision

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "train" / "rbq_reward_residual.py"
EVALUATION = (
    ROOT / "evidence" / "rbq-agent-body-20261122" / "reward-policy" / "development-evaluation.json"
)
STRICT = (
    ROOT / "evidence" / "rbq-agent-body-20261122" / "reward-policy" / "development-strict"
)


def provider(candidate: str, mechanics: str, observation: str | None) -> dict:
    return {
        "model": decision.MODEL,
        "message": {
            "role": "assistant",
            "content": json.dumps(
                {
                    "candidate_id": candidate,
                    "mechanics_fact_id": mechanics,
                    "observation_fact_id": observation,
                    "reason": (
                        "The selected control structure targets the measured residual failure."
                    ),
                }
            ),
        },
        "done": True,
    }


def test_candidate_map_is_opaque_complete_and_seeded():
    mapping = decision.candidate_map(SOURCE)
    rows = mapping["ordered_candidates"]
    assert mapping["map_seed"] == 2026112202
    assert [row["opaque_id"] for row in rows] == [
        "opt-844b",
        "opt-72b8",
        "opt-3013",
        "opt-2a5e",
        "opt-6cb8",
        "opt-01b7",
    ]
    assert {row["source_variant"] for row in rows} == set(decision.EXPECTED_VARIANTS)
    menu_text = json.dumps(decision.visible_menu(mapping))
    assert all(variant not in menu_text for variant in decision.EXPECTED_VARIANTS)


def test_observations_only_consume_learned_public_development():
    facts, sources = decision.development_observations(EVALUATION, STRICT)
    assert sources["evaluation"]["rows_consumed"] == [
        "dev-01/learned",
        "dev-02/learned",
        "dev-03/learned",
    ]
    assert facts["observation:development:case_count"] == 3
    assert facts["observation:development:work_stance_success_count"] == 3
    assert facts["observation:development:strict_success_count"] == 0
    assert facts["observation:development:hold_min_angle_deg"] == pytest.approx(89.74049400800124)
    assert facts["observation:development:nonzero_ppo_torque_case_count"] == 3


def test_prepare_freezes_same_menu_and_only_observations_differ(tmp_path):
    out = tmp_path / "decision"
    decision.prepare(SOURCE, EVALUATION, STRICT, out)
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    ablation = protocol["arms"]["ablation-no-observation"]["prompt"]
    observed = protocol["arms"]["observed-development"]["prompt"]
    assert protocol["fairness_contract"]["same_candidate_menu_order_descriptions"] is True
    assert '"development_observations":{}' in ablation
    assert '"development_observations":{' in observed
    for row in protocol["menu"]:
        encoded = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        assert encoded in ablation
        assert encoded in observed


def test_run_uses_original_typed_binding_and_records_both_raw_responses(tmp_path):
    out = tmp_path / "decision"
    decision.prepare(SOURCE, EVALUATION, STRICT, out)
    calls = []

    def chat(messages, schema):
        calls.append((messages, schema))
        if len(calls) == 1:
            return provider("opt-01b7", "mechanics:opt-01b7:control_law", None)
        return provider(
            "opt-01b7",
            "mechanics:opt-01b7:control_law",
            "observation:development:nonzero_ppo_torque_case_count",
        )

    summary = decision.run(SOURCE, EVALUATION, STRICT, out, "unused", chat=chat)
    assert len(calls) == 2
    assert summary["ablation"]["status"] == "valid"
    assert summary["observed"]["status"] == "valid"
    assert summary["observed"]["host_resolution"]["source_variant"] == "learned_no_ppo"
    assert (out / "ablation-no-observation-provider.json").is_file()
    assert (out / "observed-development-provider.json").is_file()
    assert summary["efficacy_claim_supported"] is False


def test_invalid_ablation_does_not_block_observed_arm(tmp_path):
    out = tmp_path / "decision"
    decision.prepare(SOURCE, EVALUATION, STRICT, out)
    calls = 0

    def chat(messages, schema):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"model": decision.MODEL, "message": {"content": "not-json"}}
        return provider(
            "opt-844b",
            "mechanics:opt-844b:control_law",
            "observation:development:strict_success_count",
        )

    summary = decision.run(SOURCE, EVALUATION, STRICT, out, "unused", chat=chat)
    assert calls == 2
    assert summary["ablation"]["status"] == "invalid_model_proposal"
    assert summary["observed"]["status"] == "valid"
    assert summary["retry_performed"] is False


def test_run_rejects_changed_precommit_protocol(tmp_path):
    out = tmp_path / "decision"
    decision.prepare(SOURCE, EVALUATION, STRICT, out)
    protocol_path = out / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["model_contract"]["temperature"] = 1
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="precommitted protocol"):
        decision.run(SOURCE, EVALUATION, STRICT, out, "unused", chat=lambda *_: {})
