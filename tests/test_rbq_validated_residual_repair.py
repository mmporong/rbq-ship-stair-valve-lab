from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import rbq_validated_residual_repair as repair

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
        "model": repair.base.MODEL,
        "message": {
            "role": "assistant",
            "content": json.dumps(
                {
                    "candidate_id": candidate,
                    "mechanics_fact_id": mechanics,
                    "observation_fact_id": observation,
                    "reason": "bounded repair proposal",
                }
            ),
        },
        "done": True,
    }


def receipt(candidate: str, variant: str, strict: int, collisions: int) -> dict:
    measurement = {
        "case_count": 3,
        "strict_success_count": strict,
        "body_collision_count": collisions,
        "work_stance_success_count": 3,
        "strict_source": "test",
        "case_verdicts": [],
    }
    return {
        "schema": "test",
        "opaque_id": candidate,
        "source_variant": variant,
        "measurement": measurement,
        "receipt_sha256": "0" * 64,
    }


def test_prepare_excludes_failed_controller_and_freezes_comparison(tmp_path):
    out = tmp_path / "validated"
    repair.prepare(SOURCE, EVALUATION, STRICT, out)
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    assert repair.FAILED_OPAQUE_ID not in protocol["initial_eligible_order"]
    assert protocol["recovery_contract"]["maximum_unique_new_candidates_per_arm"] == 3
    assert protocol["recovery_contract"]["candidate_order_changes_between_rounds"] is False
    assert protocol["comparison_contract"][
        "ablation_hides_only_initial_failed_controller_diagnosis"
    ]
    observed = protocol["initial_prompts"]["observed-initial-diagnosis"]["prompt"]
    ablation = protocol["initial_prompts"]["ablation-no-initial-diagnosis"]["prompt"]
    observed_menu = json.loads(observed.split("\n", 1)[1])["candidate_menu"]
    assert repair.FAILED_OPAQUE_ID not in [row["candidate_id"] for row in observed_menu]
    assert '"available_observations":{}' in ablation


def test_loop_never_repeats_and_adopts_only_strict_three_of_three(tmp_path):
    out = tmp_path / "validated"
    repair.prepare(SOURCE, EVALUATION, STRICT, out)
    calls = {"ablation": 0, "observed": 0}
    choices = ["opt-6cb8", "opt-01b7"]

    def chat(messages, schema):
        packet = json.loads(messages[0]["content"].split("\n", 1)[1])
        arm = "observed" if packet["available_observations"].get(
            "observation:development:strict_success_count"
        ) is not None else "ablation"
        index = calls[arm]
        calls[arm] += 1
        candidate = choices[index]
        mechanics = f"mechanics:{candidate}:control_law"
        observations = list(packet["available_observations"])
        observation = observations[0] if observations else None
        return provider(candidate, mechanics, observation)

    def evaluate(candidate, variant):
        return receipt(candidate, variant, 3 if candidate == "opt-01b7" else 2, 0)

    summary = repair.run(SOURCE, EVALUATION, STRICT, out, "unused", chat, evaluate)
    for arm in summary["arms"].values():
        assert arm["attempted_opaque_ids"] == choices
        assert len(set(arm["attempted_opaque_ids"])) == 2
        assert arm["adopted"]["source_variant"] == "learned_no_ppo"
    assert summary["adoption"] == {
        "ablation_variant": "learned_no_ppo",
        "observed_variant": "learned_no_ppo",
    }


def test_invalid_first_arm_does_not_block_second_arm(tmp_path):
    out = tmp_path / "validated"
    repair.prepare(SOURCE, EVALUATION, STRICT, out)
    calls = 0

    def chat(messages, schema):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"model": repair.base.MODEL, "message": {"content": "invalid"}}
        packet = json.loads(messages[0]["content"].split("\n", 1)[1])
        observations = list(packet["available_observations"])
        return provider(
            "opt-01b7",
            "mechanics:opt-01b7:control_law",
            observations[0],
        )

    summary = repair.run(
        SOURCE,
        EVALUATION,
        STRICT,
        out,
        "unused",
        chat,
        lambda candidate, variant: receipt(candidate, variant, 3, 0),
    )
    assert summary["arms"]["ablation"]["termination"] == "invalid_model_proposal_hold"
    assert summary["arms"]["observed"]["adopted"]["source_variant"] == "learned_no_ppo"
    assert summary["retry_performed"] is False


def test_early_termination_without_official_strict_is_failure(tmp_path):
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "trials": [
                    {
                        "case_id": f"dev-{index:02d}",
                        "variant": "candidate",
                        "termination": "fall",
                        "body_collision": False,
                        "fell": True,
                        "work_stance_reached": False,
                    }
                    for index in range(1, 4)
                ]
            }
        ),
        encoding="utf-8",
    )
    strict = tmp_path / "strict"
    strict.mkdir()
    result = repair.candidate_development_receipt(
        "opaque", "candidate", evaluation, strict
    )
    assert result["measurement"]["strict_success_count"] == 0
    assert all(
        row["outcome"] == "recorded_early_termination"
        for row in result["measurement"]["case_verdicts"]
    )


def write_development_fixture(tmp_path, variant="candidate"):
    evaluation = tmp_path / "evaluation.json"
    rows = [
        {
            "case_id": f"dev-{index:02d}",
            "variant": variant,
            "termination": "horizon",
            "body_collision": False,
            "fell": False,
            "work_stance_reached": True,
            "state_sha256": str(index) * 64,
        }
        for index in range(1, 4)
    ]
    evaluation.write_text(json.dumps({"trials": rows}), encoding="utf-8")
    strict = tmp_path / "strict"
    strict.mkdir()
    return evaluation, strict, rows


def test_official_strict_supports_both_identity_schemas_and_binds_state(tmp_path):
    evaluation, strict, rows = write_development_fixture(tmp_path)
    for index, row in enumerate(rows, 1):
        verdict = {
            "strict_success": True,
            "outcome": "physical_passed",
            "state_sha256": row["state_sha256"],
        }
        if index == 1:
            verdict.update(case_id=row["case_id"], candidate_id="candidate")
        else:
            verdict["trial_identity"] = {
                "case_id": row["case_id"],
                "mode": "candidate",
            }
        (strict / f"{row['case_id']}--candidate.json").write_text(
            json.dumps(verdict), encoding="utf-8"
        )
    result = repair.candidate_development_receipt(
        "opaque", "candidate", evaluation, strict
    )
    assert result["measurement"]["strict_success_count"] == 3
    assert all(
        row["state_sha256"] == source["state_sha256"]
        for row, source in zip(result["measurement"]["case_verdicts"], rows, strict=True)
    )


@pytest.mark.parametrize("field", ["case_id", "candidate_id", "state_sha256"])
def test_official_strict_rejects_mismatched_identity_or_state(tmp_path, field):
    evaluation, strict, rows = write_development_fixture(tmp_path)
    for row in rows:
        verdict = {
            "case_id": row["case_id"],
            "candidate_id": "candidate",
            "state_sha256": row["state_sha256"],
            "strict_success": True,
        }
        if row["case_id"] == "dev-02":
            verdict[field] = "wrong"
        (strict / f"{row['case_id']}--candidate.json").write_text(
            json.dumps(verdict), encoding="utf-8"
        )
    with pytest.raises(ValueError, match="strict verdict"):
        repair.candidate_development_receipt("opaque", "candidate", evaluation, strict)


def test_collision_or_fall_cannot_be_strict_success(tmp_path):
    evaluation, strict, rows = write_development_fixture(tmp_path)
    rows[0]["termination"] = "fall"
    rows[0]["fell"] = True
    evaluation.write_text(json.dumps({"trials": rows}), encoding="utf-8")
    for row in rows:
        (strict / f"{row['case_id']}--candidate.json").write_text(
            json.dumps(
                {
                    "case_id": row["case_id"],
                    "candidate_id": "candidate",
                    "state_sha256": row["state_sha256"],
                    "strict_success": True,
                }
            ),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="contradicts recorded"):
        repair.candidate_development_receipt("opaque", "candidate", evaluation, strict)


def test_protocol_change_is_rejected(tmp_path):
    out = tmp_path / "validated"
    repair.prepare(SOURCE, EVALUATION, STRICT, out)
    protocol_path = out / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["recovery_contract"]["maximum_unique_new_candidates_per_arm"] = 4
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol changed"):
        repair.run(SOURCE, EVALUATION, STRICT, out, "unused", lambda *_: {})
