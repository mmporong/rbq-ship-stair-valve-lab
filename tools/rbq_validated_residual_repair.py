"""Bounded, source-grounded repair loop for the reward residual controller.

The failed controller is excluded before proposal.  Candidate outcomes remain
hidden until selection, and adoption requires strict success on all three public
development cases with zero recorded body collisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RL_TRIAGE_OVERRIDE = os.environ.get("RBQ_TRIAGE_SRC")
RL_TRIAGE_DEFAULT = Path.home() / "rl-triage-agent" / "src"
if RL_TRIAGE_OVERRIDE:
    sys.path.insert(0, str(Path(RL_TRIAGE_OVERRIDE).resolve()))
elif RL_TRIAGE_DEFAULT.is_dir():
    sys.path.insert(0, str(RL_TRIAGE_DEFAULT))

from rl_triage import grounded_recovery as recovery  # noqa: E402

from tools import rbq_residual_decision as base  # noqa: E402

MAX_CANDIDATE_TRIALS = 3
DEVELOPMENT_CASE_COUNT = 3
FAILED_SOURCE_VARIANT = "learned"
FAILED_OPAQUE_ID = "opt-72b8"
Chat = Callable[[list[dict[str, str]], dict], dict]
Evaluator = Callable[[str, str], dict]


def file_sha256(path: Path) -> str:
    return base.file_sha256(path)


def write_json_new(path: Path, value: object) -> None:
    base.write_json_new(path, value)


def visible_repair_menu(mapping: dict, eligible: list[str]) -> list[dict]:
    rows = {row["candidate_id"]: row for row in base.visible_menu(mapping)}
    return [rows[candidate_id] for candidate_id in eligible]


def mechanics_facts(mapping: dict, eligible: list[str]) -> tuple[dict, dict]:
    facts, fact_map = base.mechanics_facts(mapping)
    return (
        {
            fact_id: value
            for fact_id, value in facts.items()
            if any(fact_id in fact_map[candidate_id] for candidate_id in eligible)
        },
        {candidate_id: fact_map[candidate_id] for candidate_id in eligible},
    )


def repair_prompt(
    mapping: dict,
    eligible: list[str],
    attempted: list[str],
    observations: dict[str, object],
) -> str:
    packet = {
        "objective": (
            "Choose one untried control repair most plausibly preserving the completed work "
            "stance while correcting the strict valve failure."
        ),
        "candidate_menu": visible_repair_menu(mapping, eligible),
        "attempted_opaque_ids": attempted,
        "completed_inadequate_opaque_ids": [FAILED_OPAQUE_ID],
        "candidate_order_policy": "fixed_seeded_order_filtered_only_for_ineligible_or_attempted",
        "candidate_outcomes_disclosed_before_selection": False,
        "available_observations": observations,
        "remaining_candidate_trials": MAX_CANDIDATE_TRIALS - len(attempted),
        "adoption_gate": {
            "strict_development_successes_required": DEVELOPMENT_CASE_COUNT,
            "body_collision_count_required": 0,
        },
        "citation_contract": (
            "Return the selected candidate mechanics_fact_id and one available observation "
            "fact ID; use null only when no observations are available."
        ),
    }
    return (
        "Choose exactly one eligible opaque candidate. Return JSON only. Opaque IDs and order "
        "do not encode quality.\n"
        + json.dumps(packet, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    )


def build_protocol(source: Path, evaluation: Path, strict_dir: Path, mapping: dict) -> dict:
    baseline, baseline_sources = base.development_observations(evaluation, strict_dir)
    catalog = [row["opaque_id"] for row in mapping["ordered_candidates"]]
    eligible = recovery.eligible_recovery_candidates(
        catalog, [FAILED_OPAQUE_ID], [], MAX_CANDIDATE_TRIALS
    )
    observed_prompt = repair_prompt(mapping, eligible, [], baseline)
    ablation_prompt = repair_prompt(mapping, eligible, [], {})
    return {
        "schema": "rbq_validated_residual_repair_protocol_v1",
        "source_bindings": {
            "reward_source": {
                "path": source.resolve().as_posix(),
                "sha256": file_sha256(source),
            },
            "baseline_development": baseline_sources,
            "original_agent": base.recovery_source_binding(),
        },
        "model_contract": {
            "model": base.MODEL,
            "temperature": 0,
            "seed": base.MODEL_SEED,
            "num_gpu": 0,
            "format_retry_budget": 0,
            "semantic_retry_budget": 0,
        },
        "recovery_contract": {
            "failed_controller_opaque_id": FAILED_OPAQUE_ID,
            "failed_controller_source_variant": FAILED_SOURCE_VARIANT,
            "failed_controller_status": "completed_inadequate",
            "maximum_unique_new_candidates_per_arm": MAX_CANDIDATE_TRIALS,
            "duplicate_candidates_allowed": False,
            "candidate_order_changes_between_rounds": False,
            "candidate_outcomes_visible_only_after_selection": True,
            "strict_development_successes_required": DEVELOPMENT_CASE_COUNT,
            "body_collision_count_required": 0,
            "heldout_or_private_inputs_allowed": False,
        },
        "comparison_contract": {
            "same_model_menu_order_descriptions_budget_and_loop": True,
            "ablation_hides_only_initial_failed_controller_diagnosis": True,
            "failed_controller_excluded_in_both_arms": True,
            "selected_candidate_receipts_available_in_both_arms": True,
        },
        "initial_eligible_order": eligible,
        "baseline_observations": baseline,
        "initial_prompts": {
            "ablation-no-initial-diagnosis": {
                "prompt": ablation_prompt,
                "prompt_sha256": hashlib.sha256(ablation_prompt.encode()).hexdigest(),
            },
            "observed-initial-diagnosis": {
                "prompt": observed_prompt,
                "prompt_sha256": hashlib.sha256(observed_prompt.encode()).hexdigest(),
            },
        },
    }


def prepare(source: Path, evaluation: Path, strict_dir: Path, out_dir: Path) -> dict:
    if out_dir.exists():
        raise ValueError("validated repair output directory already exists")
    out_dir.mkdir(parents=True)
    mapping = base.candidate_map(source)
    map_path = out_dir / "candidate-map.json"
    write_json_new(map_path, mapping)
    protocol = build_protocol(source, evaluation, strict_dir, mapping)
    protocol["candidate_map"] = {"path": map_path.name, "sha256": file_sha256(map_path)}
    protocol_path = out_dir / "protocol.json"
    write_json_new(protocol_path, protocol)
    return {"candidate_map": map_path, "protocol": protocol_path}


def candidate_development_receipt(
    opaque_id: str,
    source_variant: str,
    evaluation_path: Path,
    strict_dir: Path,
) -> dict:
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    rows = [row for row in evaluation.get("trials", []) if row.get("variant") == source_variant]
    rows.sort(key=lambda row: row.get("case_id", ""))
    if len(rows) != DEVELOPMENT_CASE_COUNT or [row.get("case_id") for row in rows] != [
        "dev-01",
        "dev-02",
        "dev-03",
    ]:
        raise ValueError("candidate needs exactly three public development rows")
    collisions = sum(bool(row.get("body_collision")) for row in rows)
    strict_paths = [
        strict_dir / f"dev-{index:02d}--{source_variant}.json"
        for index in range(1, DEVELOPMENT_CASE_COUNT + 1)
    ]
    if all(path.is_file() for path in strict_paths):
        verdicts = [json.loads(path.read_text(encoding="utf-8")) for path in strict_paths]
        strict_source = "official_strict_development_verdicts"
        case_verdicts = []
        for path, evaluation_row, verdict in zip(strict_paths, rows, verdicts, strict=True):
            identity = verdict.get("trial_identity", {})
            case_id = identity.get("case_id", verdict.get("case_id"))
            variant = identity.get(
                "mode",
                verdict.get("mode", verdict.get("candidate_id", source_variant)),
            )
            if path.name != f"{evaluation_row['case_id']}--{source_variant}.json":
                raise ValueError("strict verdict filename does not bind case and variant")
            if case_id != evaluation_row["case_id"] or variant != source_variant:
                raise ValueError("strict verdict identity does not match development row")
            if verdict.get("state_sha256") != evaluation_row.get("state_sha256"):
                raise ValueError("strict verdict state SHA does not match development row")
            recorded_failure = (
                evaluation_row.get("termination") != "horizon"
                or bool(evaluation_row.get("body_collision"))
                or bool(evaluation_row.get("fell"))
            )
            scored_success = bool(verdict.get("strict_success"))
            if recorded_failure and scored_success:
                raise ValueError(
                    "strict verdict contradicts recorded collision, fall, or termination"
                )
            case_verdicts.append(
                {
                    "case_id": case_id,
                    "strict_success": scored_success and not recorded_failure,
                    "outcome": verdict.get("outcome"),
                    "source_sha256": file_sha256(path),
                    "state_sha256": verdict["state_sha256"],
                }
            )
        strict_successes = sum(row["strict_success"] for row in case_verdicts)
    else:
        case_verdicts = []
        strict_successes = 0
        strict_source = "recorded_early_termination_or_unavailable_strict_verdict"
        for row in rows:
            early = (
                row.get("termination") != "horizon"
                or bool(row.get("body_collision"))
                or bool(row.get("fell"))
            )
            case_verdicts.append(
                {
                    "case_id": row["case_id"],
                    "strict_success": False,
                    "outcome": (
                        "recorded_early_termination"
                        if early
                        else "strict_verdict_unavailable"
                    ),
                    "termination": row.get("termination"),
                }
            )
    measurement = {
        "case_count": DEVELOPMENT_CASE_COUNT,
        "strict_success_count": strict_successes,
        "body_collision_count": collisions,
        "work_stance_success_count": sum(bool(row.get("work_stance_reached")) for row in rows),
        "strict_source": strict_source,
        "case_verdicts": case_verdicts,
    }
    return {
        "schema": "rbq_validated_residual_candidate_receipt_v1",
        "opaque_id": opaque_id,
        "source_variant": source_variant,
        "development_source": {
            "path": evaluation_path.resolve().as_posix(),
            "sha256": file_sha256(evaluation_path),
        },
        "measurement": measurement,
        "receipt_sha256": base.canonical_sha256(measurement),
    }


def receipt_facts(receipt: dict) -> dict[str, object]:
    prefix = f"receipt:{receipt['opaque_id']}"
    measurement = receipt["measurement"]
    return {
        f"{prefix}:strict_success_count": measurement["strict_success_count"],
        f"{prefix}:body_collision_count": measurement["body_collision_count"],
        f"{prefix}:work_stance_success_count": measurement["work_stance_success_count"],
        f"{prefix}:strict_source": measurement["strict_source"],
    }


def run(
    source: Path,
    evaluation: Path,
    strict_dir: Path,
    out_dir: Path,
    endpoint: str,
    chat: Chat | None = None,
    evaluator: Evaluator | None = None,
) -> dict:
    map_path = out_dir / "candidate-map.json"
    protocol_path = out_dir / "protocol.json"
    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if mapping != base.candidate_map(source):
        raise ValueError("precommitted candidate map changed")
    expected_protocol = build_protocol(source, evaluation, strict_dir, mapping)
    expected_protocol["candidate_map"] = {"path": map_path.name, "sha256": file_sha256(map_path)}
    if protocol != expected_protocol:
        raise ValueError("precommitted validated repair protocol changed")
    call = chat or (lambda messages, schema: base.ollama_chat(endpoint, messages, schema))
    evaluate = evaluator or (
        lambda opaque_id, variant: candidate_development_receipt(
            opaque_id, variant, evaluation, strict_dir
        )
    )
    catalog = [row["opaque_id"] for row in mapping["ordered_candidates"]]
    variant_by_id = {
        row["opaque_id"]: row["source_variant"] for row in mapping["ordered_candidates"]
    }

    def execute_arm(label: str, initial_observations: dict[str, object]) -> dict:
        attempted: list[str] = []
        accumulated_observations = dict(initial_observations)
        trace = []
        adopted = None
        termination = "candidate_budget_exhausted"
        for index in range(MAX_CANDIDATE_TRIALS):
            eligible = recovery.eligible_recovery_candidates(
                catalog, [FAILED_OPAQUE_ID], attempted, MAX_CANDIDATE_TRIALS
            )
            mechanics, candidate_fact_ids = mechanics_facts(mapping, eligible)
            facts = {**mechanics, **accumulated_observations}
            observation_ids = list(accumulated_observations)
            prompt = repair_prompt(mapping, eligible, attempted, accumulated_observations)
            step_label = f"{label}-step-{index:02d}"
            base_record = {
                "label": step_label,
                "eligible_order": eligible,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
            try:
                provider = call(
                    [{"role": "user", "content": prompt}],
                    recovery.typed_proposal_schema(
                        eligible, facts, candidate_fact_ids, observation_ids
                    ),
                )
            except Exception as exc:
                failure_path = out_dir / f"{step_label}-provider-failure.json"
                write_json_new(
                    failure_path,
                    {"exception_type": type(exc).__name__, "exception_message": str(exc)},
                )
                trace.append(
                    {
                        **base_record,
                        "status": "provider_failure",
                        "failure_path": failure_path.name,
                        "failure_sha256": file_sha256(failure_path),
                    }
                )
                termination = "provider_failure_hold"
                break
            raw_path = out_dir / f"{step_label}-provider.json"
            write_json_new(raw_path, provider)
            record = {
                **base_record,
                "provider_path": raw_path.name,
                "provider_sha256": file_sha256(raw_path),
            }
            try:
                proposal = recovery.bind_typed_proposal(
                    base.provider_decision(provider),
                    facts,
                    eligible,
                    candidate_fact_ids,
                    observation_ids,
                )
            except Exception as exc:
                trace.append(
                    {
                        **record,
                        "status": "invalid_model_proposal",
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    }
                )
                termination = "invalid_model_proposal_hold"
                break
            candidate_id = proposal["candidate_id"]
            variant = variant_by_id[candidate_id]
            receipt = evaluate(candidate_id, variant)
            receipt_path = out_dir / f"{step_label}-development-receipt.json"
            write_json_new(receipt_path, receipt)
            measurement = receipt["measurement"]
            state = recovery.advance_validated_recovery(
                attempted,
                candidate_id,
                eligible,
                max_attempts=MAX_CANDIDATE_TRIALS,
                development_case_count=DEVELOPMENT_CASE_COUNT,
                strict_success_count=measurement["strict_success_count"],
                body_collision_count=measurement["body_collision_count"],
            )
            attempted = state["attempted"]
            trace.append(
                {
                    **record,
                    "status": "valid",
                    "proposal": proposal,
                    "host_resolution": {
                        "opaque_id": candidate_id,
                        "source_variant": variant,
                    },
                    "development_receipt_path": receipt_path.name,
                    "development_receipt_sha256": file_sha256(receipt_path),
                    "host_state": state,
                }
            )
            if state["status"] == "adopted":
                adopted = {"opaque_id": candidate_id, "source_variant": variant}
                termination = "strict_development_adoption"
                break
            accumulated_observations.update(receipt_facts(receipt))
        return {
            "label": label,
            "attempted_opaque_ids": attempted,
            "attempted_source_variants": [variant_by_id[item] for item in attempted],
            "adopted": adopted,
            "termination": termination,
            "trace": trace,
        }

    ablation = execute_arm("ablation-no-initial-diagnosis", {})
    observed = execute_arm(
        "observed-initial-diagnosis", protocol["baseline_observations"]
    )
    summary = {
        "schema": "rbq_validated_residual_repair_result_v1",
        "model": base.MODEL,
        "protocol": {"path": protocol_path.name, "sha256": file_sha256(protocol_path)},
        "candidate_map": {"path": map_path.name, "sha256": file_sha256(map_path)},
        "original_agent": base.recovery_source_binding(),
        "arms": {"ablation": ablation, "observed": observed},
        "adoption": {
            "ablation_variant": (
                ablation["adopted"]["source_variant"] if ablation["adopted"] else None
            ),
            "observed_variant": (
                observed["adopted"]["source_variant"] if observed["adopted"] else None
            ),
        },
        "retry_performed": False,
        "heldout_or_private_inputs_consumed": False,
        "efficacy_claim_supported": False,
        "efficacy_note": "Development repair adoption is not sealed-cohort efficacy evidence.",
    }
    write_json_new(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for name in ("prepare", "run"):
        command = sub.add_parser(name)
        command.add_argument("source", type=Path)
        command.add_argument("development_evaluation", type=Path)
        command.add_argument("development_strict_dir", type=Path)
        command.add_argument("output_dir", type=Path)
        if name == "run":
            command.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare(
            args.source,
            args.development_evaluation,
            args.development_strict_dir,
            args.output_dir,
        )
    else:
        run(
            args.source,
            args.development_evaluation,
            args.development_strict_dir,
            args.output_dir,
            args.endpoint,
        )


if __name__ == "__main__":
    main()
