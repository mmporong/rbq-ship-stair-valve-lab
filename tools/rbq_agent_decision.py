"""Run a source-bound RBQ development diagnosis with the original agent contract.

This adapter uses only public development telemetry.  It never reads a holdout,
does not grant execution authority, and does not register RBQ as an official
``rl-triage-agent`` task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

RL_TRIAGE = Path.home() / "rl-triage-agent" / "src"
sys.path.insert(0, str(RL_TRIAGE))

from rl_triage import active_diagnosis as diagnosis  # noqa: E402
from rl_triage import probe_loop  # noqa: E402

Chat = Callable[[list[dict[str, str]], dict], dict]
PUBLIC_PROBE = "P_episode"
MODEL = "gemma3:4b"
FORMAT_RETRY_BUDGET = 0


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def validate_development(development: dict) -> dict:
    valid_host = (
        isinstance(development, dict)
        and development.get("host_contract") == "rbq_stair_intervention_v1"
    )
    if not valid_host:
        raise ValueError("development input is not an RBQ host result")
    if (
        development.get("registered_nvidia_task") is not False
        or development.get("cpu_only") is not True
    ):
        raise ValueError("development result must preserve the independent CPU host boundary")
    groups = development.get("groups")
    if not isinstance(groups, dict) or "no_intervention" not in groups:
        raise ValueError("development result needs a no_intervention group")
    for candidate_id, group in groups.items():
        if not isinstance(candidate_id, str) or not isinstance(group, dict):
            raise ValueError("candidate groups are malformed")
        summary, trials = group.get("summary"), group.get("trials")
        if not isinstance(summary, dict) or not isinstance(trials, list) or not trials:
            raise ValueError(f"{candidate_id} needs a summary and trials")
        if int(_finite(summary.get("n"), f"{candidate_id}.n")) != len(trials):
            raise ValueError(f"{candidate_id} trial count mismatch")
        for field in (
            "work_stance_rate",
            "mean_highest_supported_tread",
            "body_collision_rate",
            "mean_maximum_forward_x_m",
            "mean_maximum_abs_pitch_deg",
        ):
            _finite(summary.get(field), f"{candidate_id}.{field}")
        for trial in trials:
            if not isinstance(trial, dict):
                raise ValueError(f"{candidate_id} trial is malformed")
            _finite(trial.get("simulated_s"), f"{candidate_id}.simulated_s")
    return development


def validate_catalog(catalog: dict, development: dict) -> dict:
    if not isinstance(catalog, dict) or set(catalog) != {"schema", "candidates"}:
        raise ValueError("catalog fields must be schema and candidates")
    if catalog["schema"] != "rbq_intervention_catalog_v1":
        raise ValueError("unknown intervention catalog schema")
    candidates = catalog["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("catalog needs at least one candidate")
    seen = set()
    clean = []
    for row in candidates:
        expected = {"candidate_id", "description", "executable", "mechanics"}
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError(
                "candidate fields must be candidate_id, description, executable and mechanics"
            )
        candidate_id = row["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            raise ValueError("candidate IDs must be unique non-empty strings")
        if not isinstance(row["description"], str) or not row["description"].strip():
            raise ValueError("candidate description must be non-empty")
        if row["executable"] is not True:
            raise ValueError(f"candidate {candidate_id!r} is not executable")
        mechanics = row["mechanics"]
        if not isinstance(mechanics, dict) or not mechanics:
            raise ValueError("candidate mechanics must be a non-empty object")
        forbidden = ("success", "outcome", "score", "rate", "trial", "performance", "collision")
        for key, value in mechanics.items():
            if (
                not isinstance(key, str)
                or not key
                or any(term in key.lower() for term in forbidden)
            ):
                raise ValueError("candidate mechanics contain an outcome-like field")
            if not isinstance(value, (str, bool, int, float)) or (
                isinstance(value, float) and not math.isfinite(value)
            ):
                raise ValueError("candidate mechanics values must be finite JSON scalars")
        seen.add(candidate_id)
        clean.append(dict(row))
    return {"schema": catalog["schema"], "candidates": clean}


def baseline_facts(development: dict, source_sha256: str) -> dict[str, object]:
    baseline = development["groups"]["no_intervention"]
    summary, trials = baseline["summary"], baseline["trials"]
    return {
        "source:development_sha256": source_sha256,
        "host:registered_nvidia_task": False,
        "host:cpu_only": True,
        "host:checkpoint": development["checkpoint"],
        "baseline:trial_count": len(trials),
        "baseline:work_stance_rate": summary["work_stance_rate"],
        "baseline:body_collision_rate": summary["body_collision_rate"],
        "baseline:premature_termination_count": sum(
            trial.get("termination") != "phase_timeout" for trial in trials
        ),
    }


def episode_receipt(development: dict, horizon_s: float = 19.96) -> dict:
    baseline = development["groups"]["no_intervention"]
    trials = baseline["trials"]
    timeout_trials = [trial for trial in trials if trial.get("termination") == "phase_timeout"]
    ratios = [_finite(trial["simulated_s"], "simulated_s") / horizon_s for trial in timeout_trials]
    timeout_ratio = sum(ratios) / len(ratios) if ratios else None
    outcome = (
        probe_loop.classify(PUBLIC_PROBE, {"timeout_ratio": timeout_ratio}, None)
        if timeout_ratio is not None
        else "unknown"
    )
    return {
        "request_id": "rbq_development_episode",
        "probe": PUBLIC_PROBE,
        "outcome": outcome,
        "exit_code": 0,
        "over_budget": False,
        "measurement": {
            "timeout_trial_count": len(timeout_trials),
            "timeout_ratio": timeout_ratio,
            "configured_horizon_s": horizon_s,
            "classification_scope": "time_limit_terminations_only",
        },
    }


def rbq_host_receipt(development: dict) -> dict:
    """Return project-native failure evidence without reclassifying it as an original probe."""
    baseline = development["groups"]["no_intervention"]
    trials = baseline["trials"]
    terminations: dict[str, int] = {}
    for trial in trials:
        termination = str(trial.get("termination"))
        terminations[termination] = terminations.get(termination, 0) + 1
    return {
        "request_id": "rbq_host_development_failure",
        "probe": "rbq_work_stance_and_contact_outcomes",
        "outcome": "abnormal",
        "exit_code": 0,
        "over_budget": False,
        "measurement": {
            "work_stance_rate": baseline["summary"]["work_stance_rate"],
            "body_collision_rate": baseline["summary"]["body_collision_rate"],
            "early_safety_termination_fraction": sum(
                trial.get("termination") != "phase_timeout" for trial in trials
            )
            / len(trials),
            "trial_count": len(trials),
            "termination_counts": terminations,
        },
    }


def diagnosis_proposal_schema() -> dict:
    mechanisms = list(probe_loop.MECHANISMS)
    scalar = {"type": ["string", "number", "boolean"]}
    return {
        "type": "object",
        "required": ["ranking", "reason", "claims"],
        "properties": {
            "ranking": {
                "type": "array",
                "items": {"type": "string", "enum": mechanisms},
                "minItems": len(mechanisms),
                "maxItems": len(mechanisms),
                "uniqueItems": True,
            },
            "reason": {"type": "string", "minLength": 1},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "value"],
                    "properties": {"id": {"type": "string"}, "value": scalar},
                    "additionalProperties": False,
                },
                "minItems": 1,
            },
        },
        "additionalProperties": False,
    }


def bind_diagnosis(proposal: object, packet: dict, phase: str) -> dict:
    if not isinstance(proposal, dict) or set(proposal) != {"ranking", "reason", "claims"}:
        raise ValueError("diagnosis proposal fields do not match the adapter contract")
    ranking = proposal["ranking"]
    if phase == "initial":
        action, probe = "continue", PUBLIC_PROBE
        expected = {
            mechanism: probe_loop.PROBES[probe]["expect"][mechanism] for mechanism in ranking[:3]
        }
    elif phase == "after_observation":
        state = diagnosis.evidence_state(packet["observed"], packet["budget_left"])
        confirm = state["status"] == "confirmed" and state["hypotheses"] == [ranking[0]]
        action, probe, expected = ("confirm" if confirm else "hold"), None, {}
    elif phase == "without_observation_ablation":
        action, probe, expected = "hold", None, {}
    else:
        raise ValueError("unknown diagnosis binding phase")
    decision = {
        "ranking": ranking,
        "probe": probe,
        "action": action,
        "reason": proposal["reason"],
        "claims": proposal["claims"],
        "expected_outcomes": expected,
    }
    return diagnosis.validate_decision(decision, packet)


def selection_facts(catalog: dict, evidence: dict[str, object]) -> dict[str, object]:
    facts = dict(evidence)
    for row in catalog["candidates"]:
        candidate_id = row["candidate_id"]
        for field, value in row["mechanics"].items():
            facts[f"candidate:{candidate_id}:mechanics.{field}"] = value
    return facts


def selection_schema(candidate_ids: list[str], fact_ids: list[str]) -> dict:
    return {
        "type": "object",
        "required": ["candidate_id", "claims", "reason"],
        "properties": {
            "candidate_id": {"type": "string", "enum": candidate_ids},
            "claims": {
                "type": "array",
                "items": {"type": "string", "enum": fact_ids},
                "minItems": 2,
                "uniqueItems": True,
            },
            "reason": {"type": "string", "minLength": 1},
        },
        "additionalProperties": False,
    }


def validate_selection(decision: object, candidate_ids: list[str], facts: dict) -> dict:
    if not isinstance(decision, dict) or set(decision) != {"candidate_id", "claims", "reason"}:
        raise ValueError("selection response fields do not match the contract")
    if decision["candidate_id"] not in candidate_ids:
        raise ValueError("selected candidate is outside the executable catalog")
    claims = decision["claims"]
    if (
        not isinstance(claims, list)
        or len(claims) < 2
        or len(claims) != len(set(claims))
        or any(claim not in facts for claim in claims)
    ):
        raise ValueError("selection claims are not exact development fact IDs")
    candidate_prefix = f"candidate:{decision['candidate_id']}:mechanics."
    if not any(claim.startswith(candidate_prefix) for claim in claims):
        raise ValueError("selection must cite the chosen candidate mechanics")
    if any(key.startswith("receipt:") for key in facts) and not any(
        claim.startswith("receipt:") for claim in claims
    ):
        raise ValueError("observed selection must cite the source-bound receipt")
    reason = decision["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("selection reason must be non-empty")
    return {
        "candidate_id": decision["candidate_id"],
        "claims": [{"id": claim, "value": facts[claim]} for claim in claims],
        "reason": reason.strip(),
    }


def ollama_chat(endpoint: str, messages: list[dict[str, str]], schema: dict) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "format": schema,
        "options": {"temperature": 0, "seed": 42, "num_gpu": 0, "num_predict": 1024},
        "keep_alive": 0,
    }
    request = Request(
        endpoint,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        provider = json.loads(response.read().decode())
    return {
        "model": provider.get("model"),
        "decision": json.loads(provider["message"]["content"]),
    }


def selection_prompt(packet: dict) -> str:
    return (
        "Choose one executable RBQ intervention using objective_order lexicographically. "
        "No candidate outcome is disclosed. Cite at least two exact source_bound_facts IDs, "
        "including a chosen-candidate mechanics fact and every available runtime observation "
        "needed for the choice. Output JSON only.\n"
        + json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def run(
    development_path: Path,
    catalog_path: Path,
    endpoint: str,
    chat: Chat | None = None,
) -> dict:
    development = validate_development(json.loads(development_path.read_text(encoding="utf-8")))
    catalog = validate_catalog(json.loads(catalog_path.read_text(encoding="utf-8")), development)
    source_sha = file_sha256(development_path)
    catalog_sha = file_sha256(catalog_path)
    facts = baseline_facts(development, source_sha)
    receipt = episode_receipt(development)
    host_receipt = rbq_host_receipt(development)

    def ask(messages: list[dict[str, str]], schema: dict) -> dict:
        return (chat or (lambda m, s: ollama_chat(endpoint, m, s)))(messages, schema)

    initial_packet = {
        "phase": "initial",
        "facts": facts,
        "observed": [],
        "budget_left": 1,
        "execution_budget_s": 0,
    }
    proposal_system = (
        diagnosis.SYSTEM_PROMPT
        + " The host binds action, probe, and expected outcomes. Return only ranking, reason, "
        "and claims."
    )
    initial_messages = [
        {"role": "system", "content": proposal_system},
        {
            "role": "user",
            "content": diagnosis.build_prompt(initial_packet)
            + "\nRank mechanisms from the supplied baseline facts. The only executable "
            "source-bound probe is P_episode.",
        },
    ]
    before_raw = ask(initial_messages, diagnosis_proposal_schema())
    before = bind_diagnosis(before_raw["decision"], initial_packet, "initial")

    after_packet = {
        "phase": "after_observation",
        "facts": facts,
        "observed": [receipt],
        "budget_left": 0,
        "execution_budget_s": 0,
        "previous_decision": before,
    }
    after_messages = [
        {"role": "system", "content": proposal_system},
        {
            "role": "user",
            "content": diagnosis.build_prompt(after_packet)
            + "\nApply the frozen evidence-state rule. Cite at least one exact latest receipt "
            "fact and rank the sole surviving mechanism first when confirmed.",
        },
    ]
    after_raw = ask(after_messages, diagnosis_proposal_schema())
    after = bind_diagnosis(after_raw["decision"], after_packet, "after_observation")

    ablated_packet = {
        "phase": "without_observation_ablation",
        "facts": facts,
        "observed": [],
        "budget_left": 0,
        "execution_budget_s": 0,
        "previous_decision": before,
    }
    ablated_messages = [
        {"role": "system", "content": proposal_system},
        {
            "role": "user",
            "content": diagnosis.build_prompt(ablated_packet)
            + "\nThe observation is absent and budget is zero. Rank from available facts only.",
        },
    ]
    ablated_raw = ask(ablated_messages, diagnosis_proposal_schema())
    ablated = bind_diagnosis(
        ablated_raw["decision"], ablated_packet, "without_observation_ablation"
    )
    comparison = diagnosis.contrast(before, after, ablated, [receipt])

    candidate_ids = [row["candidate_id"] for row in catalog["candidates"]]
    selector_objectives = [
        "choose mechanics causally applicable to the supplied failure evidence",
        "prefer the smallest intervention justified by supplied evidence",
        "choose no_intervention when no change is justified",
    ]
    candidate_facts = selection_facts(
        catalog,
        {
            **facts,
            **diagnosis.receipt_facts(host_receipt),
        },
    )
    selection_packet = {
        "objective_order": selector_objectives,
        "diagnosis": after,
        "rbq_host_receipt_facts": diagnosis.receipt_facts(host_receipt),
        "executable_candidates": catalog["candidates"],
        "source_bound_facts": candidate_facts,
        "candidate_outcomes_disclosed": False,
        "holdout_disclosed": False,
    }
    observed_selection_prompt = selection_prompt(selection_packet)
    selected_raw = ask(
        [{"role": "user", "content": observed_selection_prompt}],
        selection_schema(candidate_ids, sorted(candidate_facts)),
    )
    selected = validate_selection(selected_raw["decision"], candidate_ids, candidate_facts)
    ablated_selection_facts = selection_facts(catalog, facts)
    ablated_selection_packet = {
        "objective_order": selector_objectives,
        "diagnosis": ablated,
        "rbq_host_receipt_facts": {},
        "executable_candidates": catalog["candidates"],
        "source_bound_facts": ablated_selection_facts,
        "candidate_outcomes_disclosed": False,
        "holdout_disclosed": False,
    }
    without_observation_prompt = selection_prompt(ablated_selection_packet)
    ablated_selected_raw = ask(
        [{"role": "user", "content": without_observation_prompt}],
        selection_schema(candidate_ids, sorted(ablated_selection_facts)),
    )
    ablated_selected = validate_selection(
        ablated_selected_raw["decision"], candidate_ids, ablated_selection_facts
    )
    return {
        "schema": "rbq_source_grounded_agent_decision_v1",
        "boundary": {
            "registered_nvidia_task": False,
            "execution_authority": False,
            "development_only": True,
            "holdout_disclosed": False,
            "cpu_only": True,
            "format_retry_budget": FORMAT_RETRY_BUDGET,
            "semantic_choice_retries": 0,
        },
        "source_bindings": {
            "development_path": development_path.as_posix(),
            "development_sha256": source_sha,
            "catalog_path": catalog_path.as_posix(),
            "catalog_sha256": catalog_sha,
            "checkpoint_sha256": development["checkpoint_sha256"],
            "original_agent_source_root": RL_TRIAGE.as_posix(),
        },
        "original_agent_contracts": [
            "rl_triage.active_diagnosis.decision_messages",
            "rl_triage.active_diagnosis.validate_decision",
            "rl_triage.active_diagnosis.contrast",
            "rl_triage.active_diagnosis.receipt_facts",
        ],
        "original_probe_observation": receipt,
        "rbq_host_observation": {
            "contract": "go2_wall.rbq_development_failure.v1",
            "registered_nvidia_probe": False,
            "receipt": host_receipt,
        },
        "evidence_state": diagnosis.evidence_state([receipt], 0),
        "model": MODEL,
        "model_runs": {
            "before": {"provider_model": before_raw.get("model"), "decision": before},
            "after": {"provider_model": after_raw.get("model"), "decision": after},
            "without_observation": {
                "provider_model": ablated_raw.get("model"),
                "decision": ablated,
            },
            "intervention_selection": {
                "provider_model": selected_raw.get("model"),
                "decision": selected,
            },
            "without_observation_selection": {
                "provider_model": ablated_selected_raw.get("model"),
                "decision": ablated_selected,
            },
        },
        "contrast": comparison,
        "selection_contrast": {
            "candidate_changed_with_observation": (
                selected["candidate_id"] != ablated_selected["candidate_id"]
            ),
            "observed_candidate": selected["candidate_id"],
            "without_observation_candidate": ablated_selected["candidate_id"],
            "same_selector_contract": True,
            "observed_prompt_sha256": hashlib.sha256(
                observed_selection_prompt.encode()
            ).hexdigest(),
            "without_observation_prompt_sha256": hashlib.sha256(
                without_observation_prompt.encode()
            ).hexdigest(),
        },
        "execution_gates": {
            "observed_proposal": "pending_paired_evaluation",
            "without_observation_proposal": "hold_no_runtime_observation",
        },
        "candidate_catalog_sha256": canonical_sha256(catalog),
        "selection_is_efficacy_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development", type=Path)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output path already exists")
    started = time.perf_counter()
    result = run(args.development, args.catalog, args.endpoint)
    result["wall_s"] = time.perf_counter() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    selection = result["model_runs"]["intervention_selection"]["decision"]
    print(json.dumps(selection, ensure_ascii=False))


if __name__ == "__main__":
    main()
