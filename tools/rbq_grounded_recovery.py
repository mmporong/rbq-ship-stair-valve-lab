"""Source-grounded development recovery loop for the RBQ articulated mission.

The local model cites public fact IDs.  The host binds those IDs to the exact
source values before using the original ``active_diagnosis.validate_decision``
contract.  Native diagnosis remains HOLD because the RBQ contact observation is
not an original registered probe.  Candidate outcomes are revealed only after a
proposal and at most three unique interventions are consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

RL_TRIAGE_OVERRIDE = os.environ.get("RBQ_TRIAGE_SRC")
RL_TRIAGE_DEFAULT = Path.home() / "rl-triage-agent" / "src"
if RL_TRIAGE_OVERRIDE:
    sys.path.insert(0, str(Path(RL_TRIAGE_OVERRIDE).resolve()))
elif RL_TRIAGE_DEFAULT.is_dir():
    sys.path.insert(0, str(RL_TRIAGE_DEFAULT))

from rl_triage import active_diagnosis as diagnosis  # noqa: E402
from rl_triage import grounded_recovery as recovery  # noqa: E402

MODEL = "gemma3:4b"
MAX_INTERVENTION_TRIALS = 3
REQUIRED_WORK_STANCE_SUCCESSES = 2
FORMAT_RETRY_BUDGET = 0
SEMANTIC_RETRY_BUDGET = 0
NO_INTERVENTION = "no_intervention"
Chat = Callable[[list[dict[str, str]], dict], dict]


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recovery_source_binding() -> dict:
    module_path = Path(recovery.__file__).resolve()
    repository = module_path.parents[2]
    process = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "module_path": module_path.as_posix(),
        "module_sha256": file_sha256(module_path),
        "git_repository": repository.as_posix(),
        "git_sha": process.stdout.strip(),
        "explicit_src_override": RL_TRIAGE_OVERRIDE is not None,
    }


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def load_catalog(path: Path) -> dict:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict) or set(catalog) != {"schema", "candidates"}:
        raise ValueError("catalog fields must be schema and candidates")
    if catalog["schema"] != "rbq_intervention_catalog_v1":
        raise ValueError("unknown candidate catalog schema")
    candidates = catalog["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate catalog is empty")
    seen = set()
    for row in candidates:
        expected = {"candidate_id", "description", "executable", "mechanics"}
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError("candidate catalog row fields do not match the frozen contract")
        candidate_id = row["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            raise ValueError("candidate IDs must be unique non-empty strings")
        if row["executable"] is not True:
            raise ValueError(f"candidate {candidate_id!r} is not executable")
        if not isinstance(row["mechanics"], dict) or not row["mechanics"]:
            raise ValueError("candidate mechanics must be a non-empty object")
        seen.add(candidate_id)
    if NO_INTERVENTION not in seen:
        raise ValueError("candidate catalog must include no_intervention")
    return catalog


def load_development(path: Path, candidate_ids: set[str]) -> dict:
    development = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "case_set_sha256",
        "reference_sha256",
        "scene_sha256",
        "control_source_sha256",
        "simulation_source_sha256",
        "model",
        "clock_s",
        "trials",
    }
    if not isinstance(development, dict) or set(development) != required:
        raise ValueError("articulated development fields do not match the frozen contract")
    trials = development["trials"]
    if not isinstance(trials, list) or not trials:
        raise ValueError("articulated development needs trials")
    by_candidate = {candidate_id: [] for candidate_id in candidate_ids}
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("development trial must be an object")
        candidate_id = trial.get("candidate_id")
        if candidate_id not in by_candidate:
            continue
        case_id = trial.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("development trial needs a case_id")
        for field in (
            "simulated_s",
            "max_nonfoot_terrain_force_n",
            "maximum_work_stance_s",
            "maximum_valve_angle_deg",
            "final_valve_angle_deg",
        ):
            _finite(trial.get(field), f"{candidate_id}.{case_id}.{field}")
        if not isinstance(trial.get("work_stance_reached"), bool) or not isinstance(
            trial.get("body_collision"), bool
        ):
            raise ValueError("trial work stance and collision fields must be boolean")
        state_sha = trial.get("state_sha256")
        if not isinstance(state_sha, str) or len(state_sha) != 64:
            raise ValueError("trial state SHA-256 is missing")
        by_candidate[candidate_id].append(trial)
    for candidate_id, rows in by_candidate.items():
        case_ids = [row["case_id"] for row in rows]
        if len(rows) != 3 or len(set(case_ids)) != 3:
            raise ValueError(
                f"candidate {candidate_id!r} needs exactly three unique development cases"
            )
    return {**development, "by_candidate": by_candidate}


def verify_physics_sources(development: dict, sources: dict[str, Path]) -> dict:
    expected_fields = {
        "scene": "scene_sha256",
        "reference": "reference_sha256",
        "control_source": "control_source_sha256",
        "simulation_source": "simulation_source_sha256",
    }
    if set(sources) != set(expected_fields):
        raise ValueError("physics source paths are incomplete")
    bindings = {}
    for source_id, field in expected_fields.items():
        path = sources[source_id].resolve()
        if not path.is_file():
            raise ValueError(f"physics source does not exist: {path}")
        actual = file_sha256(path)
        if actual != development[field]:
            raise ValueError(f"physics source hash mismatch for {source_id}")
        bindings[source_id] = {"path": path.as_posix(), "sha256": actual}
    return bindings


def mechanics_facts(catalog: dict) -> dict[str, object]:
    facts = {}
    for row in catalog["candidates"]:
        candidate_id = row["candidate_id"]
        for field, value in row["mechanics"].items():
            if not isinstance(value, (str, bool, int, float)):
                raise ValueError("candidate mechanics values must be JSON scalars")
            facts[f"candidate:{candidate_id}:mechanics.{field}"] = value
    return facts


def baseline_receipt(development: dict) -> dict:
    rows = development["by_candidate"][NO_INTERVENTION]
    return candidate_receipt(NO_INTERVENTION, rows, baseline_collision_count=None)


def candidate_receipt(
    candidate_id: str, rows: list[dict], baseline_collision_count: int | None
) -> dict:
    work = sum(row["work_stance_reached"] for row in rows)
    collisions = sum(row["body_collision"] for row in rows)
    valve_terminal = sum(
        row["work_stance_reached"]
        and not row["body_collision"]
        and abs(row["final_valve_angle_deg"] - 90.0) <= 0.25
        for row in rows
    )
    measurement = {
        "candidate_id": candidate_id,
        "case_count": len(rows),
        "work_stance_success_count": work,
        "body_collision_count": collisions,
        "mean_highest_supported_tread": sum(row["highest_supported_tread"] for row in rows)
        / len(rows),
        "valve_terminal_indicator_count": valve_terminal,
        "trial_state_set_sha256": canonical_sha256(
            [{"case_id": row["case_id"], "state_sha256": row["state_sha256"]} for row in rows]
        ),
    }
    adopted = None
    if baseline_collision_count is not None:
        adopted = work >= REQUIRED_WORK_STANCE_SUCCESSES and collisions <= baseline_collision_count
        measurement["adoption_threshold_met"] = adopted
        measurement["baseline_body_collision_count"] = baseline_collision_count
    receipt = {
        "request_id": f"rbq_dev_{candidate_id}",
        "probe": "go2_wall_articulated_candidate_trial",
        "outcome": "normal" if adopted else "abnormal",
        "exit_code": 0,
        "over_budget": False,
        "measurement": measurement,
        "trial_receipt_sha256": canonical_sha256(rows),
    }
    return receipt


def receipt_facts(receipt: dict) -> dict[str, object]:
    """Use the original flattening path without treating the receipt as a native probe."""
    return diagnosis.receipt_facts(receipt)


def candidate_fact_map(catalog: dict, available: list[str]) -> dict[str, list[str]]:
    return {
        candidate_id: [
            f"candidate:{candidate_id}:mechanics.{field}"
            for field in next(
                row["mechanics"]
                for row in catalog["candidates"]
                if row["candidate_id"] == candidate_id
            )
        ]
        for candidate_id in available
    }


def available_facts(facts: dict, available: list[str]) -> dict:
    allowed_prefixes = tuple(f"candidate:{candidate_id}:" for candidate_id in available)
    return {
        fact_id: value
        for fact_id, value in facts.items()
        if not fact_id.startswith("candidate:") or fact_id.startswith(allowed_prefixes)
    }


def selection_prompt(catalog: dict, facts: dict, attempted: list[str], available: list[str]) -> str:
    available_rows = [
        row for row in catalog["candidates"] if row["candidate_id"] in available
    ]
    packet = {
        "objective_order": [
            "select mechanics plausibly addressing the supplied source-bound failure evidence",
            "do not repeat an attempted intervention",
            "prefer the smallest intervention justified by available evidence",
            "choose no_intervention when no untried change is justified",
        ],
        "candidate_outcomes_disclosed_before_proposal": False,
        "available_candidates": available_rows,
        "attempted_interventions": attempted,
        "public_facts": facts,
        "native_agent_scope": "hold_only_no_registered_native_probe",
        "citation_mode": "return one mechanics fact ID and one observation fact ID",
    }
    return (
        "Choose one available candidate from the public packet. Return JSON only. "
        "Return that candidate's mechanics fact ID and one runtime observation fact ID when "
        "available; use null only when no observation is available. Candidate success data are "
        "disclosed only after "
        "selection.\n"
        + json.dumps(packet, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    )


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
    return provider


def provider_decision(provider: dict) -> object:
    try:
        return json.loads(provider["message"]["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("provider did not return a JSON decision") from exc


def write_json_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def run(
    development_path: Path,
    catalog_path: Path,
    sources: dict[str, Path],
    out_dir: Path,
    endpoint: str,
    chat: Chat | None = None,
) -> dict:
    if out_dir.exists():
        raise ValueError("output directory already exists")
    out_dir.mkdir(parents=True)
    catalog = load_catalog(catalog_path)
    catalog_ids = [row["candidate_id"] for row in catalog["candidates"]]
    development = load_development(development_path, set(catalog_ids))
    source_bindings = verify_physics_sources(development, sources)
    baseline = baseline_receipt(development)
    baseline_collisions = baseline["measurement"]["body_collision_count"]
    base_facts = {
        "source:development_sha256": file_sha256(development_path),
        "source:catalog_sha256": file_sha256(catalog_path),
        "host:registered_nvidia_task": False,
        "host:development_case_count": 3,
        **mechanics_facts(catalog),
    }
    observed_facts = {**base_facts, **receipt_facts(baseline)}
    call = chat or (lambda messages, schema: ollama_chat(endpoint, messages, schema))
    trace = []

    def propose(label: str, facts: dict, attempted: list[str]) -> dict:
        available = recovery.bounded_candidate_ids(
            catalog_ids, attempted, NO_INTERVENTION, MAX_INTERVENTION_TRIALS
        )
        scoped_facts = available_facts(facts, available)
        candidate_facts = candidate_fact_map(catalog, available)
        observation_facts = sorted(
            fact_id for fact_id in scoped_facts if fact_id.startswith("receipt:")
        )
        prompt = selection_prompt(catalog, scoped_facts, attempted, available)
        base = {
            "label": label,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "physics_source_bindings": source_bindings,
        }
        try:
            provider = call(
                [{"role": "user", "content": prompt}],
                recovery.typed_proposal_schema(
                    available, scoped_facts, candidate_facts, observation_facts
                ),
            )
        except Exception as exc:
            failure_path = out_dir / f"{label}-provider-failure.json"
            write_json_new(
                failure_path,
                {"exception_type": type(exc).__name__, "exception_message": str(exc)},
            )
            return {
                **base,
                "status": "provider_failure",
                "failure_path": failure_path.name,
                "failure_sha256": file_sha256(failure_path),
            }
        raw_path = out_dir / f"{label}-provider.json"
        write_json_new(raw_path, provider)
        recorded = {
            **base,
            "provider_path": raw_path.name,
            "provider_sha256": file_sha256(raw_path),
        }
        try:
            raw = provider_decision(provider)
            bound = recovery.bind_typed_proposal(
                raw, scoped_facts, available, candidate_facts, observation_facts
            )
        except Exception as exc:
            return {
                **recorded,
                "status": "invalid_model_proposal",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
        return {**recorded, "status": "valid", "proposal": bound}

    ablation = propose("ablation-no-observation", base_facts, [])
    attempted: list[str] = []
    adopted = None
    termination = "candidate_budget_exhausted"
    for index in range(MAX_INTERVENTION_TRIALS):
        step = propose(f"step-{index:02d}", observed_facts, attempted)
        if step["status"] != "valid":
            step["host_action"] = "hold_invalid_observed_proposal"
            trace.append(step)
            termination = "observed_proposal_invalid_hold"
            break
        candidate_id = step["proposal"]["candidate_id"]
        if candidate_id == NO_INTERVENTION:
            state = recovery.advance(
                attempted,
                candidate_id,
                no_intervention_id=NO_INTERVENTION,
                max_attempts=MAX_INTERVENTION_TRIALS,
                adoption_threshold_met=None,
            )
            step["host_action"] = state["status"]
            trace.append(step)
            termination = "model_selected_no_intervention"
            break
        rows = development["by_candidate"][candidate_id]
        receipt = candidate_receipt(candidate_id, rows, baseline_collisions)
        receipt_path = out_dir / f"step-{index:02d}-physics-receipt.json"
        receipt_record = {
            "schema": "rbq_articulated_candidate_receipt_v1",
            "candidate_id": candidate_id,
            "physics_source_bindings": source_bindings,
            "development_source": {
                "path": development_path.resolve().as_posix(),
                "sha256": file_sha256(development_path),
            },
            "receipt": receipt,
            "strict_full_mission": {
                "used_for_adoption": False,
                "valve_terminal_indicator_count": receipt["measurement"][
                    "valve_terminal_indicator_count"
                ],
                "independent_verdict_required": True,
            },
        }
        write_json_new(receipt_path, receipt_record)
        step["host_action"] = "candidate_trial_consumed"
        step["physics_receipt_path"] = receipt_path.name
        step["physics_receipt_sha256"] = file_sha256(receipt_path)
        step["adoption_threshold_met"] = receipt["measurement"]["adoption_threshold_met"]
        trace.append(step)
        state = recovery.advance(
            attempted,
            candidate_id,
            no_intervention_id=NO_INTERVENTION,
            max_attempts=MAX_INTERVENTION_TRIALS,
            adoption_threshold_met=step["adoption_threshold_met"],
        )
        attempted = state["attempted"]
        if state["status"] == "adopted":
            adopted = state["adopted"]
            termination = "development_adoption_threshold_met"
            break
        if state["status"] == "budget_exhausted":
            termination = "candidate_budget_exhausted"
            break
        observed_facts.update(receipt_facts(receipt))

    result = {
        "schema": "rbq_grounded_recovery_v3",
        "model": MODEL,
        "source_commit_required_before_live_run": True,
        "source_bindings": {
            "development": {
                "path": development_path.resolve().as_posix(),
                "sha256": file_sha256(development_path),
            },
            "catalog": {
                "path": catalog_path.resolve().as_posix(),
                "sha256": file_sha256(catalog_path),
            },
            "physics": source_bindings,
        },
        "fairness_contract": {
            "maximum_unique_intervention_trials": MAX_INTERVENTION_TRIALS,
            "duplicate_interventions_allowed": False,
            "candidate_outcome_visible_only_after_proposal": True,
            "adoption_work_stance_successes_required": REQUIRED_WORK_STANCE_SUCCESSES,
            "collision_increase_allowed": False,
            "strict_full_mission_used_for_adoption": False,
            "format_retry_budget": FORMAT_RETRY_BUDGET,
            "semantic_retry_budget": SEMANTIC_RETRY_BUDGET,
        },
        "original_agent_boundary": {
            "native_action": "hold",
            "registered_native_probe_used": False,
            "synthetic_native_ranking_created": False,
            "typed_id_citation_mode": True,
            "fact_flattening_api": "rl_triage.active_diagnosis.receipt_facts",
            "recovery_api": "rl_triage.grounded_recovery.bind_typed_proposal",
            "recovery_source": recovery_source_binding(),
        },
        "baseline_receipt": baseline,
        "no_observation_ablation": ablation,
        "trace": trace,
        "attempted_interventions": attempted,
        "adopted_candidate": adopted,
        "termination": termination,
        "efficacy_claim_supported": False,
        "efficacy_note": "Development adoption is not holdout efficacy evidence.",
    }
    write_json_new(out_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development", type=Path)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--control-source", type=Path, required=True)
    parser.add_argument("--simulation-source", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        result = run(
            args.development,
            args.catalog,
            {
                "scene": args.scene,
                "reference": args.reference,
                "control_source": args.control_source,
                "simulation_source": args.simulation_source,
            },
            args.out_dir,
            args.endpoint,
        )
    except Exception as exc:
        if args.out_dir.is_dir() and not (args.out_dir / "failure.json").exists():
            write_json_new(
                args.out_dir / "failure.json",
                {
                    "schema": "rbq_grounded_recovery_failure_v2",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "retry_performed": False,
                    "format_retry_budget": FORMAT_RETRY_BUDGET,
                    "semantic_retry_budget": SEMANTIC_RETRY_BUDGET,
                    "wall_s": time.perf_counter() - started,
                },
            )
        raise
    print(json.dumps({"adopted_candidate": result["adopted_candidate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
