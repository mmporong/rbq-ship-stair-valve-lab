"""Choose one immutable reward-residual repair with source-grounded fact IDs.

The model sees opaque candidate IDs and source-derived control structures, never
variant names or candidate outcomes.  ``prepare`` freezes the map and both arm
prompts before any provider call.  ``run`` independently records the ablation
and observed raw provider responses before validation through the original
``rl_triage.grounded_recovery`` typed citation API.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

RL_TRIAGE_OVERRIDE = os.environ.get("RBQ_TRIAGE_SRC")
RL_TRIAGE_DEFAULT = Path.home() / "rl-triage-agent" / "src"
if RL_TRIAGE_OVERRIDE:
    sys.path.insert(0, str(Path(RL_TRIAGE_OVERRIDE).resolve()))
elif RL_TRIAGE_DEFAULT.is_dir():
    sys.path.insert(0, str(RL_TRIAGE_DEFAULT))

from rl_triage import grounded_recovery as recovery  # noqa: E402

MODEL = "gemma3:4b"
MODEL_SEED = 2026112202
MAP_SEED = 2026112202
FORMAT_RETRY_BUDGET = 0
SEMANTIC_RETRY_BUDGET = 0
EXPECTED_VARIANTS = (
    "learned",
    "initial",
    "fixed_contact",
    "teacher_open_loop",
    "learned_no_ppo",
    "learned_contact_off",
)
OPAQUE_MAPPING = (
    ("opt-844b", "fixed_contact"),
    ("opt-72b8", "learned"),
    ("opt-3013", "learned_contact_off"),
    ("opt-2a5e", "teacher_open_loop"),
    ("opt-6cb8", "initial"),
    ("opt-01b7", "learned_no_ppo"),
)
STRUCTURES = {
    "learned": (
        "Position-error-dependent contact authority from the frozen optimized three-coefficient "
        "vector; bounded checkpoint residual authority from the same frozen vector."
    ),
    "initial": (
        "Position-error-dependent contact authority and bounded checkpoint residual authority "
        "from the fixed pre-optimization three-coefficient vector."
    ),
    "fixed_contact": (
        "Contact-feedback authority fixed at 1.0 for the full trajectory; checkpoint residual "
        "authority fixed at 0.0."
    ),
    "teacher_open_loop": (
        "Recorded teacher command only; contact-feedback and checkpoint residual authorities "
        "both fixed at 0.0."
    ),
    "learned_no_ppo": (
        "Position-error-dependent contact authority from the frozen optimized three-coefficient "
        "vector; checkpoint residual authority fixed at 0.0."
    ),
    "learned_contact_off": (
        "Contact-feedback authority fixed at 0.0; bounded checkpoint residual authority from "
        "the frozen optimized three-coefficient vector."
    ),
}
Chat = Callable[[list[dict[str, str]], dict], dict]


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


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
        "api": "rl_triage.grounded_recovery.bind_typed_proposal",
    }


def source_variants(source_path: Path) -> tuple[str, ...]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    variants = None
    function_names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.add(node.name)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "VARIANTS" for target in node.targets
        ):
            variants = tuple(ast.literal_eval(node.value))
    if variants != EXPECTED_VARIANTS:
        raise ValueError("reward source immutable variant catalog changed")
    if not {"variant_gates", "rollout"}.issubset(function_names):
        raise ValueError("reward source control functions are missing")
    return variants


def candidate_map(source_path: Path) -> dict:
    variants = source_variants(source_path)
    if {variant for _, variant in OPAQUE_MAPPING} != set(variants):
        raise ValueError("opaque map does not exactly cover the immutable source catalog")
    return {
        "schema": "rbq_residual_opaque_candidate_map_v1",
        "map_seed": MAP_SEED,
        "source": {
            "path": source_path.resolve().as_posix(),
            "sha256": file_sha256(source_path),
        },
        "ordered_candidates": [
            {
                "opaque_id": opaque_id,
                "source_variant": variant,
                "mechanics_fact_id": f"mechanics:{opaque_id}:control_law",
                "model_visible_description": STRUCTURES[variant],
            }
            for opaque_id, variant in OPAQUE_MAPPING
        ],
        "model_visibility": {
            "opaque_id": True,
            "model_visible_description": True,
            "source_variant": False,
            "candidate_development_outcome": False,
        },
    }


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def development_observations(evaluation_path: Path, strict_dir: Path) -> tuple[dict, dict]:
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    learned = [row for row in evaluation.get("trials", []) if row.get("variant") == "learned"]
    if len(learned) != 3 or {row.get("case_id") for row in learned} != {
        "dev-01",
        "dev-02",
        "dev-03",
    }:
        raise ValueError("public development evaluation needs exactly three learned rows")
    strict_paths = [strict_dir / f"dev-{index:02d}--learned.json" for index in range(1, 4)]
    strict = [json.loads(path.read_text(encoding="utf-8")) for path in strict_paths]
    if any(
        row.get("trial_identity", {}).get("case_id") != f"dev-{i:02d}"
        for i, row in enumerate(strict, 1)
    ):
        raise ValueError("strict learned receipts do not match the public development cases")
    facts = {
        "observation:development:case_count": 3,
        "observation:development:work_stance_success_count": sum(
            bool(row.get("work_stance_reached")) for row in learned
        ),
        "observation:development:strict_success_count": sum(
            bool(row.get("strict_success")) for row in strict
        ),
        "observation:development:hold_min_angle_deg": min(
            _finite(row["valve"]["first_failure"]["measured"], "hold measured")
            for row in strict
        ),
        "observation:development:hold_lower_limit_deg": 89.75,
        "observation:development:max_valve_angle_min_deg": min(
            _finite(row["valve"]["measurements"]["max_angle_deg"], "maximum angle")
            for row in strict
        ),
        "observation:development:required_open_angle_deg": 90.0,
        "observation:development:nonzero_ppo_torque_case_count": sum(
            _finite(row.get("ppo_torque_rms_nm"), "ppo torque") > 0.0 for row in learned
        ),
        "observation:development:mean_ppo_torque_rms_nm": sum(
            _finite(row.get("ppo_torque_rms_nm"), "ppo torque") for row in learned
        )
        / len(learned),
    }
    sources = {
        "evaluation": {
            "path": evaluation_path.resolve().as_posix(),
            "sha256": file_sha256(evaluation_path),
            "rows_consumed": ["dev-01/learned", "dev-02/learned", "dev-03/learned"],
        },
        "strict_learned": [
            {"path": path.resolve().as_posix(), "sha256": file_sha256(path)}
            for path in strict_paths
        ],
    }
    return facts, sources


def visible_menu(mapping: dict) -> list[dict]:
    return [
        {
            "candidate_id": row["opaque_id"],
            "mechanics_fact_id": row["mechanics_fact_id"],
            "description": row["model_visible_description"],
        }
        for row in mapping["ordered_candidates"]
    ]


def mechanics_facts(mapping: dict) -> tuple[dict[str, object], dict[str, list[str]]]:
    facts = {}
    fact_map = {}
    for row in mapping["ordered_candidates"]:
        fact_id = row["mechanics_fact_id"]
        facts[fact_id] = row["model_visible_description"]
        fact_map[row["opaque_id"]] = [fact_id]
    return facts, fact_map


def selection_prompt(menu: list[dict], observations: dict[str, object]) -> str:
    packet = {
        "objective": (
            "Select one immutable control structure most plausibly repairing the supplied "
            "development failure while preserving the completed work stance."
        ),
        "candidate_menu": menu,
        "candidate_order_seeded_before_selection": True,
        "candidate_outcomes_disclosed": False,
        "development_observations": observations,
        "citation_contract": (
            "Return the selected candidate mechanics_fact_id and one supplied observation fact "
            "ID; use null only when the observation set is empty."
        ),
    }
    return (
        "Choose exactly one candidate from the packet. Return JSON only. Do not infer that "
        "candidate order or opaque IDs encode quality.\n"
        + json.dumps(packet, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    )


def build_protocol(
    source_path: Path, evaluation_path: Path, strict_dir: Path, mapping: dict
) -> dict:
    observations, observation_sources = development_observations(evaluation_path, strict_dir)
    menu = visible_menu(mapping)
    prompts = {
        "ablation-no-observation": selection_prompt(menu, {}),
        "observed-development": selection_prompt(menu, observations),
    }
    mechanics, fact_map = mechanics_facts(mapping)
    return {
        "schema": "rbq_residual_decision_protocol_v1",
        "source_binding": {
            "reward_source": {
                "path": source_path.resolve().as_posix(),
                "sha256": file_sha256(source_path),
            },
            "development": observation_sources,
            "original_agent": recovery_source_binding(),
        },
        "model_contract": {
            "model": MODEL,
            "temperature": 0,
            "seed": MODEL_SEED,
            "num_gpu": 0,
            "format_retry_budget": FORMAT_RETRY_BUDGET,
            "semantic_retry_budget": SEMANTIC_RETRY_BUDGET,
        },
        "fairness_contract": {
            "same_model": True,
            "same_candidate_menu_order_descriptions": True,
            "only_observation_arm_differs": True,
            "candidate_results_hidden": True,
            "heldout_inputs_consumed": False,
        },
        "menu": menu,
        "mechanics_facts": mechanics,
        "candidate_fact_ids": fact_map,
        "observations": observations,
        "arms": {
            label: {"prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
            for label, prompt in prompts.items()
        },
    }


def prepare(source_path: Path, evaluation_path: Path, strict_dir: Path, out_dir: Path) -> dict:
    if out_dir.exists():
        raise ValueError("reward decision output directory already exists")
    out_dir.mkdir(parents=True)
    mapping = candidate_map(source_path)
    map_path = out_dir / "candidate-map.json"
    write_json_new(map_path, mapping)
    protocol = build_protocol(source_path, evaluation_path, strict_dir, mapping)
    protocol["candidate_map"] = {"path": map_path.name, "sha256": file_sha256(map_path)}
    protocol_path = out_dir / "protocol.json"
    write_json_new(protocol_path, protocol)
    return {"candidate_map": map_path, "protocol": protocol_path}


def ollama_chat(endpoint: str, messages: list[dict[str, str]], schema: dict) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "format": schema,
        "options": {
            "temperature": 0,
            "seed": MODEL_SEED,
            "num_gpu": 0,
            "num_predict": 512,
        },
        "keep_alive": 0,
    }
    request = Request(
        endpoint,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode())


def provider_decision(provider: dict) -> object:
    try:
        return json.loads(provider["message"]["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("provider did not return a JSON decision") from exc


def resolved_variant(mapping: dict, opaque_id: str) -> str:
    return next(
        row["source_variant"]
        for row in mapping["ordered_candidates"]
        if row["opaque_id"] == opaque_id
    )


def run(
    source_path: Path,
    evaluation_path: Path,
    strict_dir: Path,
    out_dir: Path,
    endpoint: str,
    chat: Chat | None = None,
) -> dict:
    map_path = out_dir / "candidate-map.json"
    protocol_path = out_dir / "protocol.json"
    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_mapping = candidate_map(source_path)
    if mapping != expected_mapping:
        raise ValueError("precommitted opaque candidate map does not match current source")
    expected_protocol = build_protocol(source_path, evaluation_path, strict_dir, mapping)
    expected_protocol["candidate_map"] = {"path": map_path.name, "sha256": file_sha256(map_path)}
    if protocol != expected_protocol:
        raise ValueError("precommitted protocol does not match current public development inputs")
    call = chat or (lambda messages, schema: ollama_chat(endpoint, messages, schema))
    candidate_ids = [row["opaque_id"] for row in mapping["ordered_candidates"]]
    mechanics = protocol["mechanics_facts"]
    fact_map = protocol["candidate_fact_ids"]

    def execute_arm(label: str, observations: dict[str, object]) -> dict:
        prompt = protocol["arms"][label]["prompt"]
        facts = {**mechanics, **observations}
        observation_ids = list(observations)
        base = {"label": label, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        try:
            provider = call(
                [{"role": "user", "content": prompt}],
                recovery.typed_proposal_schema(candidate_ids, facts, fact_map, observation_ids),
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
        record = {**base, "provider_path": raw_path.name, "provider_sha256": file_sha256(raw_path)}
        try:
            bound = recovery.bind_typed_proposal(
                provider_decision(provider), facts, candidate_ids, fact_map, observation_ids
            )
        except Exception as exc:
            return {
                **record,
                "status": "invalid_model_proposal",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
        return {
            **record,
            "status": "valid",
            "proposal": bound,
            "host_resolution": {
                "opaque_id": bound["candidate_id"],
                "source_variant": resolved_variant(mapping, bound["candidate_id"]),
            },
        }

    ablation = execute_arm("ablation-no-observation", {})
    observed = execute_arm("observed-development", protocol["observations"])
    summary = {
        "schema": "rbq_residual_decision_result_v1",
        "model": MODEL,
        "protocol": {"path": protocol_path.name, "sha256": file_sha256(protocol_path)},
        "candidate_map": {"path": map_path.name, "sha256": file_sha256(map_path)},
        "original_agent": recovery_source_binding(),
        "ablation": ablation,
        "observed": observed,
        "provider_calls_independent": True,
        "retry_performed": False,
        "heldout_inputs_consumed": False,
        "efficacy_claim_supported": False,
        "efficacy_note": "Selection alone is not physical efficacy evidence.",
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
