"""Select one RBQ host candidate with a local model and a strict receipt gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

RL_TRIAGE = Path.home() / "rl-triage-agent" / "src"
sys.path.insert(0, str(RL_TRIAGE))

from rl_triage.active_diagnosis import receipt_facts  # noqa: E402

ALLOWED = ("no_intervention", "phase_15", "phase_25")


def make_receipt(development: dict) -> tuple[dict, dict]:
    measurement = {
        candidate_id: {
            "work_stance_rate": group["summary"]["work_stance_rate"],
            "mean_highest_supported_tread": group["summary"]["mean_highest_supported_tread"],
            "body_collision_rate": group["summary"]["body_collision_rate"],
            "mean_maximum_forward_x_m": group["summary"]["mean_maximum_forward_x_m"],
            "mean_maximum_abs_pitch_deg": group["summary"]["mean_maximum_abs_pitch_deg"],
        }
        for candidate_id, group in development["groups"].items()
        if candidate_id in ALLOWED
    }
    receipt = {
        "request_id": "rbqdev1",
        "probe": "runtime_contact_window",
        "outcome": "abnormal",
        "exit_code": 0,
        "over_budget": False,
        "measurement": measurement,
    }
    return receipt, receipt_facts(receipt)


def schema(fact_ids: list[str]) -> dict:
    return {
        "type": "object",
        "required": ["candidate_id", "claims", "reason"],
        "properties": {
            "candidate_id": {"type": "string", "enum": list(ALLOWED)},
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


def validate(decision: object, facts: dict) -> dict:
    if not isinstance(decision, dict) or set(decision) != {"candidate_id", "claims", "reason"}:
        raise ValueError("selector response fields")
    if decision["candidate_id"] not in ALLOWED:
        raise ValueError("candidate is outside the frozen catalog")
    claims = decision["claims"]
    if (
        not isinstance(claims, list)
        or len(claims) < 2
        or len(claims) != len(set(claims))
        or any(claim not in facts for claim in claims)
    ):
        raise ValueError("claims are not exact receipt fact IDs")
    if not isinstance(decision["reason"], str) or not decision["reason"].strip():
        raise ValueError("reason must be non-empty")
    return {
        "candidate_id": decision["candidate_id"],
        "claims": [{"id": fact_id, "value": facts[fact_id]} for fact_id in claims],
        "reason": decision["reason"].strip(),
    }


def select(development: dict, endpoint: str) -> dict:
    receipt, facts = make_receipt(development)
    public = {
        "host_contract": "rbq_stair_intervention_v1_not_nvidia_registered",
        "objective_order": [
            "maximize work_stance_rate",
            "maximize mean_highest_supported_tread",
            "minimize body_collision_rate",
            "maximize mean_maximum_forward_x_m",
        ],
        "allowed_candidates": list(ALLOWED),
        "receipt_facts": facts,
        "holdout_disclosed": False,
    }
    prompt = (
        "Choose exactly one frozen RBQ host candidate using only the public development receipt. "
        "Follow objective_order lexicographically. Cite at least two exact receipt fact IDs. "
        "This is an independent host experiment, not an NVIDIA registered adaptation. "
        "Output JSON.\n"
        + json.dumps(public, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
    payload = {
        "model": "gemma3:4b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": schema(sorted(facts)),
        "options": {"temperature": 0, "seed": 42, "num_gpu": 0, "num_predict": 512},
        "keep_alive": 0,
    }
    started = time.perf_counter()
    request = Request(
        endpoint,
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        provider = json.loads(response.read().decode())
    decision = validate(json.loads(provider["message"]["content"]), facts)
    return {
        "schema": "rbq_host_selector_v1",
        "provider_class": "local_ollama_gemma_exploratory_not_registered_nemotron",
        "model": provider.get("model"),
        "options": payload["options"],
        "wall_s": time.perf_counter() - started,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "development_sha256": hashlib.sha256(
            json.dumps(development, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "original_agent_boundary_used": "rl_triage.active_diagnosis.receipt_facts",
        "receipt": receipt,
        "decision": decision,
        "holdout_disclosed": False,
        "official_rl_triage_agent_result": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output path already exists")
    development = json.loads(args.development.read_text(encoding="utf-8"))
    result = select(development, args.endpoint)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
