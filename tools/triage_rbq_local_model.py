"""Run an exploratory local model through a frozen RBQ triage host gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def facts(assessment: dict) -> dict:
    fresh = assessment["fresh_rbq_cpu_evaluation"]
    v7 = fresh["baseline_v7"]
    v8 = fresh["comparison_v8"]
    return {
        "fresh_rbq.v7.work_platform_reached": v7["work_platform_reached"],
        "fresh_rbq.v8.work_platform_reached": v8["work_platform_reached"],
        "fresh_rbq.v7.maximum_supported_tread": max(v7["maximum_supported_tread_by_trial"]),
        "fresh_rbq.v8.maximum_supported_tread": max(v8["maximum_supported_tread_by_trial"]),
        "fresh_rbq.v7.termination": sorted(set(v7["termination_by_trial"])),
        "fresh_rbq.v8.termination": sorted(set(v8["termination_by_trial"])),
        "registered_scope.supported_task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "registered_scope.rbq_compatible": False,
    }


def response_schema(protocol: dict) -> dict:
    return {
        "type": "object",
        "required": ["action", "cited_facts", "reason"],
        "properties": {
            "action": {"type": "string", "enum": protocol["allowed_actions"]},
            "cited_facts": {
                "type": "array",
                "items": {"type": "string", "enum": protocol["allowed_citations"]},
                "minItems": 2,
                "uniqueItems": True,
            },
            "reason": {"type": "string", "minLength": 1},
        },
        "additionalProperties": False,
    }


def validate(value: object, protocol: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {"action", "cited_facts", "reason"}:
        raise ValueError("local model response fields")
    if value["action"] not in protocol["allowed_actions"]:
        raise ValueError("local model action")
    cited = value["cited_facts"]
    if (
        not isinstance(cited, list)
        or len(cited) < 2
        or len(cited) != len(set(cited))
        or any(item not in protocol["allowed_citations"] for item in cited)
    ):
        raise ValueError("local model citations")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("local model reason")
    return value


def run(protocol: dict, assessment: dict) -> dict:
    fact_values = facts(assessment)
    prompt = (
        "You are an exploratory local RL triage model. Choose exactly one allowed action. "
        "The host scope is authoritative. If the case is outside the registered supported task, "
        "choose HOLD. Cite at least two exact fact IDs. Do not claim a correction or recovery.\n"
        + json.dumps({"facts": fact_values, "allowed_actions": protocol["allowed_actions"]})
    )
    payload = {
        "model": protocol["model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": response_schema(protocol),
        "options": protocol["options"],
        "keep_alive": 0,
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    request = Request(
        protocol["endpoint"], raw, method="POST", headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    with urlopen(request, timeout=300) as response:
        provider = json.loads(response.read().decode("utf-8"))
    decision = validate(json.loads(provider["message"]["content"]), protocol)
    gate = protocol["host_gate"]
    disposition = "model_hold" if decision["action"] == "HOLD" else gate["selected_check_result"]
    return {
        "schema": "rbq_local_triage_result_v1",
        "provider_class": protocol["provider_class"],
        "model_requested": protocol["model"],
        "model_returned": provider.get("model"),
        "options": protocol["options"],
        "wall_s": time.perf_counter() - started,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "decision": decision,
        "host_disposition": disposition,
        "host_gate": gate,
        "model_calls": 1,
        "checks_executed": 0,
        "approvals_created": 0,
        "interventions_created": 0,
        "rbq_improvement_measured": False,
        "claim_limits": protocol["claim_limits"],
        "facts": fact_values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    protocol = read_json(args.protocol)
    assessment = read_json(args.assessment)
    if digest(args.assessment) != protocol["assessment_sha256"]:
        raise ValueError("assessment SHA-256 changed")
    value = run(protocol, assessment)
    args.out.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
