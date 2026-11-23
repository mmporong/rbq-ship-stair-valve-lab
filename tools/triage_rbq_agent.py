"""Audit RL Triage Agent applicability against frozen RBQ stair policies."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

START_X_M = -1.1
RBQ_TASK = {
    "robot": "RBQ-10+RB1",
    "direction": "up",
    "slope_deg": 50,
    "rise_m": 0.19,
    "goal": "reach the work platform and operate the valve",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_metrics(left: object, right: object) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            same_metrics(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            same_metrics(a, b) for a, b in zip(left, right)
        )
    return left == right


def route_score(result: dict) -> dict:
    trials = result["trials"]
    reached = [trial for trial in trials if "on_top_s" in trial["events"]]
    supported = [max(trial["ascent_treads"], default=0) for trial in trials]
    progress = [trial["maximum_forward_x_m"] - START_X_M for trial in trials]
    forces = [
        trial["first_illegal_contacts"][0]["force_n"] if trial["first_illegal_contacts"] else None
        for trial in trials
    ]
    return {
        "checkpoint": result["checkpoint"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "trials": len(trials),
        "work_platform_reached": len(reached),
        "round_trip_passed": sum(trial["passed_round_trip"] for trial in trials),
        "maximum_supported_tread_by_trial": supported,
        "maximum_forward_progress_m_by_trial": progress,
        "first_illegal_contact_force_n_by_trial": forces,
        "termination_by_trial": [trial["termination"] for trial in trials],
    }


def paired_effect(baseline: dict, comparison: dict) -> list[dict]:
    base = {trial["speed_mps"]: trial for trial in baseline["trials"]}
    changed = {trial["speed_mps"]: trial for trial in comparison["trials"]}
    if set(base) != set(changed):
        raise ValueError("paired policies must use the same speeds")
    return [
        {
            "speed_mps": speed,
            "forward_progress_delta_m": (
                changed[speed]["maximum_forward_x_m"] - base[speed]["maximum_forward_x_m"]
            ),
            "supported_tread_delta": (
                max(changed[speed]["ascent_treads"], default=0)
                - max(base[speed]["ascent_treads"], default=0)
            ),
            "work_platform_reached_delta": (
                int("on_top_s" in changed[speed]["events"])
                - int("on_top_s" in base[speed]["events"])
            ),
        }
        for speed in sorted(base)
    ]


def load_contract(triage_repo: Path):
    source = triage_repo / "evals/stairs_agent_repair_v2.py"
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("rbq_triage_contract", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recorded_control(triage_repo: Path, contract) -> dict:
    folder = triage_repo / "evals/results/go2_stairs_agent_adaptation_s200_20261106_v1"
    source = read_json(folder / "source.json")
    diagnosis = read_json(folder / "diagnosis.json")
    summary = read_json(folder / "summary.json")
    before_rows = [
        json.loads(line)
        for line in (folder / "baseline-state-trace.jsonl").read_text().splitlines()
    ]
    after_rows = [
        json.loads(line)
        for line in (folder / "adapted-state-trace.jsonl").read_text().splitlines()
    ]
    before = contract.assess(before_rows)
    after = contract.assess(after_rows)
    if not same_metrics(before, source["assessment"]) or not same_metrics(
        before, summary["before"]
    ):
        raise ValueError("recorded baseline assessment did not reproduce")
    if not same_metrics(after, summary["after"]):
        raise ValueError("recorded adapted assessment did not reproduce")
    receipts = [contract.probe(probe_id, source) for probe_id in contract.PROBES]
    if receipts != diagnosis["receipts"]:
        raise ValueError("recorded probe receipts did not reproduce")
    contract.validate_decision(diagnosis["decision"], contract.packet(source, receipts))
    authorization = contract.authorize_adaptation(source, receipts, diagnosis["decision"], True)
    stored_authorization = read_json(folder / "adaptation-authorization.json")
    if authorization != stored_authorization:
        raise ValueError("recorded adaptation authorization did not reproduce")
    return {
        "status": "passed_recorded_control_revalidation",
        "fresh_simulation": False,
        "task": contract.TASK,
        "before": before,
        "after": after,
        "probe_receipts_reproduced": True,
        "decision_contract_reproduced": True,
        "authorization_reproduced": True,
        "summary_sha256": sha256(folder / "summary.json"),
        "scope": (
            "This proves the recorded standard-Go2 control remains internally valid; "
            "it does not transfer its effect to RBQ-10+RB1."
        ),
    }


def git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def build_report(args: argparse.Namespace) -> dict:
    triage_repo = args.triage_repo.resolve(strict=True)
    contract = load_contract(triage_repo)
    prepare = read_json(args.agent_prepare / "entry_report.json")
    v7 = read_json(args.v7)
    v8 = read_json(args.v8)
    registered_task = contract.TASK
    task_mismatches = {
        "robot": [registered_task["robot"], RBQ_TASK["robot"]],
        "direction": [registered_task["direction"], RBQ_TASK["direction"]],
        "step_height_m": [registered_task["step_height_m"], RBQ_TASK["rise_m"]],
        "geometry": [
            "0.05 m high, 0.30 m wide descending steps",
            "50 deg, 0.19 m rise ship stair",
        ],
    }
    return {
        "schema": "rbq_rl_triage_applicability_v1",
        "triage_agent": {
            "repository_head": git_head(triage_repo),
            "registered_control": recorded_control(triage_repo, contract),
            "rbq_prepare_only": {
                "status": "prepared" if prepare["mode"] == "prepare_only" else "unexpected",
                "dependencies_ok": prepare["dependencies"]["ok"],
                "config_ok": prepare["config"]["ok"],
                "telemetry_evidence_count": prepare["catalog"]["evidence_count"],
                "telemetry_group_count": prepare["catalog"]["group_count"],
                "ready_for_model_execute": prepare["ready_for_execute"],
                "model_calls": prepare["model_calls"],
                "checks_executed": prepare["check_execution"]["performed"],
                "cause_confirmed": prepare["claims"]["cause_confirmed"],
                "correction_applied": prepare["claims"]["correction_applied"],
                "policy_recovery_verified": prepare["claims"]["policy_recovery_verified"],
            },
            "registered_adaptation_compatible_with_rbq_task": False,
            "task_mismatches": task_mismatches,
        },
        "fresh_rbq_cpu_evaluation": {
            "task": RBQ_TASK,
            "baseline_v7": route_score(v7),
            "comparison_v8": route_score(v8),
            "paired_v8_minus_v7": paired_effect(v7, v8),
            "agent_selected_or_authorized_v8": False,
            "comparison_interpretation": (
                "V8 is a historical human-designed termination-penalty experiment, "
                "not an RL Triage Agent intervention."
            ),
        },
        "agent_effect_on_rbq": {
            "measured": False,
            "improvement_verified": False,
            "reason": (
                "The repository ingested the RBQ training telemetry, but no registered RBQ check, "
                "RBQ adaptation authorization, adapted checkpoint, or held-out "
                "post-evaluation exists."
            ),
        },
        "minimum_independent_host_adapter": {
            "required": True,
            "inputs": [
                "frozen RBQ checkpoint and exact CPU bundle",
                "training telemetry and source hashes",
                "fresh matched 50 deg, 190 mm route traces",
            ],
            "registered_checks": [
                "task and observation integrity",
                "body-to-tread clearance and stability",
                "actuator saturation and torque limits",
                "PPO optimizer update count",
            ],
            "evaluation": (
                "Freeze task, command grid, seed split, simulator and acceptance before "
                "intervention; compare baseline, agent-selected intervention, and "
                "no-intervention or rule-selected control on unseen seeds. Count "
                "work-platform reach and valve-ready stability as primary outcomes."
            ),
        },
        "resource_use": {"new_api_calls": 0, "new_gpu_training_runs": 0, "new_gpu_evaluations": 0},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triage-repo", type=Path, required=True)
    parser.add_argument("--agent-prepare", type=Path, required=True)
    parser.add_argument("--v7", type=Path, required=True)
    parser.add_argument("--v8", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    report = build_report(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["agent_effect_on_rbq"], ensure_ascii=False))


if __name__ == "__main__":
    main()
