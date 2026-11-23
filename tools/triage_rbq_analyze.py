"""Compute paired holdout effects for the independent RBQ host experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def sign_test_two_sided(differences: list[float]) -> dict:
    nonzero = [value for value in differences if value != 0]
    positives = sum(value > 0 for value in nonzero)
    n = len(nonzero)
    if n == 0:
        return {"nonzero_pairs": 0, "positive_pairs": 0, "two_sided_p": 1.0}
    tail = sum(math.comb(n, k) for k in range(positives, n + 1)) / 2**n
    opposite = sum(math.comb(n, k) for k in range(0, positives + 1)) / 2**n
    return {
        "nonzero_pairs": n,
        "positive_pairs": positives,
        "two_sided_p": min(1.0, 2 * min(tail, opposite)),
    }


def paired(baseline: list[dict], treatment: list[dict], field: str) -> dict:
    baseline_by_case = {row["case_id"]: row for row in baseline}
    treatment_by_case = {row["case_id"]: row for row in treatment}
    if not baseline or not treatment:
        raise ValueError("paired arms must contain observations")
    if len(baseline_by_case) != len(baseline) or len(treatment_by_case) != len(treatment):
        raise ValueError("case IDs must be unique within each arm")
    if baseline_by_case.keys() != treatment_by_case.keys():
        raise ValueError("arms do not contain identical paired case IDs")
    differences = [
        treatment_by_case[case_id][field] - baseline_by_case[case_id][field]
        for case_id in sorted(baseline_by_case)
    ]
    return {
        "field": field,
        "paired_differences": differences,
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "sign_test": sign_test_two_sided(differences),
    }


def analyze(result: dict, agent_candidate: str) -> dict:
    groups = result["groups"]
    baseline = groups["no_intervention"]["trials"]
    agent = groups[agent_candidate]["trials"]
    rule = groups["edge_slow_rule"]["trials"]
    base_summary = groups["no_intervention"]["summary"]
    agent_summary = groups[agent_candidate]["summary"]
    primary_delta = agent_summary["work_stance_rate"] - base_summary["work_stance_rate"]
    collision_delta = agent_summary["body_collision_rate"] - base_summary["body_collision_rate"]
    interpretation = (
        "The selected candidate increased the observed paired work-stance rate. "
        "The finite holdout and collision outcomes bound this effect."
        if primary_delta > 0
        else "The selected candidate did not improve the pre-registered work-stance endpoint."
    )
    return {
        "schema": "rbq_host_paired_analysis_v1",
        "denominator_per_arm": len(baseline),
        "agent_candidate": agent_candidate,
        "primary_endpoint": {
            "baseline_work_stance_rate": base_summary["work_stance_rate"],
            "agent_work_stance_rate": agent_summary["work_stance_rate"],
            "absolute_difference": primary_delta,
            "observed_rate_improved_without_more_collisions": primary_delta > 0
            and collision_delta <= 0,
        },
        "agent_vs_no_intervention": {
            "highest_supported_tread": paired(baseline, agent, "highest_supported_tread"),
            "maximum_forward_x_m": paired(baseline, agent, "maximum_forward_x_m"),
            "maximum_body_z_m": paired(baseline, agent, "maximum_body_z_m"),
            "maximum_abs_pitch_deg": paired(baseline, agent, "maximum_abs_pitch_deg"),
            "body_collision_rate_difference": collision_delta,
        },
        "rule_vs_no_intervention": {
            "highest_supported_tread": paired(baseline, rule, "highest_supported_tread"),
            "maximum_forward_x_m": paired(baseline, rule, "maximum_forward_x_m"),
            "maximum_body_z_m": paired(baseline, rule, "maximum_body_z_m"),
            "maximum_abs_pitch_deg": paired(baseline, rule, "maximum_abs_pitch_deg"),
            "body_collision_rate_difference": (
                groups["edge_slow_rule"]["summary"]["body_collision_rate"]
                - base_summary["body_collision_rate"]
            ),
        },
        "interpretation": interpretation + " Secondary locomotion deltas are exploratory.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("holdout", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output path already exists")
    holdout = json.loads(args.holdout.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    result = analyze(holdout, selection["decision"]["candidate_id"])
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
