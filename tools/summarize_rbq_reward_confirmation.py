"""Bind the reward-policy repair comparison to the original, per-case physical verdicts."""

import argparse
import json
from collections import Counter
from pathlib import Path

from verify_rbq_agent_study import case_hash, contrast, file_sha, read, require


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def selection(directory):
    one_path = directory / "reward-decision/summary.json"
    validated_path = directory / "validated-repair/summary.json"
    protocol_path = directory / "repair-confirmation/protocol.json"
    one = read(one_path)
    validated = read(validated_path)
    choices = {"unchanged_learned": "learned", "fixed_contact": "fixed_contact"}
    for arm in ("observed", "ablation"):
        proposal = one[arm].get("host_resolution")
        choices["single_" + arm] = proposal["source_variant"] if proposal else "learned"
        adopted = validated["adoption"][arm + "_variant"]
        details = validated["arms"][arm]
        require(
            len(details["attempted_opaque_ids"]) <= 3
            and len(set(details["attempted_opaque_ids"])) == len(details["attempted_opaque_ids"]),
            "candidate budget exceeded or repeated",
        )
        require(
            adopted is None
            or (
                details["adopted"]["source_variant"] == adopted
                and details["termination"] == "strict_development_adoption"
                and adopted != "learned"
            ),
            "failed or unverified residual was adopted",
        )
        choices["validated_" + arm] = adopted if adopted is not None else "learned"
    return {
        "choices": choices,
        "adoption": validated["adoption"],
        "decision_relpath": validated_path.relative_to(directory).as_posix(),
        "decision_sha256": file_sha(validated_path),
        "single_decision_sha256": file_sha(one_path),
        "protocol_sha256": file_sha(protocol_path),
        "policy_sha256": file_sha(directory / "reward-policy/training.json"),
    }


def evaluate(directory):
    folder = directory / "repair-confirmation"
    protocol = read(folder / "protocol.json")
    frozen = read(folder / "selection-freeze.json")
    require(frozen == selection(directory), "frozen assignments differ from the actual proposals")
    cases = read(folder / "cases.json")
    seen_seeds = set()
    for prior in (
        "articulated/holdout-cases.json",
        "articulated/confirmation-cases.json",
        "reward-policy/holdout-cases.json",
    ):
        earlier = {c["joint_seed"] for c in read(directory / prior)}
        require(not seen_seeds.intersection(earlier), "earlier evaluation sets overlap")
        seen_seeds.update(earlier)
    require(
        not seen_seeds.intersection(c["joint_seed"] for c in cases),
        "new recovery comparison reuses a prior case seed",
    )
    physics_path = folder / "physics.json"
    physics = read(physics_path)
    mode_names = protocol["unchanged_controls"]
    require(
        len(cases) == protocol["count"] == len({c["joint_seed"] for c in cases}),
        "repair cases are missing or duplicated",
    )
    require(
        case_hash(cases) == physics["case_set_sha256"] == protocol["case_set_sha256"],
        "case-set commitment mismatch",
    )
    require(
        physics["policy_sha256"]
        == frozen["policy_sha256"]
        == protocol["policy_sha256"]
        == file_sha(directory / "reward-policy/training.json"),
        "different learned policy",
    )
    require(
        frozen["protocol_sha256"] == file_sha(folder / "protocol.json")
        and frozen["decision_sha256"] == file_sha(directory / frozen["decision_relpath"]),
        "selection was changed after release",
    )
    require(set(physics["variants"]) == set(mode_names), "missing physical comparator")
    require(
        physics["parameters"] == read(directory / "reward-policy/training.json")["parameters"],
        "wrong learned parameters",
    )
    return folder, protocol, frozen, cases, physics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("freeze", "prepare-strict", "summarize", "verify"))
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    if args.operation == "freeze":
        write_new(
            args.evidence / "repair-confirmation/selection-freeze.json", selection(args.evidence)
        )
        return
    folder, protocol, freeze, cases, physics = evaluate(args.evidence)
    study_path = folder / "strict-input.json"
    if args.operation == "prepare-strict":
        write_new(
            study_path,
            {
                "scene_sha256": read(args.evidence / "selection-freeze.json")["scene_sha256"],
                "physics_sha256": file_sha(folder / "physics.json"),
                "trials": physics["trials"],
            },
        )
        return
    strict = read(folder / "strict.json")
    require(
        strict["study_sha256"] == file_sha(study_path)
        and read(study_path)["physics_sha256"] == file_sha(folder / "physics.json")
        and strict["scoring_source_sha256"] == protocol["scoring_source_sha256"],
        "strict verdict was produced for a different study or scorer",
    )
    rows = {(r["case_id"], r["candidate_id"]): r for r in physics["trials"]}
    bills = {(r["case_id"], r["candidate_id"]): r for r in strict["trials"]}
    expected = {
        (case["case_id"], mode) for case in cases for mode in protocol["unchanged_controls"]
    }
    require(
        set(rows) == set(bills) == expected
        and len(rows) == len(physics["trials"]) == len(strict["trials"]),
        "incomplete pairs",
    )
    groups = {}
    for mode in protocol["unchanged_controls"]:
        measurements = {"strict": [], "work": [], "collision": []}
        failures = Counter()
        for case in cases:
            key = case["case_id"], mode
            row, bill = rows[key], bills[key]
            require(
                row["state_sha256"] == bill["state_sha256"] and row["variant"] == mode,
                "state identity mismatch",
            )
            require(
                row["work_stance_reached"] == (row["maximum_work_stance_s"] >= 0.5),
                "incorrect work-stance classification",
            )
            if bill["strict_success"]:
                require(
                    row["work_stance_reached"]
                    and not row["body_collision"]
                    and bill["complete_horizon"]
                    and bill["walking"]["landings_evaluated"] == 18
                    and all(bill[stage]["verified"] for stage in ("walking", "valve", "station")),
                    "reported success lacks a physical stage",
                )
            else:
                failure = bill["first_failure"]
                failures[failure if isinstance(failure, str) else failure["metric"]] += 1
            measurements["strict"].append(bill["strict_success"])
            measurements["work"].append(row["work_stance_reached"])
            measurements["collision"].append(row["body_collision"])
        groups[mode] = {**measurements, "failures": dict(failures)}
    choices = freeze["choices"]
    observed = groups[choices["validated_observed"]]
    comparisons = {
        label: {k: contrast(observed[k], groups[mode][k]) for k in ("strict", "work", "collision")}
        for label, mode in choices.items()
    }
    baseline = comparisons["unchanged_learned"]
    passed = baseline["strict"]["paired_gain"] >= 0.25 and baseline["collision"]["paired_gain"] <= 0
    result = {
        "case_count": len(cases),
        "choices": choices,
        "criterion_passed": passed,
        "strict_sha256": file_sha(folder / "strict.json"),
        "physics_sha256": file_sha(folder / "physics.json"),
        "groups": {
            mode: {k: (sum(v) if isinstance(v, list) else v) for k, v in group.items()}
            for mode, group in groups.items()
        },
        "paired_validated_observed_vs": comparisons,
    }
    result_path = folder / "results.json"
    if args.operation == "verify":
        require(
            read(result_path) == result, "published paired result differs from the raw verdicts"
        )
    else:
        write_new(result_path, result)
    print("PASS" if passed else "NO GAIN", result["groups"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
