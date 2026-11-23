"""Verify the frozen RBQ choice against two separate paired physical test sets."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from scipy.stats import beta


def read(path):
    return json.loads(path.read_bytes())


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def case_hash(cases):
    return hashlib.sha256(
        json.dumps(cases, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_interval(count, n, tail):
    lo = float(beta.ppf(tail, count, n + 1 - count)) if count else 0.0
    hi = float(beta.ppf(1 - tail, count + 1, n - count)) if count < n else 1.0
    return lo, hi


def contrast(selected, comparator):
    if len(selected) != len(comparator) or not selected:
        raise ValueError("A contrast needs one observation per arm for every case")
    n = len(selected)
    wins = sum(a and not b for a, b in zip(selected, comparator, strict=True))
    losses = sum(b and not a for a, b in zip(selected, comparator, strict=True))
    discordant = wins + losses
    p = min(
        1.0, 2 * sum(math.comb(discordant, j) for j in range(min(wins, losses) + 1)) / 2**discordant
    )
    # Two simultaneous exact binomial intervals bound p(win)-p(loss) with >=95% coverage.
    win_ci = exact_interval(wins, n, 0.0125)
    loss_ci = exact_interval(losses, n, 0.0125)
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "paired_gain": (wins - losses) / n,
        "paired_difference_ci95_conservative": [win_ci[0] - loss_ci[1], win_ci[1] - loss_ci[0]],
        "mcnemar_two_sided_exact_p": p,
    }


def inspect_cohort(directory, split, protocol, freeze):
    folder = directory / "articulated"
    cases = read(folder / f"{split}-cases.json")
    cohort_path = folder / f"{split}.json"
    cohort = read(cohort_path)
    scored = read(folder / f"{split}-strict.json")
    native = read(directory / "frozen-ppo" / f"{split}.json")
    commit = protocol["case_set_commitments"][split]
    case_ids = [case["case_id"] for case in cases]
    count = len(case_ids)
    require(count == commit["count"] and len(set(case_ids)) == count, "missing or duplicate case")
    require(
        case_hash(cases)
        == commit["sha256"]
        == cohort["case_set_sha256"]
        == native["case_set_sha256"],
        "case commitments differ",
    )
    require(
        cohort["scene_sha256"] == scored["scene_sha256"] == freeze["scene_sha256"]
        and cohort["reference_sha256"] == freeze["reference_sha256"],
        "articulated input was changed",
    )
    for field in ("control_source_sha256", "simulation_source_sha256"):
        require(cohort[field] == freeze[field], "acting controller changed after selection")
    require(cohort["clock_s"] == freeze["clock_s"], "clock changed")
    require(scored["study_sha256"] == file_sha(cohort_path), "judge belongs to a different trial")
    require(
        scored["scoring_source_sha256"] == freeze["scoring_source_sha256"],
        "physical verdict changed after selection",
    )
    require(
        set(native["groups"]) == {"no_intervention", "phase_25", "edge_slow_rule"},
        "native policy controls are incomplete",
    )
    require(
        native["checkpoint_sha256"] == protocol["actor_checkpoint_sha256"], "checkpoint changed"
    )
    groups = {}
    all_rows = cohort["trials"]
    strict = {(r["case_id"], r["candidate_id"]): r for r in scored["trials"]}
    modes = freeze["evaluated_modes"]
    require(
        len(all_rows) == len(strict) == len(scored["trials"]) == count * len(modes),
        "incomplete case/mode matrix",
    )
    for mode in modes:
        rows = {r["case_id"]: r for r in all_rows if r["candidate_id"] == mode}
        require(set(rows) == set(case_ids), f"missing {mode} trial")
        ordered = [rows[c] for c in case_ids]
        bills = [strict[c, mode] for c in case_ids]
        for r, bill in zip(ordered, bills, strict=True):
            require(bill["state_sha256"] == r["state_sha256"], "state fingerprint mismatch")
            require(
                r["work_stance_reached"] == (r["maximum_work_stance_s"] >= 0.5),
                "reported contact endpoint differs from physical dwell",
            )
            if bill["strict_success"]:
                require(
                    bill["complete_horizon"]
                    and all(bill[key]["verified"] for key in ("walking", "valve", "station")),
                    "incomplete strict success",
                )
                require(
                    bill["walking"]["landings_evaluated"] == 18
                    and r["work_stance_reached"]
                    and not r["body_collision"]
                    and abs(r["final_valve_angle_deg"] - 90.0) <= 0.25,
                    "inconsistent completed mission",
                )
                require(
                    bill["station"]["measurements"]["station_s"] >= 5.0 - 1e-8
                    and bill["valve"]["measurements"]["hold_s"] >= 5.0 - 1e-8,
                    "incomplete hold/release",
                )
        groups[mode] = {
            "work": [r["work_stance_reached"] for r in ordered],
            "strict": [bill["strict_success"] for bill in bills],
            "collision": [r["body_collision"] for r in ordered],
            "work_at_19_96s": [r["original_horizon_work_stance"] for r in ordered],
            "strict_first_failure": dict(
                Counter(
                    (
                        b["first_failure"]
                        if isinstance(b["first_failure"], str)
                        else (b["first_failure"] or {}).get("metric", "none")
                    )
                    for b in bills
                )
            ),
        }
    chosen = freeze["adopted_candidate"]
    summary = {
        mode: {
            **{k: sum(v) for k, v in g.items() if isinstance(v, list)},
            "strict_first_failure": g["strict_first_failure"],
        }
        for mode, g in groups.items()
    }
    comparisons = {}
    for mode in modes:
        comparisons[mode] = {
            outcome: contrast(groups[chosen][outcome], groups[mode][outcome])
            for outcome in ("work", "strict", "collision")
        }
    rule = protocol["criterion"]
    passed = count >= rule["minimum_pairs"] and all(
        comparisons[arm]["work"]["paired_gain"]
        >= rule["min_absolute_success_gain_over_noop_and_fixed_slow"]
        and comparisons[arm]["collision"]["paired_gain"] <= rule["max_collision_rate_increase"]
        for arm in ("no_intervention", "edge_slow_rule")
    )
    native_summary = {}
    for name, group in native["groups"].items():
        require([r["case_id"] for r in group["trials"]] == case_ids, "native pairs differ")
        native_summary[name] = {
            key: sum(r[key] for r in group["trials"])
            for key in ("work_stance_reached", "body_collision")
        }
    return {
        "pairs": count,
        "criterion_passed": passed,
        "native_ppo_19_96_s": native_summary,
        "articulated_127_75_s": summary,
        "paired": comparisons,
    }, {c["joint_seed"] for c in cases}


def check_trial_video(directory, freeze, *, valve_view=False):
    folder = directory / "articulated"
    heatmap = directory / "heatmap"
    first_case = read(folder / "holdout-cases.json")[0]["case_id"]
    (trial,) = [
        row
        for row in read(folder / "holdout.json")["trials"]
        if row["case_id"] == first_case and row["candidate_id"] == freeze["adopted_candidate"]
    ]
    clip = "agent-holdout-valve" if valve_view else "agent-holdout"
    bill = read(heatmap / f"{clip}.receipt.json")
    require(
        bill["case_id"] == first_case
        and bill["candidate_id"] == trial["candidate_id"]
        and bill["state_sha256"] == trial["state_sha256"]
        and bill["scene_sha256"] == freeze["scene_sha256"]
        and bill["selection_sha256"] == file_sha(directory / "selection-freeze.json")
        and bill["reference_sha256"] == read(folder / "holdout-strict.json")["reference_sha256"],
        "video was not rendered from the frozen choice's first held-out physical trial",
    )
    for suffix, field in (
        ("mp4", "video_sha256"),
        ("ass", "subtitle_sha256"),
        ("surface.json", "surface_sha256"),
        ("links.json", "links_sha256"),
    ):
        require(
            file_sha(heatmap / f"{clip}.{suffix}") == bill[field],
            "changed movie or metadata",
        )
    indices = bill["frame_state_indices"]
    frames = (heatmap / f"{clip}.ass").read_text(encoding="utf-8-sig")
    first_index = round(bill["start_s"] / freeze["clock_s"]["physics"]) - 1 if valve_view else 0
    require(
        len(indices) == sum(line.startswith("Dialogue: 0,") for line in frames.splitlines())
        and indices == sorted(set(indices))
        and indices[0] == first_index
        and indices[-1] <= bill["terminal_state_index"] == bill["state_count"] - 1
        and math.isclose(bill["end_s"], trial["simulated_s"], abs_tol=1e-6)
        and math.isclose(bill["final_valve_deg"], trial["final_valve_angle_deg"], abs_tol=1e-4),
        "movie captions, states and terminal measurements differ",
    )
    surface = read(heatmap / f"{clip}.surface.json")
    require(surface["added_overlay_geoms"] == 0, "video uses added geometry")
    if not valve_view and bill["end_s"] > 67:
        require(
            len(surface["samples"]) == 2
            and surface["visible_mean_change_mpa"] > 0
            and all(
                s["mesh_pixel_count"] > 0
                and (
                    s["unique_20mpa_colour_bands"] >= 2
                    if "unique_20mpa_colour_bands" in s
                    else s["visible_max_mpa"] > s["visible_min_mpa"]
                )
                for s in surface["samples"]
            ),
            "moving chassis surface load was not resolved",
        )
    return {
        "case_id": first_case,
        "frames": len(indices),
        "state_sha256": bill["state_sha256"],
        "video_sha256": bill["video_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    directory = args.evidence
    protocol = read(directory / "protocol.json")
    freeze = read(directory / "selection-freeze.json")
    require(freeze["protocol_sha256"] == file_sha(directory / "protocol.json"), "changed protocol")
    bundle = directory / "frozen-ppo" / "t1-v8-cpu-9100"
    for name, expected in freeze["ppo_bundle_hashes"].items():
        require(file_sha(bundle / name) == expected, f"changed frozen CPU bundle file {name}")
    decision_file = directory / freeze["decision_relpath"]
    require(file_sha(decision_file) == freeze["decision_sha256"], "changed agent proposal")
    decision = read(decision_file)
    require(
        decision["adopted_candidate"] == freeze["adopted_candidate"], "human substituted choice"
    )
    dev = read(directory / "articulated/development.json")["trials"]

    def dev_order(mode):
        rows = [r for r in dev if r["candidate_id"] == mode]
        return (
            -sum(r["work_stance_reached"] for r in rows),
            sum(r["body_collision"] for r in rows),
            mode,
        )

    best = min(freeze["evaluated_modes"], key=dev_order)
    require(best == freeze["development_best_fixed"], "development comparator changed")
    results = {}
    used = set()
    for split in ("holdout", "confirmation"):
        results[split], seeds = inspect_cohort(directory, split, protocol, freeze)
        require(not used.intersection(seeds), "case seeds overlap between test sets")
        used.update(seeds)
    heatmap = directory / "heatmap"
    surfaces = read(heatmap / "link-stress.json")
    for filename in ("body-stress-follow.mp4", "body-stress-valve.mp4"):
        require(
            (heatmap / filename).stat().st_size > 10000
            and file_sha(heatmap / filename) in json.dumps(surfaces),
            "uncertified heatmap movie",
        )
    result = {
        "criterion_passed": all(r["criterion_passed"] for r in results.values()),
        "adopted_candidate": freeze["adopted_candidate"],
        "best_fixed": best,
        "heatmap_verified_against_metadata": True,
        "selected_trial_video": check_trial_video(directory, freeze),
        "selected_trial_valve_video": check_trial_video(directory, freeze, valve_view=True),
        "cohorts": results,
    }
    if args.output:
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write("\n")
    print(
        "PASS" if result["criterion_passed"] else "FAIL",
        {
            k: v["articulated_127_75_s"][freeze["adopted_candidate"]]["strict"]
            for k, v in results.items()
        },
    )
    return 0 if result["criterion_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
