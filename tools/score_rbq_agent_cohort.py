"""Reconstruct full physical verdicts for the paired articulated cohort."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rbq_articulated_control import digest
from rbq_articulated_verdict import score_state_file


def score_one(scene, ref, walk, state_dir, row):
    case, mode = row["case_id"], row["candidate_id"]
    state = state_dir / f"{case}--{mode}.npz"
    if digest(state) != row["state_sha256"]:
        raise ValueError(f"Changed physics evidence for {case}/{mode}")
    if row["termination"] in {"terrain_body_contact", "fall"}:
        # The recorded terminal collision/fall ended the only physical trajectory.
        return dict(
            case_id=case,
            candidate_id=mode,
            state_sha256=row["state_sha256"],
            strict_success=False,
            complete_horizon=False,
            outcome="early_termination",
            first_failure=row["termination"],
            simulated_s=row["simulated_s"],
            maximum_angle_deg=row["maximum_valve_angle_deg"],
        )
    result = score_state_file(scene, state, ref, walk)
    result.update(case_id=case, candidate_id=mode)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "reference", "walk", "study", "states", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    study = json.loads(args.study.read_bytes())
    if args.output.exists() or digest(args.scene) != study["scene_sha256"]:
        parser.error("output exists or scene commitment mismatch")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        work = [
            pool.submit(score_one, args.scene, args.reference, args.walk, args.states, row)
            for row in study["trials"]
        ]
        for future in as_completed(work):
            row = future.result()
            rows.append(row)
            print(row["case_id"], row["candidate_id"], row["strict_success"], flush=True)
    output = {
        "study_sha256": digest(args.study),
        "reference_sha256": digest(args.reference),
        "walk_sha256": digest(args.walk),
        "scene_sha256": digest(args.scene),
        "scoring_source_sha256": digest(Path(__file__).with_name("rbq_articulated_verdict.py")),
        "trials": sorted(rows, key=lambda r: (r["case_id"], r["candidate_id"])),
    }
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
