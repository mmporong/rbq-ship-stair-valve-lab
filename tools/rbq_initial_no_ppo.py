"""Run the preregistered initial-theta, PPO-off residual-policy control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rbq_articulated_control import digest  # noqa: E402
from triage_rbq_host_ab import canonical_sha256, validate_cases  # noqa: E402

from train.rbq_reward_residual import (  # noqa: E402
    FULL_HORIZON_S,
    INITIAL_PARAMETERS,
    evaluate_batch,
)


def initial_no_ppo_job() -> tuple[str, np.ndarray, str]:
    """Use the initial contact gate with the existing PPO-off execution branch."""
    return "initial_no_ppo", INITIAL_PARAMETERS.copy(), "learned_no_ppo"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "reference", "bundle", "cases", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--expected-cases-sha256", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    cases = validate_cases(json.loads(args.cases.read_text(encoding="utf-8")))
    case_sha256 = canonical_sha256(cases)
    if case_sha256 != args.expected_cases_sha256:
        parser.error("case commitment mismatch")
    if args.output.exists():
        parser.error("output already exists")
    trials = evaluate_batch(
        args.scene,
        args.reference,
        args.bundle,
        cases,
        [initial_no_ppo_job()],
        FULL_HORIZON_S,
        args.workers,
        args.state_dir,
    )
    result = {
        "schema": "rbq_initial_no_ppo_evaluation_v1",
        "case_set_sha256": case_sha256,
        "scene_sha256": digest(args.scene),
        "reference_sha256": digest(args.reference),
        "source_sha256": digest(Path(__file__)),
        "frozen_reward_policy_source_sha256": digest(
            Path(__file__).parents[1] / "train" / "rbq_reward_residual.py"
        ),
        "parameters": INITIAL_PARAMETERS.tolist(),
        "ppo_authority_forced": 0.0,
        "underlying_variant": "learned_no_ppo",
        "horizon_s": FULL_HORIZON_S,
        "trials": sorted(trials, key=lambda row: row["case_id"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
