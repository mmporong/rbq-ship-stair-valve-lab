"""Reproduce, independently verify, and assess the tracked T4 condition table.

Raw runs stay outside this evidence directory. The default walk input is
``evidence/stair-corridor-20261121/tracked/walk50.npz``. A typical fresh-clone
invocation (the run root may also be outside the repository) is:

    python evidence/stair-corridor-20261121/conditions-tracked/replay.py all \
        --run-root .local/t4-conditions-tracked
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TOOLS = ROOT / "tools"
WALK = ROOT / "evidence/stair-corridor-20261121/tracked/walk50.npz"
PLANNING = {
    "slope_60": ROOT / "evidence/stair-mission-20261121/conditions/v2/condition-60.failure.json",
    "rise_165": ROOT / "evidence/stair-mission-20261121/conditions/v2/condition-165.failure.json",
    "rise_200": ROOT / "evidence/stair-mission-20261121/conditions/v2/condition-200.failure.json",
}

CASES = {
    "nominal": [],
    "floor_mu_050": ["--floor-mu", "0.5"],
    "floor_mu_080": ["--floor-mu", "0.8"],
    "resistance_5": ["--case", "resistance_5"],
    "resistance_15": ["--case", "resistance_15"],
    "sensitivity_lever_m_0_1": ["--handle-length-m", "0.1"],
    "sensitivity_lever_m_0_14": ["--handle-length-m", "0.14"],
    "sensitivity_pad_mu_0_6": ["--pad-mu", "0.6"],
    "sensitivity_pad_mu_1": ["--pad-mu", "1.0"],
    "sensitivity_finger_force_limit_n_75": ["--finger-force-n", "75"],
    "sensitivity_finger_slide_frictionloss_n_15": ["--slide-loss-n", "15"],
    "sensitivity_arm_limits_named_lims_ex_joint_limits": ["--arm-limits", "lims-ex"],
    "sensitivity_timestep_s_0_0025": ["--timestep", "0.0025"],
    "sensitivity_impratio_10": ["--impratio", "10"],
}

BASE_ARGS = [
    "--extent",
    "through-task",
    "--corridor-feedback",
    "--motion-seconds",
    "9",
    "--jaw-load-bias-m",
    "0.0005",
    "--stance-spring-gain-n-per-m",
    "2500",
    "--terminal-overtravel-deg",
    "1.2",
    "--timestep",
    "0.001",
]


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def replay(run_root: Path, walk: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    for case_id, overrides in CASES.items():
        output = run_root / case_id
        if (output / "result.json").exists():
            print(f"skip complete replay: {case_id}", flush=True)
            continue
        if output.exists():
            raise SystemExit(f"incomplete output exists; inspect without deleting: {output}")
        args = ["--walk", str(walk)] + list(BASE_ARGS)
        if "--timestep" in overrides:
            index = args.index("--timestep")
            del args[index : index + 2]
        run(
            [sys.executable, str(TOOLS / "replay_stair_mission.py"), str(output)] + args + overrides
        )


def verify(run_root: Path, walk: Path) -> None:
    verification_root = run_root / "verification"
    verification_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fresh-", dir=verification_root) as temp:
        temp_root = Path(temp)
        for case_id in CASES:
            source = run_root / case_id
            if not (source / "result.json").exists():
                raise SystemExit(f"missing completed replay: {source}")
            output = verification_root / f"{case_id}.json"
            fresh_output = temp_root / f"{case_id}.json"
            run(
                [
                    sys.executable,
                    str(TOOLS / "verify_stair_replay.py"),
                    str(source),
                    str(fresh_output),
                    "--walk",
                    str(walk),
                ]
            )
            fresh = json.loads(fresh_output.read_text(encoding="utf-8"))
            if output.exists():
                existing = json.loads(output.read_text(encoding="utf-8"))
                if existing != fresh:
                    raise SystemExit(f"fresh verification differs from cached result: {case_id}")
                print(f"fresh verification matches cache: {case_id}", flush=True)
            else:
                fresh_output.replace(output)


def assess(run_root: Path, evidence_root: Path) -> None:
    protocol = evidence_root / "frozen-protocol.json"
    if not protocol.exists():
        raise SystemExit(f"missing frozen protocol: {protocol}")
    assessment_root = run_root / "assessment"
    assessment_root.mkdir(parents=True, exist_ok=True)
    planning_root = assessment_root / "planning"
    planning_root.mkdir(parents=True, exist_ok=True)
    planning = {}
    for case_id, source in PLANNING.items():
        destination = planning_root / source.name
        shutil.copy2(source, destination)
        planning[case_id] = destination
    command = [
        sys.executable,
        str(TOOLS / "assess_stair_conditions.py"),
        str(protocol),
        "--output",
        str(assessment_root / "official-aggregate.json"),
        "--relative-to",
        str(run_root),
        "--snapshot-dir",
        str(assessment_root / "scenes"),
    ]
    for case_id in CASES:
        result = run_root / case_id / "result.json"
        if not result.exists():
            raise SystemExit(f"missing completed replay: {result.parent}")
        command += ["--run", f"{case_id}={result}"]
    for case_id, failure in planning.items():
        command += ["--run", f"{case_id}={failure}"]
    run(command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("replay", "verify", "assess", "all"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--walk", type=Path, default=WALK)
    parser.add_argument("--evidence-root", type=Path, default=HERE)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    walk = args.walk.resolve()
    evidence_root = args.evidence_root.resolve()
    if args.action in ("replay", "all"):
        replay(run_root, walk)
    if args.action in ("verify", "all"):
        verify(run_root, walk)
    if args.action in ("assess", "all"):
        assess(run_root, evidence_root)


if __name__ == "__main__":
    main()
