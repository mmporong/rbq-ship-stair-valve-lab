"""Freeze, replay, verify, and assess the source-matched replant conditions."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TOOLS = ROOT / "tools"
WALK_RELATIVE = Path("evidence/stair-corridor-20261121/local-repair/walk50.npz")
LOW_RUN_RELATIVE = Path("evidence/stair-corridor-20261121/tracked/low-control/loaded.json.gz")
WALK = ROOT / WALK_RELATIVE
LOW_RUN = ROOT / LOW_RUN_RELATIVE
COMMON_PATH = HERE.with_name("conditions-tracked") / "replay.py"
BASE_PROTOCOL = HERE.with_name("conditions-local") / "frozen-protocol.json"
PROTOCOL = HERE / "frozen-protocol.json"
PROFILE = "rl-mid-replant-k3075-v1"
SOURCE_COMMIT = "2c91d83ff3a222bc2919c3156b93920ebcff5ec4"
PINNED_CPU_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "-1",
}

SOURCE_FILES = [
    "tools/replay_stair_mission.py",
    "tools/stair_mission_control_profile.py",
    "tools/stair_mission_route.py",
    "tools/stair_mission_reference.py",
    "tools/stair_mission_trace.py",
    "tools/replay_stair_sequence.py",
    "src/go2_wall/grasp_motion.py",
    "src/go2_wall/stair_valve.py",
    "src/go2_wall/transition.py",
    "src/go2_wall/gripper_control.py",
    "src/go2_wall/stair_loads.py",
    "src/go2_wall/mission_verdict.py",
    "tools/stair_mission_limits.py",
    "src/go2_wall/stair_walking_control.py",
]


def load_common():
    spec = importlib.util.spec_from_file_location("tracked_condition_replay", COMMON_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def freeze_protocol() -> dict:
    protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    protocol["physics"]["timestep_s"] = 0.001
    protocol["nominal_round_trip"]["landings"] = 157
    protocol["nominal_round_trip"]["original_landings"] = 156
    protocol["nominal_round_trip"]["supplementary_landings"] = 1
    protocol["evaluation_scope"] = {
        "physical_rows": 14,
        "planning_rows": 3,
        "extent": "through-task",
        "walking_reference": WALK.relative_to(ROOT).as_posix(),
        "walking_reference_sha256": sha256(WALK),
        "low_load_control": LOW_RUN.relative_to(ROOT).as_posix(),
        "low_load_control_sha256": sha256(LOW_RUN),
        "control_profile": PROFILE,
        "nominal_endpoint_s": 135.715,
        "expected_landings_at_endpoint": 20,
        "original_scheduled_landings": 156,
        "supplementary_landings": 1,
    }
    protocol["control_profile"] = {
        "name": PROFILE,
        "replant_after_landing_index": 15,
        "replant_trigger_s": 78.8,
        "replant_duration_s": 2.8,
        "replant_leg": "RL",
        "slip_spring_gain_n_per_m": 3075,
        "slip_spring_active_s": [116.65, 129.4],
        "stance_spring_gain_n_per_m": 2500,
        "terminal_overtravel_deg": 1.3,
    }
    protocol["frozen_cli"] = {
        "common": [
            "--walk",
            WALK.relative_to(ROOT).as_posix(),
            "--extent",
            "through-task",
            "--corridor-feedback",
            "--control-profile",
            PROFILE,
            "--low-run",
            LOW_RUN.relative_to(ROOT).as_posix(),
        ],
        "baseline_timestep_s": 0.001,
        "timestep_sensitivity_s": 0.0025,
    }
    protocol["execution_environment"] = {
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "-1",
    }
    protocol["source_commit"] = SOURCE_COMMIT
    protocol["source_file_sha256"] = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    protocol["source_sha256"] = {name: source_sha256_lf(ROOT / name) for name in SOURCE_FILES}
    PROTOCOL.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return protocol


def validate_source_root(source_root: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    mismatches = []
    for name, expected in protocol["source_sha256"].items():
        path = source_root / name
        actual = source_sha256_lf(path) if path.exists() else None
        if actual != expected:
            mismatches.append({"file": name, "expected": expected, "actual": actual})
    if mismatches:
        raise SystemExit(
            "source root differs from the frozen controller sources:\n"
            + json.dumps(mismatches, indent=2)
        )
    input_mismatches = []
    for name, expected in (
        (WALK_RELATIVE, protocol["evaluation_scope"]["walking_reference_sha256"]),
        (LOW_RUN_RELATIVE, protocol["evaluation_scope"]["low_load_control_sha256"]),
    ):
        path = source_root / name
        actual = sha256(path) if path.exists() else None
        if actual != expected:
            input_mismatches.append(
                {"file": name.as_posix(), "expected": expected, "actual": actual}
            )
    if input_mismatches:
        raise SystemExit(
            "source root differs from the frozen binary inputs:\n"
            + json.dumps(input_mismatches, indent=2)
        )
    assets = source_root / ".assets/rbq"
    if not assets.is_dir() or sum(path.is_file() for path in assets.rglob("*")) < 18:
        raise SystemExit(f"source root lacks the prepared 18-file RBQ asset tree: {assets}")


def case_command(common, source_root: Path, run_root: Path, case_id: str) -> list[str]:
    return [
        sys.executable,
        str(source_root / "tools/replay_stair_mission.py"),
        str(run_root / case_id),
        "--walk",
        str(source_root / WALK_RELATIVE),
        "--extent",
        "through-task",
        "--corridor-feedback",
        "--control-profile",
        PROFILE,
        "--low-run",
        str(source_root / LOW_RUN_RELATIVE),
        *common.CASES[case_id],
    ]


def run_replay(command: list[str], cpu_environment: str, source_root: Path) -> None:
    environment = os.environ.copy()
    if cpu_environment == "one-thread-cpu":
        environment.update(PINNED_CPU_ENVIRONMENT)
    python_paths = [str(source_root / "tools"), str(source_root / "src")]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=source_root, env=environment, check=True)


def replay(common, source_root: Path, run_root: Path, jobs: int, cpu_environment: str) -> None:
    validate_source_root(source_root)
    run_root.mkdir(parents=True, exist_ok=True)
    environment_path = run_root / "execution-environment.json"
    environment_record = {
        "mode": cpu_environment,
        "variables": (
            PINNED_CPU_ENVIRONMENT
            if cpu_environment == "one-thread-cpu"
            else {key: os.environ.get(key) for key in PINNED_CPU_ENVIRONMENT}
        ),
    }
    if (
        environment_path.exists()
        and json.loads(environment_path.read_text(encoding="utf-8")) != environment_record
    ):
        raise SystemExit(f"run root uses a different execution environment: {run_root}")
    environment_path.write_text(json.dumps(environment_record, indent=2) + "\n", encoding="utf-8")
    commands = []
    for case_id in common.CASES:
        output = run_root / case_id
        if (output / "result.json").exists():
            print(f"skip complete replay: {case_id}", flush=True)
            continue
        if output.exists():
            raise SystemExit(f"incomplete output exists; inspect without deleting: {output}")
        commands.append(case_command(common, source_root, run_root, case_id))
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(run_replay, command, cpu_environment, source_root) for command in commands
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def assess(common, run_root: Path) -> None:
    """Run the official aggregate and add last-sample scenes for successful rows."""
    common.assess(run_root, HERE)
    aggregate_path = run_root / "assessment/official-aggregate.json"
    report = json.loads(aggregate_path.read_text(encoding="utf-8"))
    sys.path[:0] = [str(TOOLS), str(ROOT / "src")]
    import assess_stair_conditions as condition_assessor

    scene_root = run_root / "assessment/scenes"
    for row in report["conditions"]:
        if row["outcome"] != "success":
            continue
        result_path = run_root / row["id"] / "result.json"
        with np.load(result_path.with_name("mission.npz"), allow_pickle=False) as raw:
            last_time = float(raw["time_s"][-1])
        output = scene_root / f"{row['id']}.png"
        snapshot = condition_assessor.render_physical_snapshot(result_path, last_time + 1.0, output)
        row["snapshot"] = {
            "type": "successful last actual frame",
            **snapshot,
            "path": output.relative_to(run_root).as_posix(),
        }
    aggregate_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify(common, run_root: Path) -> None:
    """Freshly verify all raw traces with the current compatible validator."""
    output_root = run_root / "verification-canonical"
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fresh-", dir=output_root) as temp:
        temp_root = Path(temp)
        for case_id in common.CASES:
            source = run_root / case_id
            if not (source / "result.json").exists():
                raise SystemExit(f"missing completed replay: {source}")
            fresh = temp_root / f"{case_id}.json"
            common.run(
                [
                    sys.executable,
                    str(TOOLS / "verify_stair_replay.py"),
                    str(source),
                    str(fresh),
                    "--walk",
                    str(WALK),
                ]
            )
            output = output_root / fresh.name
            if output.exists():
                if json.loads(output.read_text(encoding="utf-8")) != json.loads(
                    fresh.read_text(encoding="utf-8")
                ):
                    raise SystemExit(f"fresh canonical verification differs from cache: {case_id}")
                print(f"fresh canonical verification matches cache: {case_id}")
            else:
                fresh.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "status", "replay", "verify", "assess", "all"))
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / ".omx/mission-success-20261121/conditions-replant",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT,
        help="checkout/archive root whose controller sources must match frozen SHA-256",
    )
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument(
        "--cpu-environment",
        choices=("inherit", "one-thread-cpu"),
        default="inherit",
        help="inherit the launcher environment or pin BLAS/OpenMP to one thread and disable CUDA",
    )
    args = parser.parse_args()
    common = load_common()
    run_root = args.run_root.resolve()
    source_root = args.source_root.resolve()
    if args.action == "freeze":
        print(json.dumps(freeze_protocol()["evaluation_scope"], indent=2))
        return
    if not PROTOCOL.exists():
        raise SystemExit("missing frozen-protocol.json; run freeze before measurement")
    if args.action == "status":
        completed = sum((run_root / case_id / "result.json").exists() for case_id in common.CASES)
        print(json.dumps({"completed": completed, "required": 14, "jobs": args.jobs}, indent=2))
        return
    if args.action in ("replay", "all"):
        default_root = (ROOT / ".omx/mission-success-20261121/conditions-replant").resolve()
        if args.cpu_environment == "one-thread-cpu" and run_root == default_root:
            raise SystemExit("one-thread-cpu requires a separate --run-root")
        replay(common, source_root, run_root, args.jobs, args.cpu_environment)
    if args.action in ("verify", "all"):
        verify(common, run_root)
    if args.action in ("assess", "all"):
        assess(common, run_root)


if __name__ == "__main__":
    main()
