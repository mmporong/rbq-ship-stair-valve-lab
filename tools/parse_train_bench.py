"""Summarize tools/train_bench.sh output into one JSON file.

Usage: python tools/parse_train_bench.py <bench dir> [--skip N]

The first iterations include CUDA graph capture and kernel compilation, so
they are skipped before taking medians.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
FIELDS = {
    "steps_per_s": r"Steps per second:\s*([\d.]+)",
    "collection_s": r"Collection time:\s*([\d.]+)s",
    "learning_s": r"Learning time:\s*([\d.]+)s",
    "iteration_s": r"Iteration time:\s*([\d.]+)s",
}


def parse_log(text: str) -> list[dict]:
    """Split an RSL-RL console log into per-iteration values.

    >>> log = "Learning iteration 0/2\\nSteps per second: 10\\nCollection time: 1.0s\\n"
    >>> log += "Learning time: 0.5s\\nIteration time: 1.5s\\n"
    >>> parse_log(log)
    [{'steps_per_s': 10.0, 'collection_s': 1.0, 'learning_s': 0.5, 'iteration_s': 1.5}]
    """
    text = ANSI.sub("", text)
    blocks = re.split(r"Learning iteration \d+/\d+", text)[1:]
    rows = []
    for block in blocks:
        row = {}
        for key, pattern in FIELDS.items():
            match = re.search(pattern, block)
            if match:
                row[key] = float(match.group(1))
        if len(row) == len(FIELDS):
            rows.append(row)
    return rows


def gpu_stats(path: Path, idle_mib: float) -> dict:
    used, util = [], []
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            used.append(float(parts[0]))
            util.append(float(parts[1]))
    if not used:
        return {}
    return {
        "peak_used_mib": max(used),
        "peak_over_idle_mib": max(used) - idle_mib,
        "mean_util_pct": round(statistics.mean(util), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bench_dir", type=Path)
    parser.add_argument("--skip", type=int, default=5)
    args = parser.parse_args()
    d = args.bench_dir
    idle = d.joinpath("gpu_idle.csv").read_text().strip().split(",")
    idle_mib = float(idle[3].split()[0])
    result = {
        "gpu": idle[0].strip(),
        "driver": idle[1].strip(),
        "idle_used_mib": idle_mib,
        "versions": d.joinpath("versions.txt").read_text().strip(),
        "unitree_rl_mjlab_commit": d.joinpath("unitree_rl_mjlab_commit.txt").read_text().strip(),
        "skip_iterations": args.skip,
        "runs": {},
    }
    for log in sorted(d.glob("train_*.log"), key=lambda p: int(p.stem.split("_")[1])):
        n = int(log.stem.split("_")[1])
        rows = parse_log(log.read_text(errors="replace"))
        kept = rows[args.skip :]
        run = {"iterations_logged": len(rows)}
        if kept:
            run.update({key: round(statistics.median(r[key] for r in kept), 3) for key in FIELDS})
            run["steps_per_iteration"] = n * 24
        run.update(gpu_stats(d / f"gpu_{n}.csv", idle_mib))
        result["runs"][str(n)] = run
    out = d / "summary.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
