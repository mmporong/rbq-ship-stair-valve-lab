"""Compare inherited and one-thread CPU replant condition runs without mixing tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "src")]

import assess_stair_conditions as assess  # noqa: E402

CASES = (
    "nominal",
    "floor_mu_050",
    "floor_mu_080",
    "resistance_5",
    "resistance_15",
    "sensitivity_lever_m_0_1",
    "sensitivity_lever_m_0_14",
    "sensitivity_pad_mu_0_6",
    "sensitivity_pad_mu_1",
    "sensitivity_finger_force_limit_n_75",
    "sensitivity_finger_slide_frictionloss_n_15",
    "sensitivity_arm_limits_named_lims_ex_joint_limits",
    "sensitivity_timestep_s_0_0025",
    "sensitivity_impratio_10",
)
LABELS = {
    "nominal": "기준 조건",
    "floor_mu_050": "발–계단 마찰 0.50",
    "floor_mu_080": "발–계단 마찰 0.80",
    "resistance_5": "밸브 저항 5 N·m",
    "resistance_15": "밸브 저항 15 N·m",
    "sensitivity_lever_m_0_1": "레버 0.10 m",
    "sensitivity_lever_m_0_14": "레버 0.14 m",
    "sensitivity_pad_mu_0_6": "패드 마찰 0.6",
    "sensitivity_pad_mu_1": "패드 마찰 1.0",
    "sensitivity_finger_force_limit_n_75": "손가락 힘 한계 75 N",
    "sensitivity_finger_slide_frictionloss_n_15": "손가락 미끄럼 손실 15 N",
    "sensitivity_arm_limits_named_lims_ex_joint_limits": "팔 관절 한계 LIMS-EX",
    "sensitivity_timestep_s_0_0025": "물리 간격 2.5 ms",
    "sensitivity_impratio_10": "접촉 임피던스 비 10",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(root: Path, case_id: str):
    directory = root / case_id
    result, scene = load(directory / "result.json"), load(directory / "scene.json")
    with np.load(directory / "mission.npz", allow_pickle=False) as raw:
        endpoint = float(raw["time_s"][-1])
        sample_count = int(len(raw["time_s"]))
    first = result.get("first_failure")
    return {
        "outcome": result.get("outcome"),
        "m1_passed": bool((result.get("valve") or {}).get("verified")),
        "n1_passed": bool((result.get("walking") or {}).get("verified")),
        "first_failure": assess._first_failure_summary(result),
        "archived_result_first_failure": first,
        "fell": bool((result.get("metrics") or {}).get("fell")),
        "landings_evaluated": (result.get("walking") or {}).get("landings_evaluated"),
        "raw_endpoint_s": endpoint,
        "raw_sample_count": sample_count,
        "raw_sha256": result["raw_sha256"],
        "scene_sha256": result["scene_sha256"],
        "source_sha256": scene["source_sha256"],
        "planner_source_sha256": scene["planner_source_sha256"],
        "walking_reference_sha256": scene["walking_reference"]["sha256"],
        "settings": scene["settings"],
        "timestep_s": scene["timestep_s"],
        "actuation_period_s": scene["actuation_period_s"],
        "impratio": scene["impratio"],
        "arm_limits": scene["arm_limits"],
        "control_profile": scene["control_profile"],
    }


def format_delta(metric, value):
    if value is None:
        return "-"
    if metric and metric.endswith("_m"):
        return f"{value * 1000:+.6f} mm"
    if metric and metric.endswith("_s"):
        return f"{value:+.6f} s"
    if metric and metric.endswith("_deg"):
        return f"{value:+.6f}°"
    return f"{value:+.9g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inherited-root", type=Path, required=True)
    parser.add_argument("--one-thread-root", type=Path, required=True)
    args = parser.parse_args()
    inherited_root = args.inherited_root.resolve()
    one_thread_root = args.one_thread_root.resolve()
    rows = []
    for case_id in CASES:
        inherited = summarize(inherited_root, case_id)
        one_thread = summarize(one_thread_root, case_id)
        fixed_keys = (
            "source_sha256",
            "planner_source_sha256",
            "walking_reference_sha256",
            "settings",
            "timestep_s",
            "actuation_period_s",
            "impratio",
            "arm_limits",
            "control_profile",
        )
        same_inputs = all(inherited[key] == one_thread[key] for key in fixed_keys)
        if not same_inputs:
            raise ValueError(f"{case_id}: source or command inputs differ")
        inherited_first = inherited.get("first_failure") or {}
        one_thread_first = one_thread.get("first_failure") or {}
        rows.append(
            {
                "id": case_id,
                "same_frozen_source_and_inputs": same_inputs,
                "raw_byte_equal": inherited["raw_sha256"] == one_thread["raw_sha256"],
                "verdict_equal": all(
                    inherited[key] == one_thread[key]
                    for key in ("outcome", "m1_passed", "n1_passed")
                ),
                "first_failure_metric_equal": inherited_first.get("metric")
                == one_thread_first.get("metric"),
                "first_failure_time_delta_s": (
                    one_thread_first.get("time_s") - inherited_first.get("time_s")
                    if isinstance(one_thread_first.get("time_s"), (int, float))
                    and isinstance(inherited_first.get("time_s"), (int, float))
                    else None
                ),
                "first_failure_measured_delta": (
                    one_thread_first.get("measured") - inherited_first.get("measured")
                    if isinstance(one_thread_first.get("measured"), (int, float))
                    and isinstance(inherited_first.get("measured"), (int, float))
                    else None
                ),
                "inherited_environment": inherited,
                "one_thread_environment": one_thread,
            }
        )
    report = {
        "schema": 1,
        "scope": "comparison only; published 17-row table uses one-thread_environment",
        "environments": {
            "inherited": {
                "OPENBLAS_NUM_THREADS": None,
                "MKL_NUM_THREADS": None,
                "OMP_NUM_THREADS": None,
                "CUDA_VISIBLE_DEVICES": None,
            },
            "one_thread": {
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "-1",
            },
        },
        "rows": rows,
        "counts": {
            "raw_byte_equal": sum(row["raw_byte_equal"] for row in rows),
            "raw_byte_different": sum(not row["raw_byte_equal"] for row in rows),
            "verdict_equal": sum(row["verdict_equal"] for row in rows),
            "verdict_different": sum(not row["verdict_equal"] for row in rows),
        },
        "raw_files_copied": 0,
    }
    (HERE / "comparison-default-vs-one-thread.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 기본 환경과 1-thread 환경 비교",
        "",
        (
            "발표용 17행은 1-thread CPU 환경만 사용한다. 아래 표는 같은 frozen "
            "source·명령에서 실행 환경에 따른 바이트·수치·판정 차이를 분리해 기록한다."
        ),
        "",
        (
            "| 조건 | raw 동일 | 판정 동일 | 기본 최초 위반 | 1-thread 최초 위반 | "
            "시각 차이 | 측정값 차이 |"
        ),
        "|---|---:|---:|---|---|---:|---:|",
    ]
    for row in rows:
        inherited = row["inherited_environment"].get("first_failure") or {}
        one_thread = row["one_thread_environment"].get("first_failure") or {}
        delta = row["first_failure_time_delta_s"]
        delta_text = "-" if delta is None else f"{delta:+.6f} s"
        value_delta = format_delta(one_thread.get("metric"), row["first_failure_measured_delta"])
        lines.append(
            f"| {LABELS[row['id']]} | {'같음' if row['raw_byte_equal'] else '다름'} | "
            f"{'같음' if row['verdict_equal'] else '다름'} | "
            f"{inherited.get('metric', '-')} @ {inherited.get('time_s', 0):.3f} s | "
            f"{one_thread.get('metric', '-')} @ {one_thread.get('time_s', 0):.3f} s | "
            f"{delta_text} | {value_delta} |"
        )
    lines += [
        "",
        "원시는 두 `.omx` run-root에 보존하며 evidence 디렉터리에 복사하지 않았다.",
        "",
    ]
    (HERE / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
