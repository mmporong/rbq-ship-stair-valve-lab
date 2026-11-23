"""Validate and publish the source-matched replant 17-condition evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TOOLS = ROOT / "tools"
sys.path[:0] = [str(TOOLS), str(ROOT / "src")]

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
PLANNING = {
    "slope_60": ROOT / "evidence/stair-mission-20261121/conditions/v2/condition-60.failure.json",
    "rise_165": ROOT / "evidence/stair-mission-20261121/conditions/v2/condition-165.failure.json",
    "rise_200": ROOT / "evidence/stair-mission-20261121/conditions/v2/condition-200.failure.json",
}
LABELS = {
    "nominal": "기준 조건",
    "slope_60": "선박형 경사 60°",
    "rise_165": "단 높이 165 mm",
    "rise_200": "단 높이 200 mm",
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


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def failure_summary(payload):
    return assess._first_failure_summary(payload)


def validate_profile(scene, protocol, case_id):
    expected = protocol["control_profile"]
    observed = scene.get("control_profile") or {}
    checks = {
        "name": expected["name"],
        "trigger_landing_index": expected["replant_after_landing_index"],
        "trigger_time_s": expected["replant_trigger_s"],
        "duration_s": expected["replant_duration_s"],
        "replant_leg": expected["replant_leg"],
        "slip_spring_gain_n_per_m": expected["slip_spring_gain_n_per_m"],
        "slip_spring_active_s": expected["slip_spring_active_s"],
        "stance_spring_gain_n_per_m": expected["stance_spring_gain_n_per_m"],
        "terminal_overtravel_deg": expected["terminal_overtravel_deg"],
    }
    for key, value in checks.items():
        if observed.get(key) != value:
            raise ValueError(f"{case_id}: control profile {key} differs")


def physical_row(case_id, run_root, protocol, protocol_row, official_row):
    directory = run_root / case_id
    result_path = directory / "result.json"
    scene_path = directory / "scene.json"
    raw_path = directory / "mission.npz"
    result, scene = load(result_path), load(scene_path)
    if sha256(scene_path) != result["scene_sha256"] or sha256(raw_path) != result["raw_sha256"]:
        raise ValueError(f"{case_id}: result fingerprints differ")
    if scene["source_sha256"] != protocol["source_sha256"]:
        raise ValueError(f"{case_id}: controller source differs from frozen source")
    assess.observed_settings(protocol, protocol_row, result_path, result)
    validate_profile(scene, protocol, case_id)
    scope = protocol["evaluation_scope"]
    if scene["walking_reference"]["sha256"] != scope["walking_reference_sha256"]:
        raise ValueError(f"{case_id}: walking reference differs")
    expected_dt = 0.0025 if case_id == "sensitivity_timestep_s_0_0025" else 0.001
    if scene["timestep_s"] != expected_dt:
        raise ValueError(f"{case_id}: timestep differs")
    with np.load(raw_path, allow_pickle=False) as raw:
        times = raw["time_s"]
        if len(times) < 2 or not np.isfinite(times).all():
            raise ValueError(f"{case_id}: invalid raw clock")
        if not math.isclose(float(np.max(np.diff(times))), expected_dt, abs_tol=2e-11):
            raise ValueError(f"{case_id}: raw timestep differs")
        for key in ("qpos", "qvel", "ctrl", "stance_slip_m", "fell", "qp_attempt"):
            if key not in raw.files or raw[key].shape[0] != len(times):
                raise ValueError(f"{case_id}: malformed {key}")
        raw_endpoint_s = float(times[-1])
    walking = result.get("walking") or {}
    valve = result.get("valve") or {}
    snapshot = official_row.get("snapshot")
    if not snapshot:
        raise ValueError(f"{case_id}: official aggregate lacks a scene")
    first = failure_summary(result)
    archived_result_first = result.get("first_failure")
    special_failures = {}
    if case_id == "sensitivity_impratio_10":
        replant = next(
            failure
            for failure in walking.get("failures", [])
            if failure.get("metric") == "physical_replant"
        )
        special_failures = {
            "archived_result_first_failure": archived_result_first,
            "chronological_first_failure": first,
            "control_profile_failure": replant,
            "additional_snapshots": [
                {
                    "type": "physical replant failure",
                    "time_s": replant["time_s"],
                    "path": "scenes/sensitivity_impratio_10-replant.png",
                }
            ],
        }
        snapshot = {
            "type": "chronological first physical failure",
            "time_s": first["time_s"],
        }
    return {
        "id": case_id,
        "label": LABELS[case_id],
        "group": protocol_row["group"],
        "changes": protocol_row["changes"],
        "execution": "physical",
        "scope": result["extent"],
        "physics_timestep_s": scene["timestep_s"],
        "actuation_period_s": scene["actuation_period_s"],
        "impratio": scene["impratio"],
        "walking_reference_sha256": scene["walking_reference"]["sha256"],
        "control_profile": scene["control_profile"],
        "raw_endpoint_s": raw_endpoint_s,
        "m1": {"passed": bool(valve.get("verified")), "first_failure": valve.get("first_failure")},
        "n1": {
            "passed": bool(walking.get("verified")),
            "landings_evaluated": walking.get("landings_evaluated"),
            "max_stance_slip_mm": max(
                (result.get("metrics") or {}).get("max_stance_slip_mm", {}).values(),
                default=None,
            ),
            "first_failure": walking.get("first_failure"),
        },
        "outcome": official_row["outcome"],
        "first_failure": first,
        "snapshot": {**snapshot, "path": f"scenes/{case_id}.png"},
        "source": {
            "result": result_path.relative_to(ROOT).as_posix(),
            "scene": scene_path.relative_to(ROOT).as_posix(),
            "raw": raw_path.relative_to(ROOT).as_posix(),
            "result_sha256": sha256(result_path),
            "scene_sha256": result["scene_sha256"],
            "raw_sha256": result["raw_sha256"],
        },
        **special_failures,
    }


def planning_row(case_id, path, protocol_row, official_row):
    payload = load(path)
    first = payload["first_failure"]
    if first.get("unexpected_overlaps"):
        metric = "unexpected_overlap_m"
        measured = max(item["penetration_m"] for item in first["unexpected_overlaps"])
        limit = first.get("collision_tolerance_m")
    else:
        metric = "max_foot_ik_error_m"
        measured = max(first["foot_ik_error_m"].values())
        limit = first.get("foot_ik_limit_m")
    return {
        "id": case_id,
        "label": LABELS[case_id],
        "group": protocol_row["group"],
        "changes": protocol_row["changes"],
        "execution": "planning",
        "scope": "reference_generation",
        "m1": {"passed": None, "reached": False, "first_failure": None},
        "n1": {"passed": None, "reached": False, "landings_evaluated": 0},
        "outcome": official_row["outcome"],
        "first_failure": {
            "stage": f"segment_{first['segment']}_{first['kind']}",
            "time_s": first["planned_time_s"],
            "metric": metric,
            "measured": measured,
            "limit": limit,
        },
        "snapshot": {**official_row["snapshot"], "path": f"scenes/{case_id}.png"},
        "source": {
            "failure": path.relative_to(ROOT).as_posix(),
            "failure_sha256": sha256(path),
            "planner_source_sha256": payload["planner_source_sha256"],
        },
    }


def verdict(check):
    return "N/A" if check["passed"] is None else "PASS" if check["passed"] else "FAIL"


def format_value(metric, value):
    if not isinstance(value, (int, float)):
        return "-" if value is None else str(value)
    if metric and metric.endswith("_m"):
        return f"{value * 1000:.6f} mm"
    if metric and metric.endswith("_deg"):
        return f"{value:.3f}°"
    return f"{value:.6g}"


def format_limit(metric, limit):
    if isinstance(limit, (int, float)):
        return format_value(metric, limit)
    if not isinstance(limit, str):
        return str(limit)
    values = re.findall(r"[-+]?\d+(?:\.\d+)?", limit)
    if len(values) != 1:
        return limit.replace("|", "절댓값")
    operator = "≤ " if "<=" in limit else "≥ " if ">=" in limit else ""
    absolute = "절댓값 " if "abs" in limit else ""
    return f"{absolute}{operator}{format_value(metric, float(values[0]))}"


def format_failure(first):
    if not first:
        return "없음"
    metric = first.get("metric", "-")
    if metric == "physical_replant":
        time_s = first.get("time_s")
        time_text = f" @ {time_s:.3f} s" if isinstance(time_s, (int, float)) else ""
        return "physical_replant 실패 (무하중·양의 들기·35 mm 이내 재착지 조건)" + time_text
    text = f"{metric} {format_value(metric, first.get('measured'))}"
    if first.get("limit") is not None:
        text += f" (한계 {format_limit(metric, first['limit'])})"
    if isinstance(first.get("time_s"), (int, float)):
        text += f" @ {first['time_s']:.3f} s"
    return text


def write_results(rows, counts):
    nominal = next(row for row in rows if row["id"] == "nominal")
    impratio = next(row for row in rows if row["id"] == "sensitivity_impratio_10")
    lines = [
        "# 재디딤 제어 17조건 결과",
        "",
        (
            "`rl-mid-replant-k3075-v1` 제어를 같은 고정 입력에 적용한 CPU MuJoCo "
            "조건표다. 물리 14행은 바닥에서 작업 뒤 다음 착지까지의 `through-task` "
            "범위이며, 전체 157착지 일정의 완료 판정과 구분한다."
        ),
        "",
        (
            f"이 범위에서는 성공 {counts['success']}건, "
            f"물리 실패 {counts['physical_failed']}건, "
            f"계획 실패 {counts['planning_failed']}건."
        ),
        "",
        "| 조건 | 실행 | 정격 조작(M1) | N1 | 최초 위반 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        label = f"[{row['label']}]({row['snapshot']['path']})"
        lines.append(
            f"| {label} | {row['execution']} | {verdict(row['m1'])} | "
            f"{verdict(row['n1'])} | {format_failure(row.get('first_failure'))} |"
        )
    lines += [
        "",
        (
            f"- 스레드 고정 물리행은 성공 {counts['success']}/14다. nominal 자체가 "
            f"{format_failure(nominal.get('first_failure'))}로 실패하고 M1도 실패해 "
            "성공 기준점이 없으므로 이 14행으로 pass/fail 강건성 브래킷을 만들 수 없다. "
            "각 민감도 행은 최초 물리 위반 비교로만 사용한다."
        ),
        (
            "- 발표 표의 물리 14행은 OPENBLAS/MKL/OMP=1, CUDA_VISIBLE_DEVICES=-1로 "
            "고정했다. 환경변수 미설정 14행은 `COMPARISON.md`에 분리했으며 "
            "공식 행과 섞지 않았다."
        ),
        (
            "- geometry 3행은 물리 실행 전 참조 생성 실패다. M1/N1에 도달하지 않아 "
            "N/A이며 물리 14행의 분모에 포함하지 않는다."
        ),
        (
            "- 60° 행의 13.600초 몸통–3단 겹침은 고정 계획의 결과다. "
            "[`별도 개선 계획`](../ship60/report.json)은 이 충돌을 피한 뒤 "
            "38.060초 RL swing IK 2.927358 mm에서 중단됐으므로 두 결과를 섞지 않았다."
        ),
        (
            "- 임피던스비 10은 실제 시간순 최초 위반인 "
            f"{format_failure(impratio['chronological_first_failure'])}와 "
            f"재디딤 판정 {format_failure(impratio['control_profile_failure'])}를 함께 "
            "기록했다. 표 링크는 시간순 최초 위반 직전, 추가 링크는 "
            "[`재디딤 실패 직전`](scenes/sensitivity_impratio_10-replant.png)이다."
        ),
        "- 성공 물리행 그림은 마지막 실측 프레임, 실패 물리행 그림은 최초 위반 직전 프레임이다.",
        (
            "- 원래 일정 156착지에 재디딤 1착지를 추가해 전체 일정은 157착지다. "
            "이 표의 nominal 단기 범위는 20착지를 평가한다."
        ),
        (
            "- 재생 기록은 `.omx/mission-success-20261121/conditions-replant-one-thread`와 "
            "`.omx/mission-success-20261121/conditions-replant`에 있다. 보관한 항목은 "
            "각각 스레드 고정 발표 원시와 환경변수 미설정 대조 원시다. evidence에는 "
            "원시를 복사하지 않고 SHA-256·공식 재판정·장면만 넣었다."
        ),
        "",
        (
            "동결 당시 소스를 다시 실행하려면 아래처럼 commit `2c91d83`의 9개 "
            "source/input 경로와 고정 바이너리 입력, 준비된 RBQ 자산을 한 트리에 모은다. "
            "[`replay.py`](replay.py)는 실행 전에 SHA-256을 대조한다."
        ),
        "",
        "```powershell",
        'cd "$HOME/go2-wall-reach-lab"',
        '$sourceRoot = ".local/source-2c91d83"',
        'New-Item -ItemType Directory -Force "$sourceRoot" | Out-Null',
        "$sourcePaths = @(",
        '  "src"',
        '  "tools"',
        '  "configs"',
        '  "pyproject.toml"',
        '  "evidence/stair-mission-20261121/protocol.json"',
        '  "evidence/t1-stair-ascent-20261120/target-round-trip-geometry.json"',
        '  "evidence/t1-stair-ascent-20261120/target-round-trip-plan.json"',
        '  "evidence/t1-stair-posture-20261120/front-pair.json"',
        '  "evidence/t3-valve-20261120/fixed-body-candidates.json"',
        ")",
        'git archive --format=tar --output="$sourceRoot.tar" 2c91d83 -- $sourcePaths',
        'tar -xf "$sourceRoot.tar" -C "$sourceRoot"',
        'New-Item -ItemType Directory -Force "$sourceRoot/.assets" | Out-Null',
        'Copy-Item -Recurse ".assets/rbq" "$sourceRoot/.assets/rbq"',
        (
            'New-Item -ItemType Directory -Force "$sourceRoot/evidence/'
            'stair-corridor-20261121/local-repair" | Out-Null'
        ),
        (
            'Copy-Item "evidence/stair-corridor-20261121/local-repair/walk50.npz" '
            '"$sourceRoot/evidence/stair-corridor-20261121/local-repair/walk50.npz"'
        ),
        (
            'New-Item -ItemType Directory -Force "$sourceRoot/evidence/'
            'stair-corridor-20261121/tracked/low-control" | Out-Null'
        ),
        (
            'Copy-Item "evidence/stair-corridor-20261121/tracked/low-control/'
            'loaded.json.gz" "$sourceRoot/evidence/stair-corridor-20261121/'
            'tracked/low-control/loaded.json.gz"'
        ),
        (
            ".\\.venv\\Scripts\\python.exe evidence/stair-corridor-20261121/"
            "conditions-replant/replay.py all `"
        ),
        '  --source-root "$sourceRoot" `',
        '  --run-root ".local/t4-conditions-replant" `',
        "  --jobs 3 --cpu-environment one-thread-cpu",
        "```",
        "",
    ]
    (HERE / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / ".omx/mission-success-20261121/conditions-replant",
    )
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    protocol = load(HERE / "frozen-protocol.json")
    execution_environment = load(HERE / "execution-environment.json")
    run_environment = load(run_root / "execution-environment.json")
    if (
        run_environment.get("mode") != "one-thread-cpu"
        or run_environment.get("variables") != protocol["execution_environment"]
    ):
        raise ValueError("run root does not use the frozen one-thread CPU environment")
    official_path = run_root / "assessment/official-aggregate.json"
    official = load(official_path)
    official_rows = {row["id"]: row for row in official["conditions"]}
    rows = []
    for row in assess.protocol_rows(protocol):
        case_id = row["id"]
        if case_id in CASES:
            rows.append(physical_row(case_id, run_root, protocol, row, official_rows[case_id]))
        else:
            rows.append(planning_row(case_id, PLANNING[case_id], row, official_rows[case_id]))
    nominal = next(row for row in rows if row["id"] == "nominal")
    scope = protocol["evaluation_scope"]
    if nominal["n1"]["landings_evaluated"] != scope["expected_landings_at_endpoint"]:
        raise ValueError("nominal landing count differs from frozen acceptance")
    if not math.isclose(nominal["raw_endpoint_s"], scope["nominal_endpoint_s"], abs_tol=2e-9):
        raise ValueError("nominal raw endpoint differs from frozen acceptance")
    counts = {
        name: sum(row["outcome"] == name for row in rows)
        for name in ("success", "planning_failed", "physical_failed", "pending")
    }
    if counts != official["counts"]:
        raise ValueError("official aggregate counts differ")
    verification_root = run_root / "verification-canonical"
    published_verification = HERE / "raw-verification"
    published_verification.mkdir(exist_ok=True)
    for case_id in CASES:
        source = verification_root / f"{case_id}.json"
        verified = load(source)
        row = next(item for item in rows if item["id"] == case_id)
        if verified.get("reproduced") is not True:
            raise ValueError(f"{case_id}: official verifier disagrees")
        if verified["raw_sha256"] != row["source"]["raw_sha256"]:
            raise ValueError(f"{case_id}: verifier raw hash differs")
        shutil.copy2(source, published_verification / source.name)
        row["official_reverification"] = {
            "reproduced": True,
            "path": f"raw-verification/{case_id}.json",
        }
    shutil.copy2(official_path, HERE / "official-aggregate.json")
    scene_root = HERE / "scenes"
    scene_root.mkdir(exist_ok=True)
    for row in rows:
        source_name = (
            "sensitivity_impratio_10-first-chronological.png"
            if row["id"] == "sensitivity_impratio_10"
            else f"{row['id']}.png"
        )
        source = run_root / "assessment/scenes" / source_name
        if not source.exists():
            raise FileNotFoundError(f"missing scene: {source}")
        shutil.copy2(source, scene_root / f"{row['id']}.png")
    replant_scene = run_root / "assessment/scenes/sensitivity_impratio_10-replant.png"
    if not replant_scene.exists():
        raise FileNotFoundError(f"missing replant scene: {replant_scene}")
    shutil.copy2(replant_scene, scene_root / replant_scene.name)
    table = {
        "schema": 1,
        "counts": counts,
        "scope_verdict": "success means success only within through-task scope",
        "schedule": {
            "original_landings": scope["original_scheduled_landings"],
            "supplementary_replant_landings": scope["supplementary_landings"],
            "total_landings": scope["original_scheduled_landings"]
            + scope["supplementary_landings"],
            "nominal_short_scope_landings_evaluated": nominal["n1"]["landings_evaluated"],
            "nominal_raw_endpoint_s": nominal["raw_endpoint_s"],
        },
        "rows": rows,
        "integrity": {
            "frozen_source_commit": protocol["source_commit"],
            "walking_reference_sha256": scope["walking_reference_sha256"],
            "low_load_control_sha256": scope["low_load_control_sha256"],
            "controller_source_sha256": protocol["source_sha256"],
            "official_reverification_count": len(CASES),
            "official_scene_count": len(rows),
            "supplemental_failure_scene_count": 1,
            "raw_files_copied": 0,
            "execution_environment": execution_environment,
        },
    }
    (HERE / "table.json").write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    (HERE / "reproduction.json").write_text(
        json.dumps(
            {
                "profile": protocol["control_profile"],
                "frozen_source_commit": protocol["source_commit"],
                "scope": scope,
                "results": [
                    {
                        "id": row["id"],
                        "outcome": row["outcome"],
                        "m1": verdict(row["m1"]),
                        "n1": verdict(row["n1"]),
                        "first_failure": row.get("first_failure"),
                    }
                    for row in rows
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (HERE / "replay-manifest.json").write_text(
        json.dumps(
            {
                "raw_policy": "do not duplicate multi-GB raw; verify in-place by SHA-256",
                "physical": {
                    row["id"]: row["source"] for row in rows if row["execution"] == "physical"
                },
                "planning": {
                    row["id"]: row["source"] for row in rows if row["execution"] == "planning"
                },
                "commands": {
                    "source_archive_paths": [
                        "src",
                        "tools",
                        "configs",
                        "pyproject.toml",
                        "evidence/stair-mission-20261121/protocol.json",
                        "evidence/t1-stair-ascent-20261120/target-round-trip-geometry.json",
                        "evidence/t1-stair-ascent-20261120/target-round-trip-plan.json",
                        "evidence/t1-stair-posture-20261120/front-pair.json",
                        "evidence/t3-valve-20261120/fixed-body-candidates.json",
                    ],
                    "prepared_asset_copy": ".assets/rbq -> <source-root>/.assets/rbq",
                    "fresh_all": (
                        "python evidence/stair-corridor-20261121/conditions-replant/"
                        "replay.py all --source-root .local/source-2c91d83 --run-root "
                        ".local/t4-conditions-replant --jobs 3 --cpu-environment one-thread-cpu"
                    ),
                    "assemble": (
                        "python evidence/stair-corridor-20261121/conditions-replant/"
                        "assemble.py --run-root .local/t4-conditions-replant"
                    ),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (HERE / "verification.json").write_text(
        json.dumps(
            {
                "passed": True,
                "checks": {
                    "row_count": len(rows),
                    "physical_row_count": len(CASES),
                    "planning_row_count": len(PLANNING),
                    "all_scene_result_raw_hashes_match": True,
                    "all_raw_timesteps_match_scene": True,
                    "all_observed_settings_match_protocol": True,
                    "all_control_profiles_match_frozen_profile": True,
                    "frozen_source_commit": protocol["source_commit"],
                    "source_root_sha256_gate": True,
                    "official_reverification_count": len(CASES),
                    "official_scene_count": len(rows),
                    "supplemental_failure_scene_count": 1,
                    "nominal_landings_evaluated": nominal["n1"]["landings_evaluated"],
                    "nominal_raw_endpoint_s": nominal["raw_endpoint_s"],
                    "raw_files_copied": 0,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_results(rows, counts)
    print(json.dumps({"rows": len(rows), "counts": counts, "nominal": table["schedule"]}, indent=2))


if __name__ == "__main__":
    main()
