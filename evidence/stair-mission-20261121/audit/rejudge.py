"""Independently rejudge full stair-valve mission recordings from raw evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

HISTORICAL_OPTIONAL_ARRAYS = frozenset({"pad_closing_tangent_force_n"})
EXPECTED_LOW_LOAD_POLARITY = {
    "loaded": True,
    "open_hand": False,
    "frozen-grip-loaded": True,
    "frozen-grip-open": False,
}
EXPECTED_CONDITION_IDS = frozenset(
    {
        "nominal",
        "slope_60",
        "rise_165",
        "rise_200",
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
    }
)

from replay_stair_mission import SHIFT_SECONDS  # noqa: E402
from replay_stair_transfer import expected_surfaces  # noqa: E402
from stair_mission_route import route  # noqa: E402
from verify_stair_replay import verify  # noqa: E402

from go2_wall import mission_verdict  # noqa: E402
from go2_wall.stair_loads import (  # noqa: E402
    FOOT_NAMES,
    JOINT_NAMES,
    MOUNT_WRENCH_FIELDS,
    SURFACE_NAMES,
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def display_path(path: Path) -> str:
    """Use a stable repository-relative label, while accepting evidence from elsewhere."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def recorded_task_duration(motion: dict[str, Any]) -> float:
    """Recover the duration of the frozen trajectory rather than its declared request."""
    if "track_duration_s" in motion:
        return float(motion["track_duration_s"])
    reference_step_s = 0.02
    declared = float(motion["boundaries_s"]["duration"])
    return round(declared / reference_step_s) * reference_step_s


def scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def snapshot(source: Any, index: int) -> dict[str, Any]:
    fields = (
        "time_s",
        "stage",
        "valve_angle_deg",
        "valve_speed_rad_s",
        "pad_normal_force_n",
        "pad_surface_gap_m",
        "pad_closing_torque_nm",
        "hinge_friction_torque_nm",
        "handle_nonpad_force_n",
        "finger_slide_m",
        "gripper_open",
    )
    return {name: scalar(source[name][index]) for name in fields}


def first_sustained(mask: np.ndarray, samples: int) -> int | None:
    if samples < 1 or len(mask) < samples:
        return None
    hits = np.convolve(mask.astype(np.int64), np.ones(samples, dtype=np.int64), mode="valid")
    found = np.flatnonzero(hits == samples)
    return int(found[0]) if len(found) else None


def first_excess_run(mask: np.ndarray, dt: float, limit_s: float) -> tuple[int, float] | None:
    start = None
    for index, bad in enumerate(mask):
        if bad and start is None:
            start = index
        elif not bad:
            start = None
        if start is not None:
            duration = (index - start + 1) * dt
            if duration > limit_s + 1e-12:
                return index, duration
    return None


def quat_yaw_deg(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat.T
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def peak_scalar(values: np.ndarray, time: np.ndarray, stage: np.ndarray) -> dict[str, Any]:
    index = int(np.argmax(np.abs(values)))
    return {
        "index": index,
        "time_s": float(time[index]),
        "stage": str(stage[index]),
        "signed_value": float(values[index]),
        "absolute_value": float(abs(values[index])),
    }


def peak_matrix(
    values: np.ndarray,
    time: np.ndarray,
    stage: np.ndarray,
    axis_names: tuple[str, ...],
) -> dict[str, Any]:
    flat = int(np.argmax(np.abs(values)))
    index, axis = np.unravel_index(flat, values.shape)
    result = peak_scalar(values[:, axis], time, stage)
    result.update(axis_index=int(axis), axis_name=axis_names[axis])
    return result


def audit_shapes(source: Any, n: int) -> dict[str, Any]:
    expected = {
        "torque_nm": (n, 18),
        "robot_torque_requested_nm": (n, 18),
        "joint_force_world_n": (n, 18, 3),
        "joint_moment_world_nm": (n, 18, 3),
        "joint_off_axis_bending_nm": (n, 18),
        "joint_reaction_n": (n, 18),
        "rb1_mount_wrench_world": (n, 6),
        "feet_xyz_m": (n, 4, 3),
        "foot_normal_force_n": (n, 4),
        "tread_load_n": (n, 4, 12),
        "foot_surface_force_world_n": (n, 4, 12, 3),
        "pad_normal_force_n": (n, 2),
        "pad_closing_torque_nm": (n,),
    }
    checks = {}
    for name, shape in expected.items():
        actual = tuple(source[name].shape) if name in source else None
        finite = bool(np.isfinite(source[name]).all()) if name in source else False
        checks[name] = {
            "expected": list(shape),
            "actual": list(actual) if actual is not None else None,
            "finite": finite,
            "passed": actual == shape and finite,
        }
    return {"passed": all(item["passed"] for item in checks.values()), "checks": checks}


def grip_audit(source: Any, dt: float) -> dict[str, Any]:
    time = source["time_s"]
    stage = source["stage"]
    pad = source["pad_normal_force_n"]
    gap = source["pad_surface_gap_m"]
    first_contact = int(np.flatnonzero(stage == "contact")[0])
    pre_grip = max(0, first_contact - 1)
    secured = first_sustained(np.min(pad, axis=1) >= 1.0, round(0.2 / dt))
    operate_loss = (stage == "operate") & (np.min(pad, axis=1) < 1.0)
    violation = first_excess_run(operate_loss, dt, 0.02)
    release_start = int(np.flatnonzero(stage == "release")[0])
    force_free_candidates = np.flatnonzero(
        (np.arange(len(time)) >= release_start) & (np.max(pad, axis=1) <= 1e-6)
    )
    force_free = int(force_free_candidates[0]) if len(force_free_candidates) else None
    separation_candidates = np.flatnonzero(
        (np.arange(len(time)) >= release_start) & (np.min(gap, axis=1) >= 0.005)
    )
    separation = int(separation_candidates[0]) if len(separation_candidates) else None
    end_task = int(np.flatnonzero(stage == "return")[0])
    post = slice(separation, end_task) if separation is not None else slice(end_task, end_task)
    result: dict[str, Any] = {
        "pre_grip": snapshot(source, pre_grip),
        "first_contact_stage": snapshot(source, first_contact),
        "first_0_2s_two_pad_grip": snapshot(source, secured) if secured is not None else None,
        "release_start": snapshot(source, release_start),
        "first_force_free_after_release": snapshot(source, force_free)
        if force_free is not None
        else None,
        "first_5mm_two_pad_separation": snapshot(source, separation)
        if separation is not None
        else None,
        "post_separation_until_return": {
            "sample_count": int(max(0, end_task - (separation or end_task))),
            "max_pad_force_n": float(np.max(pad[post])) if separation is not None else None,
            "max_nonpad_force_n": float(np.max(source["handle_nonpad_force_n"][post]))
            if separation is not None
            else None,
            "angle_range_deg": [
                float(np.min(source["valve_angle_deg"][post])),
                float(np.max(source["valve_angle_deg"][post])),
            ]
            if separation is not None
            else None,
        },
    }
    if violation is not None:
        index, duration = violation
        result["independent_first_contact_loss_violation"] = {
            **snapshot(source, index),
            "duration_s": float(duration),
        }
    else:
        result["independent_first_contact_loss_violation"] = None
    return result


def landing_audit(source: Any, scene: dict[str, Any]) -> dict[str, Any]:
    settings = scene["settings"]
    geometry, _, _, _, ends, anchor = route(settings["slope_deg"], settings["rise_m"])
    duration = recorded_task_duration(scene["motion"])
    extended = [
        (end if index <= anchor else end + 2 * SHIFT_SECONDS + duration, stance, leg)
        for index, (end, stance, leg) in enumerate(ends)
    ]
    dt = float(scene["timestep_s"])
    time = source["time_s"]
    command_time = np.arange(len(time)) * dt
    loads = source["tread_load_n"]
    qpos = source["qpos"]
    supported = 0
    evaluated = 0
    first_bad = None
    supported_platform: list[dict[str, Any]] = []
    for ordinal, (end, stance, leg) in enumerate(extended, 1):
        if end > time[-1] + 1e-7:
            continue
        indices = np.flatnonzero((command_time >= end - 0.3) & (command_time < end))
        if not len(indices):
            continue
        evaluated += 1
        surfaces = expected_surfaces(stance, geometry["rise_m"], geometry["foot_radius_m"])
        per_sample = loads[indices][:, np.arange(4), surfaces].min(axis=1)
        minimum = float(np.min(per_sample))
        passed = minimum >= 1.0
        supported += int(passed)
        if first_bad is None and not passed:
            bad_index = int(indices[np.flatnonzero(per_sample < 1.0)[0]])
            first_bad = {
                "landing": ordinal,
                "time_s": float(time[bad_index]),
                "minimum_expected_support_n": float(
                    per_sample[np.flatnonzero(per_sample < 1.0)[0]]
                ),
                "expected_surfaces": [SURFACE_NAMES[item] for item in surfaces],
                "moved_leg": leg,
            }
        if passed and all(item == len(SURFACE_NAMES) - 1 for item in surfaces):
            index = int(indices[-1])
            supported_platform.append(
                {
                    "landing": ordinal,
                    "time_s": float(time[index]),
                    "heading_deg": float(quat_yaw_deg(qpos[index : index + 1, 3:7])[0]),
                }
            )
    platform_turn = None
    if supported_platform:
        start = supported_platform[0]["heading_deg"]
        errors = [abs((item["heading_deg"] - start) % 360 - 180) for item in supported_platform]
        platform_turn = {"maximum_turn_deg": float(max(errors)), "samples": supported_platform}
    final = command_time >= command_time[-1] - 0.5 - 1e-8
    quat = qpos[:, 3:7]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), -1, 1)))
    return {
        "planned_landings": len(extended),
        "evaluated_landings": evaluated,
        "supported_landings": supported,
        "first_bad_support": first_bad,
        "supported_all_platform_landings": len(supported_platform),
        "platform_turn": platform_turn,
        "maximum_base_xyz_m": np.max(source["base_xyz_m"], axis=0).tolist(),
        "return_last_0_5s": {
            "sample_count": int(np.count_nonzero(final)),
            "max_start_xy_error_m": float(
                np.max(np.linalg.norm(qpos[final, :2] - qpos[0, :2], axis=1))
            ),
            "max_tilt_deg": float(np.max(tilt[final])),
            "max_linear_speed_mps": float(
                np.max(np.linalg.norm(source["qvel"][final, :3], axis=1))
            ),
            "max_angular_speed_radps": float(
                np.max(np.linalg.norm(source["qvel"][final, 3:6], axis=1))
            ),
            "min_floor_support_n": float(np.min(loads[final, :, 0])),
        },
        "last_planned_end_s": float(extended[-1][0]),
        "last_observation_s": float(time[-1]),
        "last_landing_omitted_by_time_gate": bool(extended[-1][0] > time[-1] + 1e-7),
    }


def load_peaks(source: Any, scene: dict[str, Any]) -> dict[str, Any]:
    time, stage = source["time_s"], source["stage"]
    limits = np.asarray(scene["joint_limits_nm"], dtype=float)
    torque = peak_matrix(source["torque_nm"], time, stage, JOINT_NAMES)
    torque["joint_limit_nm"] = float(limits[torque["axis_index"]])
    torque["relative_to_joint_limit"] = torque["absolute_value"] / torque["joint_limit_nm"]

    force_vectors = source["foot_surface_force_world_n"]
    magnitudes = np.linalg.norm(force_vectors, axis=3)
    flat = int(np.argmax(magnitudes))
    index, foot, surface = np.unravel_index(flat, magnitudes.shape)
    reaction = {
        "index": int(index),
        "time_s": float(time[index]),
        "stage": str(stage[index]),
        "foot": FOOT_NAMES[foot],
        "surface": SURFACE_NAMES[surface],
        "magnitude_n": float(magnitudes[index, foot, surface]),
        "world_vector_n": force_vectors[index, foot, surface].tolist(),
    }
    mount = {}
    for index, name in enumerate(MOUNT_WRENCH_FIELDS):
        mount[name] = peak_scalar(source["rb1_mount_wrench_world"][:, index], time, stage)
    result = {
        "actual_18_axis_torque": torque,
        "foot_surface_4x12_reaction": reaction,
        "measured_pad_closing_torque": peak_scalar(source["pad_closing_torque_nm"], time, stage),
        "rb1_mount_wrench_component_peaks": mount,
    }
    if "pad_closing_tangent_force_n" in source:
        signed_force = source["pad_closing_tangent_force_n"]
        angle = source["valve_angle_deg"]
        loaded = (
            (stage == "operate")
            & (angle >= 5)
            & (angle < 85)
            & (source["valve_speed_rad_s"] > 0.02)
            & (source["hinge_friction_torque_nm"] >= scene["settings"]["resistance_nm"] - 0.02)
            & (np.min(source["pad_normal_force_n"], axis=1) >= 1.0)
            & (source["pad_closing_torque_nm"] > 0)
        )
        loaded_count = int(np.count_nonzero(loaded))
        result["signed_pad_closing_tangential_force"] = {
            "shape": list(signed_force.shape),
            "finite": bool(np.isfinite(signed_force).all()),
            "overall_peak": peak_scalar(signed_force, time, stage),
            "loaded_rotation_samples": loaded_count,
            "loaded_rotation_min_n": float(np.min(signed_force[loaded])) if loaded_count else None,
            "loaded_rotation_max_n": float(np.max(signed_force[loaded])) if loaded_count else None,
            "loaded_rotation_all_positive": bool(np.all(signed_force[loaded] > 0))
            if loaded_count
            else None,
        }
    return result


def compare_official(independent: dict[str, Any], official: dict[str, Any]) -> dict[str, Any]:
    official_failure = next(
        (item for item in official["valve"]["failures"] if item["metric"] == "pad_contact_loss_s"),
        None,
    )
    found = independent["grip"]["independent_first_contact_loss_violation"]
    result = {
        "pad_contact_loss_presence_match": (official_failure is None) == (found is None),
        "walking_evaluated_count_match": official["walking"]["landings_evaluated"]
        == independent["landings"]["evaluated_landings"],
        "route_loaded_count_match": sum(
            item["support"]["all_expected_feet_ge_1n"] for item in official["route"]["milestones"]
        )
        == independent["landings"]["supported_landings"],
    }
    if official_failure is not None and found is not None:
        result.update(
            pad_contact_loss_time_delta_s=abs(official_failure["time_s"] - found["time_s"]),
            pad_contact_loss_duration_delta_s=abs(
                official_failure["measured"] - found["duration_s"]
            ),
        )
    else:
        result.update(
            pad_contact_loss_time_delta_s=None,
            pad_contact_loss_duration_delta_s=None,
        )
    return result


def audit_run(folder: Path, *, require_full_endpoint: bool) -> dict[str, Any]:
    scene = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
    official = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    raw = folder / "mission.npz"
    if raw.stat().st_size == 0:
        raise ValueError("mission.npz is empty")
    replay = verify(folder)
    with np.load(raw, allow_pickle=False) as source:
        if "time_s" not in source or len(source["time_s"]) == 0:
            raise ValueError("mission.npz has no physical samples")
        n = len(source["time_s"])
        seconds = float(official["metrics"]["seconds"])
        dt = float(scene["timestep_s"])
        independent = {
            "steps": n,
            "time_axis": {
                "first_observation_s": float(source["time_s"][0]),
                "last_observation_s": float(source["time_s"][-1]),
                "planned_seconds": seconds,
                "planned_seconds_divided_by_dt": seconds / dt,
                "int_steps": int(seconds / dt),
                "round_steps": round(seconds / dt),
                "missing_steps_from_round": round(seconds / dt) - n,
            },
            "shapes": audit_shapes(source, n),
            "grip": grip_audit(source, dt),
            "landings": landing_audit(source, scene),
            "peaks": load_peaks(source, scene),
            "signed_closing_tangential_force_available": "pad_closing_tangent_force_n" in source,
        }
    independent["official_comparison"] = compare_official(independent, official)
    time_axis = independent["time_axis"]
    endpoint_complete = (
        independent["steps"] == time_axis["round_steps"]
        and not independent["landings"]["last_landing_omitted_by_time_gate"]
    )
    raw_hash = sha256(raw)
    scene_hash = sha256(folder / "scene.json")
    hashes_match = (
        official.get("raw_sha256") == raw_hash and official.get("scene_sha256") == scene_hash
    )
    comparison = independent["official_comparison"]
    independent_matches = all(
        value for key, value in comparison.items() if key.endswith("_match")
    ) and all(
        value is None or value <= 1e-7
        for key, value in comparison.items()
        if key.endswith("_delta_s")
    )
    objective_checks = {
        "stored_fingerprints_present": bool(
            official.get("raw_sha256") and official.get("scene_sha256")
        ),
        "stored_fingerprints_match": hashes_match,
        "frozen_replay_reproduced": replay["reproduced"] is True,
        "independent_shapes_and_finiteness": independent["shapes"]["passed"],
        "independent_measurements_match": independent_matches,
        "full_endpoint_present_if_required": endpoint_complete if require_full_endpoint else True,
    }
    return {
        "folder": display_path(folder),
        "raw_sha256": raw_hash,
        "scene_sha256": scene_hash,
        "stored_raw_sha256": official.get("raw_sha256"),
        "stored_scene_sha256": official.get("scene_sha256"),
        "hashes_match_stored": hashes_match,
        "recorded_mission_verified": official["mission"]["verified"],
        "endpoint_complete": endpoint_complete,
        "objective_checks": objective_checks,
        "audit_passed": all(objective_checks.values()),
        "frozen_replay": {
            "reproduced": replay["reproduced"],
            "comparisons": replay["comparison_with_original"],
            "first_valve_failure": replay["valve"]["first_failure"],
            "loaded_landings": replay["loaded_landings"],
            "first_all_platform_s": replay["first_all_platform_s"],
            "first_turn_s": replay["first_turn_s"],
            "stable_floor_return": replay["stable_floor_return"],
        },
        "independent_numpy": independent,
    }


def audit_low_load() -> dict[str, Any]:
    summaries = [
        ROOT / "evidence/stair-mission-20261121/low-grip/results.json",
        ROOT / "evidence/stair-mission-20261121/conditions/physical-b/low-control/results.json",
    ]
    reports = []
    observed_names: set[str] = set()
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if "runs" in summary:
            runs = summary["runs"]
        else:
            runs = {name: item for name, item in summary.items() if name in ("loaded", "open_hand")}
        checked = {}
        for name, item in runs.items():
            observed_names.add(name)
            raw_path = summary_path.parent / item["file" if "file" in item else "raw"]
            compressed = raw_path.read_bytes()
            expanded = gzip.decompress(compressed)
            payload = json.loads(expanded)
            rejudged = mission_verdict.evaluate_low_load(payload["trace"], payload["metadata"])
            recorded = item["verdict"]
            checked[name] = {
                "gzip_sha256": hashlib.sha256(compressed).hexdigest(),
                "expanded_json_sha256": hashlib.sha256(expanded).hexdigest(),
                "stored_hashes_match": hashlib.sha256(compressed).hexdigest()
                == item.get("gzip_sha256", item.get("raw_gzip_sha256"))
                and hashlib.sha256(expanded).hexdigest()
                == item.get("json_sha256", item.get("expanded_json_sha256")),
                "rejudged_verified": rejudged["verified"],
                "recorded_verified": recorded["verified"],
                "expected_verified": EXPECTED_LOW_LOAD_POLARITY.get(name),
                "expected_polarity_match": rejudged["verified"]
                == EXPECTED_LOW_LOAD_POLARITY.get(name),
                "verdict_exact_match": rejudged == recorded,
                "first_failure": rejudged["first_failure"],
            }
        reports.append(
            {
                "summary": display_path(summary_path),
                "summary_sha256": sha256(summary_path),
                "runs": checked,
            }
        )
    flat = [item for report in reports for item in report["runs"].values()]
    checks = {
        "run_ids_match_expected": observed_names == set(EXPECTED_LOW_LOAD_POLARITY),
        "all_raw_hashes_match": all(item["stored_hashes_match"] for item in flat),
        "all_verdicts_exactly_reproduced": all(item["verdict_exact_match"] for item in flat),
        "all_expected_polarities_match": all(item["expected_polarity_match"] for item in flat),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed_run_ids": sorted(observed_names),
        "reports": reports,
    }


def matching_failure(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("stage", "metric", "limit")
    if any(left.get(key) != right.get(key) for key in keys):
        return False
    for key in ("time_s", "measured"):
        if isinstance(left.get(key), (int, float)) and isinstance(right.get(key), (int, float)):
            if not np.isclose(left[key], right[key], atol=1e-9, rtol=0):
                return False
        elif left.get(key) != right.get(key):
            return False
    return True


def audit_conditions() -> dict[str, Any]:
    path = ROOT / "evidence/stair-mission-20261121/conditions/physical-b/condition-results.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    checks = []
    for condition in summary["conditions"]:
        evidence = ROOT / condition["evidence_path"]
        payload = json.loads(evidence.read_text(encoding="utf-8"))
        if condition["outcome"] == "physical_failed":
            evidence_failure = payload["first_failure"]
            raw_check_path = evidence.parent / "raw-check.json"
            raw_check = json.loads(raw_check_path.read_text(encoding="utf-8"))
            raw_path = evidence.parent / "mission.npz"
            scene_path = evidence.parent / "scene.json"
            actual_raw_hash = sha256(raw_path)
            actual_scene_hash = sha256(scene_path)
            fingerprint_ok = (
                payload.get("raw_sha256") == raw_check.get("raw_sha256") == actual_raw_hash
                and payload.get("scene_sha256")
                == raw_check.get("scene_sha256")
                == actual_scene_hash
            )
            replay_ok = all(
                (
                    raw_check.get("reproduced") is True,
                    raw_check.get("fingerprints_match_original") is True,
                    raw_check.get("time_axis_verified") is True,
                    raw_check.get("stage_schedule_verified") is True,
                    all(raw_check.get("comparison_with_original", {}).values()),
                    fingerprint_ok,
                )
            )
        else:
            raw_failure = payload["first_failure"]
            overlaps = raw_failure.get("unexpected_overlaps", [])
            if overlaps:
                evidence_failure = {
                    "stage": raw_failure["kind"],
                    "time_s": raw_failure["planned_time_s"],
                    "metric": "unexpected_overlap_m",
                    "measured": max(item["penetration_m"] for item in overlaps),
                    "limit": raw_failure["collision_tolerance_m"],
                }
            else:
                evidence_failure = {
                    "stage": raw_failure["kind"],
                    "time_s": raw_failure["planned_time_s"],
                    "metric": "max_foot_ik_error_m",
                    "measured": max(raw_failure["foot_ik_error_m"].values()),
                    "limit": raw_failure["foot_ik_limit_m"],
                }
            replay_ok = None
            fingerprint_ok = None
        checks.append(
            {
                "id": condition["id"],
                "outcome": condition["outcome"],
                "evidence_outcome": payload.get("outcome"),
                "evidence_outcome_matches_summary": payload.get("outcome") == condition["outcome"],
                "evidence_exists": evidence.is_file(),
                "evidence_sha256": sha256(evidence),
                "first_failure_matches_summary": matching_failure(
                    condition["first_failure"], evidence_failure
                ),
                "raw_replay_reproduced": replay_ok,
                "raw_evidence_fingerprints_match": fingerprint_ok,
            }
        )
    ids = [item["id"] for item in checks]
    outcome_counts = Counter(item["outcome"] for item in checks)
    computed_counts = {
        "success": outcome_counts["success"],
        "planning_failed": outcome_counts["planning_failed"],
        "physical_failed": outcome_counts["physical_failed"],
        "pending": outcome_counts["pending"],
    }
    result = {
        "summary": display_path(path),
        "summary_sha256": sha256(path),
        "stored_counts": summary["counts"],
        "computed_counts": computed_counts,
        "condition_count": len(checks),
        "condition_ids_match_expected": set(ids) == set(EXPECTED_CONDITION_IDS)
        and len(ids) == len(set(ids)),
        "stored_counts_match_conditions": summary["counts"] == computed_counts,
        "all_evidence_exists": all(item["evidence_exists"] for item in checks),
        "all_evidence_outcomes_match": all(
            item["evidence_outcome_matches_summary"] for item in checks
        ),
        "all_first_failures_match": all(item["first_failure_matches_summary"] for item in checks),
        "all_physical_raw_replays_reproduced": all(
            item["raw_replay_reproduced"] is True
            for item in checks
            if item["outcome"] == "physical_failed"
        ),
        "conditions": checks,
    }
    result["passed"] = all(
        result[key]
        for key in (
            "condition_ids_match_expected",
            "stored_counts_match_conditions",
            "all_evidence_exists",
            "all_evidence_outcomes_match",
            "all_first_failures_match",
            "all_physical_raw_replays_reproduced",
        )
    )
    return result


def array_bytes_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare shape, dtype, and payload bytes; distinguish +0.0 from -0.0."""
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_bytes = np.ascontiguousarray(left).view(np.uint8).reshape(-1)
    right_bytes = np.ascontiguousarray(right).view(np.uint8).reshape(-1)
    return bool(np.array_equal(left_bytes, right_bytes))


def compare_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    for left, right in combinations(runs, 2):
        left_folder = Path(left["folder"])
        right_folder = Path(right["folder"])
        if not left_folder.is_absolute():
            left_folder = ROOT / left_folder
        if not right_folder.is_absolute():
            right_folder = ROOT / right_folder
        left_path = left_folder / "mission.npz"
        right_path = right_folder / "mission.npz"
        left_stat = left_path.stat()
        right_stat = right_path.stat()
        distinct_folders = left_folder.resolve() != right_folder.resolve()
        distinct_raw_files = distinct_folders and not left_path.samefile(right_path)
        with (
            np.load(left_path, allow_pickle=False) as lhs,
            np.load(right_path, allow_pickle=False) as rhs,
        ):
            left_items = set(lhs.files)
            right_items = set(rhs.files)
            shared = sorted(left_items & right_items)
            extras = sorted(left_items ^ right_items)
            equal = [name for name in shared if array_bytes_equal(lhs[name], rhs[name])]
            differing = [name for name in shared if name not in equal]
            prefix = min(len(lhs["time_s"]), len(rhs["time_s"]))
            if len(lhs["time_s"]) == len(rhs["time_s"]):
                prefix_equal = equal.copy()
            else:
                prefix_equal = [
                    name
                    for name in shared
                    if array_bytes_equal(lhs[name][:prefix], rhs[name][:prefix])
                ]
        item_sets_equal = not extras
        all_shared_arrays_equal = len(equal) == len(shared)
        all_shared_prefixes_equal = len(prefix_equal) == len(shared)
        historical_extra_allowed = set(extras) <= HISTORICAL_OPTIONAL_ARRAYS and len(extras) <= 1
        strict_passed = all(
            (
                distinct_raw_files,
                left["raw_sha256"] == right["raw_sha256"],
                left["scene_sha256"] == right["scene_sha256"],
                item_sets_equal,
                all_shared_arrays_equal,
            )
        )
        historical_passed = all(
            (
                distinct_raw_files,
                historical_extra_allowed,
                all_shared_arrays_equal,
            )
        )
        comparisons.append(
            {
                "left": left["folder"],
                "right": right["folder"],
                "left_raw_file_identity": {
                    "device": left_stat.st_dev,
                    "inode": left_stat.st_ino,
                },
                "right_raw_file_identity": {
                    "device": right_stat.st_dev,
                    "inode": right_stat.st_ino,
                },
                "distinct_folders": distinct_folders,
                "distinct_raw_files": distinct_raw_files,
                "raw_sha256_equal": left["raw_sha256"] == right["raw_sha256"],
                "scene_sha256_equal": left["scene_sha256"] == right["scene_sha256"],
                "raw_item_sets_equal": item_sets_equal,
                "raw_item_symmetric_difference": extras,
                "shared_arrays": len(shared),
                "byte_identical_shared_arrays": len(equal),
                "byte_differing_shared_arrays": differing,
                "common_prefix_samples": prefix,
                "byte_identical_common_prefix_arrays": len(prefix_equal),
                "all_shared_prefixes_byte_identical": all_shared_prefixes_equal,
                "historical_optional_array_difference_allowed": historical_extra_allowed,
                "historical_compatibility_passed": historical_passed,
                "strict_reproduction_passed": strict_passed,
            }
        )
    return comparisons


def comparison_gate(
    runs: list[dict[str, Any]], comparisons: list[dict[str, Any]], *, strict: bool
) -> dict[str, Any]:
    if len(runs) < 2:
        return {
            "mode": "single_run",
            "required": False,
            "pair_count": 0,
            "passed": True,
            "message": "one run was supplied; independent repeatability was not evaluated",
        }
    field = "strict_reproduction_passed" if strict else "historical_compatibility_passed"
    return {
        "mode": "strict_reproduction" if strict else "historical_compatibility",
        "required": True,
        "pair_count": len(comparisons),
        "passed": bool(comparisons) and all(item[field] for item in comparisons),
        "message": (
            "strict mode requires distinct raw files, identical raw/scene SHA256, item sets, "
            "dtypes, shapes, and array bytes"
            if strict
            else "historical mode permits only the signed pad-force array to be added"
        ),
    }


def side_evidence_gate(low_load: dict[str, Any], conditions: dict[str, Any]) -> dict[str, bool]:
    result = {
        "low_load_positive_negative_passed": low_load.get("passed") is True,
        "physical_b_17_conditions_passed": conditions.get("passed") is True,
    }
    result["passed"] = all(result.values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folders",
        nargs="+",
        type=Path,
        help="one or more folders containing mission.npz, scene.json, and result.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new or replaceable JSON summary path",
    )
    parser.add_argument(
        "--require-full-endpoint",
        action="store_true",
        help="fail if floating-point truncation omitted the final planned landing endpoint",
    )
    args = parser.parse_args()
    runs = []
    errors = []
    for folder in args.folders:
        folder = (folder if folder.is_absolute() else ROOT / folder).resolve()
        required = [folder / name for name in ("mission.npz", "scene.json", "result.json")]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            errors.append(
                {
                    "folder": display_path(folder),
                    "error_type": "MissingEvidence",
                    "message": "missing required evidence: " + ", ".join(missing),
                }
            )
            continue
        try:
            runs.append(audit_run(folder, require_full_endpoint=args.require_full_endpoint))
        except Exception as exc:  # Preserve a machine-readable failure instead of a false PASS.
            errors.append(
                {
                    "folder": display_path(folder),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    comparisons = compare_runs(runs)
    comparison = comparison_gate(runs, comparisons, strict=args.require_full_endpoint)
    try:
        low_load = audit_low_load()
    except Exception as exc:  # Keep side-evidence errors visible in the machine-readable result.
        low_load = {"passed": False, "error_type": type(exc).__name__, "message": str(exc)}
    try:
        conditions = audit_conditions()
    except Exception as exc:  # Keep side-evidence errors visible in the machine-readable result.
        conditions = {"passed": False, "error_type": type(exc).__name__, "message": str(exc)}
    side_evidence = side_evidence_gate(low_load, conditions)
    audit_passed = all(
        (
            bool(runs),
            not errors,
            all(run["audit_passed"] for run in runs),
            comparison["passed"],
            side_evidence["passed"],
        )
    )
    output = {
        "schema": 1,
        "invocation": {
            "argv": sys.argv.copy(),
            "tool": display_path(Path(__file__)),
            "tool_sha256": sha256(Path(__file__)),
            "python": sys.version,
        },
        "audit_passed": audit_passed,
        "require_full_endpoint": args.require_full_endpoint,
        "runs": runs,
        "run_comparisons": comparisons,
        "comparison_gate": comparison,
        "errors": errors,
        "side_evidence_gate": side_evidence,
        "prior_low_load_positive_negative": low_load,
        "physical_b_17_conditions": conditions,
        "decision_log": [
            *[
                {
                    "event": "run_audited",
                    "folder": run["folder"],
                    "audit_passed": run["audit_passed"],
                    "recorded_mission_verified": run["recorded_mission_verified"],
                    "endpoint_complete": run["endpoint_complete"],
                }
                for run in runs
            ],
            {
                "event": "comparison_gate",
                "mode": comparison["mode"],
                "passed": comparison["passed"],
                "pair_count": comparison["pair_count"],
            },
            {
                "event": "side_evidence_gate",
                "passed": side_evidence["passed"],
            },
            {"event": "audit_complete", "passed": audit_passed},
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    failed_checks = sum(not run["audit_passed"] for run in runs)
    failed_global_gates = int(not comparison["passed"]) + int(not side_evidence["passed"])
    print(
        f"wrote {args.output}: {len(runs)} audited, "
        f"{len(errors)} invalid, {failed_checks} run-check failures, "
        f"{failed_global_gates} global-gate failures"
    )
    if not audit_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
