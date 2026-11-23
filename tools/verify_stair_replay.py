"""Rejudge a recorded stair-valve replay without advancing MuJoCo or retraining a policy."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from render_stair_mission import build_model
from replay_stair_mission import SHIFT_SECONDS, first_component_failure
from replay_stair_sequence import MAX_FOOT_XY_ERROR_M, evaluate_sequence, measure_milestone
from stair_mission_control_profile import (
    REPLANT_AFTER_LANDING,
    REPLANT_LEG,
    REPLANT_SECONDS,
    RL_REPLANT_PROFILE,
    SLIP_SPRING_ACTIVE_S,
    SLIP_SPRING_GAIN_N_PER_M,
    STANCE_SPRING_GAIN_N_PER_M,
    TERMINAL_OVERTRAVEL_DEG,
    with_failed_replant,
)
from stair_mission_reference import verified_landings
from stair_mission_route import route
from stair_mission_trace import measured_walk
from stair_open_station import STATION_SECONDS, evaluate_station

from go2_wall import grasp_motion, mission_verdict, stair_valve
from go2_wall.statics import torque_limits


def recorded_task_duration(motion):
    if "track_duration_s" in motion:
        return float(motion["track_duration_s"])
    # Older reference trajectories rounded the declared end to the 20 ms planning grid.
    reference_step_s = 0.02
    declared = float(motion["boundaries_s"]["duration"])
    return round(declared / reference_step_s) * reference_step_s


def failed_replant_verdicts(walking, replant_time_s, manifest):
    observed = with_failed_replant(walking, replant_time_s)
    writer = manifest.get("source_sha256", {}).get("tools/replay_stair_mission.py")
    # The frozen writer appended a failed relanding and made it the first walking failure.
    if (
        manifest.get("control_profile", {}).get("name") == RL_REPLANT_PROFILE
        and writer == "511f2ba2595fd6fd0691799b0999139a3eb2fdcaa1f5b566176b3f4697adcb79"
    ):
        return observed, with_failed_replant(walking, replant_time_s, archived_order=True)
    return observed, observed


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def grasp_diagnostics(model, data, source, poses, trace, valve, resistance):
    time = trace["time_s"]
    angle = trace["valve_angle_deg"]
    loaded = (
        (trace["stage"] == "operate")
        & (angle >= 5)
        & (angle < 85)
        & (trace["valve_speed_rad_s"] > 0.02)
        & (trace["hinge_friction_torque_nm"] >= resistance - 0.02)
        & (trace["pad_normal_force_n"].min(axis=1) >= 1.0)
        & (trace["pad_closing_torque_nm"] > 0)
    )
    indices = np.flatnonzero(loaded)
    result = {"bilateral_loaded_rotation_samples_5_to_85_deg": len(indices)}
    stamps = {}
    if valve["first_failure"] and valve["first_failure"]["time_s"] is not None:
        stamps["first_valve_failure"] = int(np.argmin(abs(time - valve["first_failure"]["time_s"])))
    for field in ("pad_closing_torque_nm", "pad_closing_tangent_force_n"):
        if len(indices) and field in source:
            values = source[field][indices]
            result[field] = {"min": float(values.min()), "max": float(values.max())}
            stamps["peak_" + field] = int(indices[np.argmax(values)])
    slides = source["finger_slide_m"]
    axial = source["handle_axial_position_m"]
    gaps = source["pad_surface_gap_m"]
    lever = model.geom(stair_valve.VALVE_LEVER).id
    pads = {model.geom(name).id: name for name in stair_valve.GRIPPER_PADS}
    snapshots = {}
    for label, i in stamps.items():
        data.qpos[:] = poses[i]
        mujoco.mj_forward(model, data)
        contacts = []
        for contact in data.contact[: data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if lever in pair and pair & pads.keys():
                pad = (pair & pads.keys()).pop()
                contacts.append({"pad": pads[pad], "position_world_m": contact.pos.tolist()})
        snapshots[label] = {
            "time_s": float(time[i]),
            "valve_angle_deg": float(angle[i]),
            "pad_normal_force_n": trace["pad_normal_force_n"][i].tolist(),
            "finger_slide_m": slides[i].tolist(),
            "pad_surface_gap_m": gaps[i].tolist(),
            "pad_midpoint_along_lever_m": float(axial[i]),
            "pose_reconstructed_contacts": contacts,
        }
    result["snapshots"] = snapshots
    return result


def verify(folder, walk=None):
    path = folder / "mission.npz"
    scene_path = folder / "scene.json"
    manifest = json.loads(scene_path.read_bytes())
    official = json.loads((folder / "result.json").read_bytes())
    settings = manifest["settings"]
    raw_hash, scene_hash = file_sha(path), file_sha(scene_path)
    fingerprints = {"raw_sha256", "scene_sha256"} & official.keys()
    if fingerprints and len(fingerprints) != 2:
        raise ValueError("Recorded results must include the scene and trace fingerprints together")
    for key, actual in (("raw_sha256", raw_hash), ("scene_sha256", scene_hash)):
        if key in official and official[key] != actual:
            raise ValueError(f"{key} does not match the recorded result")
    geo, _, study, segments, ends, anchor = route(settings["slope_deg"], settings["rise_m"])
    reference = manifest.get("walking_reference")
    if reference is not None:
        ends = verified_landings(
            reference, walk, ends, anchor, settings["slope_deg"], settings["rise_m"]
        )
    elif walk is not None:
        raise ValueError("This recording did not fingerprint its walking reference")
    control_profile = manifest.get("control_profile")
    if control_profile is not None:
        expected_profile = {
            "name": RL_REPLANT_PROFILE,
            "trigger_landing_index": REPLANT_AFTER_LANDING,
            "trigger_time_s": 78.8,
            "trigger_leg": "RR",
            "replant_leg": REPLANT_LEG,
            "replant_time_s": 78.8 + REPLANT_SECONDS,
            "duration_s": REPLANT_SECONDS,
            "slip_spring_gain_n_per_m": SLIP_SPRING_GAIN_N_PER_M,
            "slip_spring_active_s": list(SLIP_SPRING_ACTIVE_S),
            "slip_spring_leg": REPLANT_LEG,
            "terminal_overtravel_deg": TERMINAL_OVERTRAVEL_DEG,
            "stance_spring_gain_n_per_m": STANCE_SPRING_GAIN_N_PER_M,
        }
        if any(control_profile.get(key) != value for key, value in expected_profile.items()):
            raise ValueError("Recorded RL replant profile differs from the frozen definition")
        trigger_t, stance, trigger_leg = ends[REPLANT_AFTER_LANDING]
        if trigger_leg != "RR" or abs(trigger_t - 78.8) > 1e-8:
            raise ValueError("The recorded route lacks the frozen RL replant trigger")
        ends = [
            (time_s + (REPLANT_SECONDS if index > REPLANT_AFTER_LANDING else 0.0), state, leg)
            for index, (time_s, state, leg) in enumerate(ends)
        ]
        ends.insert(
            REPLANT_AFTER_LANDING + 1,
            (trigger_t + REPLANT_SECONDS, stance, REPLANT_LEG),
        )
        anchor += 1
    model, data = build_model(manifest)
    metadata = official["audit"]
    dt = manifest["timestep_s"]
    extent = manifest["extent"]
    opening = extent == "open-at-station"
    if opening:
        from go2_wall.open_valve_grasp import opening_candidate

        installed = study["candidates"][4]
        candidate = opening_candidate(installed)
        stance_settings = manifest.get("walking_control")
        if stance_settings is not None:
            from stair_open_approach import rear_foot_reference

            from go2_wall.stair_open_balance import OpeningStanceControl

            grip_start = (
                ends[anchor][0] + SHIFT_SECONDS + manifest["motion"]["boundaries_s"]["grip_start"]
            )
            expected = OpeningStanceControl(model, ends[anchor][0], grip_start).describe()
            if stance_settings.get("name") == "opening_stance_joint_feedback_v1":
                expected.update(name="opening_stance_joint_feedback_v1", base_position_scale=1.5)
                candidate["base_pose"][1] -= 0.016
                approach = None
            else:
                candidate["base_pose"] = OpeningStanceControl.work_pose(candidate["base_pose"])
                _, ends, approach = rear_foot_reference(
                    geo,
                    segments,
                    ends,
                    anchor,
                    installed["feet_xyz_m"],
                    OpeningStanceControl.REAR_OFFSET_M,
                )
            if stance_settings != expected:
                raise ValueError(
                    "Recorded opening stance controller differs from its dated profile"
                )
            if manifest.get("opening_approach") != approach:
                raise ValueError("Replanned working-tread footsteps differ from the recording")
        if (
            manifest.get("valve_operation") != "open"
            or metadata.get("valve_operation") != "open"
            or metadata.get("stage_map") != {"lower": ["position"]}
            or manifest.get("post_retract_station_s") != STATION_SECONDS
            or manifest.get("station_base_pose") != candidate["base_pose"]
            or manifest["start_deg"] != candidate["start_deg"]
            or manifest["sense"] != candidate["sense"]
            or not np.allclose(manifest["centre_m"], installed["centre_m"], atol=1e-10, rtol=0)
        ):
            raise ValueError(
                "Opening scene differs from the registered installation or station endpoint"
            )
    with np.load(path, allow_pickle=False) as source:
        required = set(mission_verdict.COMMON_FIELDS + mission_verdict.M1_FIELDS)
        optional = {
            "qp_attempt",
            "robot_torque_requested_nm",
            "base_track_m",
            "stance_slip_m",
            "fell",
        }
        trace = {key: source[key] for key in required | (optional & set(source.files))}
        if fingerprints and not optional <= set(trace):
            raise ValueError("Fingerprinted replay is missing per-step walking measurements")
        time = trace["time_s"]
        command_time = np.arange(len(time)) * dt
        if len(time) != metadata["physics_steps"] or not np.allclose(
            time, command_time + dt, rtol=0, atol=1e-7
        ):
            raise ValueError("Physical time axis does not agree with an uninterrupted replay")
        start = ends[anchor][0]
        extended = ends[: anchor + 1]
        if extent != "approach":
            boundaries = manifest["motion"]["boundaries_s"]
            local_start = start + SHIFT_SECONDS
            duration = recorded_task_duration(manifest["motion"])

            def phase(t):
                if t < start:
                    return "prewalk"
                if t < local_start:
                    return "position" if opening else "lower"
                if t < local_start + duration:
                    return grasp_motion.stage_at(t - local_start, boundaries, "operate")
                return "station" if opening else "return"

            expected = [phase(t) for t in command_time]
            if opening:
                if abs(time[-1] - (local_start + duration + STATION_SECONDS)) > 1e-7:
                    raise ValueError("Recording does not end after the complete stationary dwell")
            else:
                extended = [
                    (t if i <= anchor else t + 2 * SHIFT_SECONDS + duration, s, leg)
                    for i, (t, s, leg) in enumerate(ends)
                ]
        else:
            expected = ["prewalk"] * len(time)
        if not np.array_equal(expected, trace["stage"]):
            raise ValueError("Recorded stages differ from the frozen control schedule")
        poses = source["qpos"]
        loads = source["tread_load_n"]
        log = {
            "time": command_time,
            "qpos": poses,
            "qvel": source["qvel"],
            "base": poses[:, :7],
            "extra": loads,
            "feet": trace["feet_xyz_m"],
        }
        report = {
            "scope": extent,
            "physics_steps": len(time),
            "raw_sha256": raw_hash,
            "scene_sha256": scene_hash,
            "fingerprints_match_original": bool(fingerprints),
            "time_axis_verified": True,
            "stage_schedule_verified": True,
        }
        if reference is not None:
            report["verified_walk_sha256"] = reference["sha256"]
        if opening:
            angle = trace["valve_angle_deg"]
            held = angle[trace["stage"] == "hold"]
            observed = {
                "hold_angle_min_deg": float(held.min()),
                "hold_angle_max_deg": float(held.max()),
                "last_valve_travel_deg": float(angle[-1]),
                "duration_s": float(time[-1]),
            }
            if "stance_slip_m" in trace:
                observed["full_route_max_stance_slip_m"] = float(trace["stance_slip_m"].max())
            if "unload_start" in boundaries:
                index = int(np.argmin(np.abs(time - (local_start + boundaries["unload_start"]))))
                observed["tangent_unload_start_s"] = float(time[index])
                observed["valve_travel_at_unload_start_deg"] = float(angle[index])
            report["opening_observations"] = observed
        if "actuation_period_s" in manifest:
            ratio = manifest["actuation_period_s"] / dt
            stride = round(ratio)
            if stride < 1 or not np.isclose(ratio, stride, atol=1e-9, rtol=0):
                raise ValueError("Actuation period is not a whole number of recorded steps")
            if stride > 1:
                for key in ("ctrl", "robot_torque_requested_nm", "requested_finger_force_n"):
                    commands = source[key]
                    changed = np.any(np.diff(commands.reshape(len(time), -1), axis=0) != 0, axis=1)
                    if np.any(changed & ((np.arange(1, len(time)) % stride) != 0)):
                        raise ValueError(f"{key} changed between scheduled actuation ticks")
            report["actuation_period_s"] = manifest["actuation_period_s"]
            report["physics_steps_per_actuation_tick"] = stride
        comparisons = {}
        recorded_walking = None
        if optional <= set(trace):
            walking = measured_walk(
                geo,
                model,
                extended,
                log,
                loads,
                trace,
                np.asarray(manifest.get("joint_limits_nm", torque_limits("rb1"))),
                full=extent == "full",
            )
            report["walking"] = walking
            comparisons["walking"] = walking == official.get("walking")
        if control_profile is not None:
            profile_landing = extended[REPLANT_AFTER_LANDING + 1]
            trigger_time_s = expected_profile["trigger_time_s"]
            replant_time_s = expected_profile["replant_time_s"]
            frames = (log["time"] >= trigger_time_s) & (log["time"] < replant_time_s)
            leg_index = 2
            start = int(np.flatnonzero(log["time"] < trigger_time_s)[-1])
            physical_replant = {
                "original_scheduled_landings": len(extended) - 1,
                "auxiliary_replant_landings": 1,
                "total_scheduled_landings": len(extended),
                "minimum_total_replant_leg_load_n": float(
                    loads[frames, leg_index].sum(axis=1).min()
                ),
                "maximum_foot_lift_m": float(
                    log["feet"][frames, leg_index, 2].max() - log["feet"][start, leg_index, 2]
                ),
                "released_from_support": bool(loads[frames, leg_index].sum(axis=1).min() < 1.0),
                "relanding": measure_milestone({**log, "extra": loads}, geo, *profile_landing),
            }
            physical_replant["verified"] = bool(
                physical_replant["released_from_support"]
                and physical_replant["maximum_foot_lift_m"] > 0.0
                and physical_replant["relanding"]["support"]["all_expected_feet_ge_1n"]
                and physical_replant["relanding"]["terminal_foot_xy_error_m"] <= MAX_FOOT_XY_ERROR_M
            )
            physical_replant = json.loads(json.dumps(physical_replant))
            report["control_profile"] = {
                "name": RL_REPLANT_PROFILE,
                "physical_replant": physical_replant,
            }
            comparisons["control_profile_physics"] = physical_replant == official.get(
                "control_profile", {}
            ).get("physical_replant")
            if not physical_replant["verified"] and "walking" in report:
                report["walking"], recorded_walking = failed_replant_verdicts(
                    report["walking"], replant_time_s, manifest
                )
                if recorded_walking is not report["walking"]:
                    report["archived_walking_first_failure"] = recorded_walking["first_failure"]
                comparisons["walking"] = recorded_walking == official.get("walking")
        if extent != "approach":
            valve = mission_verdict.evaluate_valve(trace, metadata)
            report["valve"] = valve
            report["grip_diagnostics"] = grasp_diagnostics(
                model, data, source, poses, trace, valve, settings["resistance_nm"]
            )
            comparisons["valve_pass_fail"] = valve["verified"] == official["valve"]["verified"]
            comparisons["valve_measurements"] = (
                valve["measurements"] == official["valve"]["measurements"]
            )
            if fingerprints:
                comparisons["valve_failures"] = valve["failures"] == official["valve"]["failures"]
        if opening:
            station = evaluate_station(
                model,
                installed,
                candidate["base_pose"],
                trace,
                poses,
                log["qvel"],
                loads,
                report["walking"]["landings_evaluated"],
                len(extended),
            )
            report["station"] = station
            comparisons["station"] = station == official.get("station")
            complete = all(report[key]["verified"] for key in ("walking", "valve", "station"))
            report["open_at_station_verified"] = complete
            comparisons["station_outcome"] = official.get("outcome") == (
                "success" if complete else "physical_failed"
            )
        if extent == "full":
            planned_end = float(extended[-1][0])
            report["last_landing_coverage"] = {
                "planned_end_s": planned_end,
                "last_physical_sample_s": float(time[-1]),
                "scheduled_landings": len(extended),
                "evaluated_landings": report.get("walking", {}).get("landings_evaluated"),
                "endpoint_recorded": bool(abs(time[-1] - planned_end) <= 1e-7),
            }
            n1 = evaluate_sequence(
                geo, segments, extended, model, official["metrics"], log, round_trip=True
            )
            n1["walking_failures"] = report.get("walking", {}).get("failures", [])
            result = mission_verdict.evaluate_mission(trace, metadata, n1)
            report.update(
                mission=result,
                n1_verified=n1["physical_round_trip_verified"],
                stable_floor_return=n1["stable_floor_return"],
                landings=len(n1["milestones"]),
                loaded_landings=sum(
                    m["support"]["all_expected_feet_ge_1n"] for m in n1["milestones"]
                ),
                first_all_platform_s=n1["first_all_platform_s"],
                first_turn_s=n1["first_turn_on_platform_s"],
            )
            comparisons["mission_pass_fail"] = result["verified"] == official["mission"]["verified"]
            comparisons["n1_route"] = (
                n1["physical_round_trip_verified"]
                == official["route"]["physical_round_trip_verified"]
            )
            if fingerprints:
                comparisons["landing_measurements"] = (
                    json.loads(json.dumps(n1["milestones"])) == official["route"]["milestones"]
                )
                comparisons["mission_first_failure"] = (
                    result["first_failure"] == official["mission"]["first_failure"]
                )
        if (control_profile is not None or opening) and "walking" in report:
            components = [report.get(name) for name in ("valve", "mission", "station")]
            report["first_failure"] = first_component_failure([report["walking"], *components])
            # A failed separate low-load replay has its own clock and needs its own recording.
            low = (official.get("validation") or {}).get("low_load")
            if not low or low.get("verified") is True:
                legacy_order = "archived_walking_first_failure" in report
                expected = first_component_failure(
                    [recorded_walking or report["walking"], *components],
                    first_fields_only=legacy_order,
                )
                if legacy_order:
                    report["archived_result_first_failure"] = expected
                comparisons["overall_first_failure"] = expected == official.get("first_failure")
        report["comparison_with_original"] = comparisons
        report["reproduced"] = all(comparisons.values()) if comparisons else None
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    endpoint = parser.add_mutually_exclusive_group()
    endpoint.add_argument("--require-full-endpoint", action="store_true")
    endpoint.add_argument(
        "--require-open-station",
        action="store_true",
        help="Require approach, two-pad opening and stationary endpoint to pass",
    )
    parser.add_argument("--walk", type=Path, help="original walking reference, for archived runs")
    args = parser.parse_args()
    summary = verify(args.folder, args.walk)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{summary['physics_steps']} raw steps, verdict comparison: {summary['reproduced']}")
    if args.require_full_endpoint:
        coverage = summary.get("last_landing_coverage", {})
        if not coverage.get("endpoint_recorded") or (
            coverage["evaluated_landings"] != coverage["scheduled_landings"]
        ):
            raise SystemExit("The replay does not cover the last scheduled landing")
    if args.require_open_station and (
        summary["scope"] != "open-at-station"
        or not all(summary.get(key, {}).get("verified") for key in ("walking", "valve", "station"))
    ):
        raise SystemExit("The approach, rated valve opening and stationary endpoint must all pass")
    if summary["reproduced"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
