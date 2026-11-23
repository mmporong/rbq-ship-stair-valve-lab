"""Replay the contact-gripped stair-valve mission from the floor in one MuJoCo state."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from replay_stair_sequence import MAX_FOOT_XY_ERROR_M, evaluate_sequence, measure_milestone
from replay_stair_transfer import TIMESTEP_S
from stair_mission_control_profile import (
    RL_REPLANT_PROFILE,
    SLIP_SPRING_ACTIVE_S,
    SLIP_SPRING_GAIN_N_PER_M,
    apply_replant,
    with_failed_replant,
)
from stair_mission_limits import configure_mission_torque_limits
from stair_mission_reference import CORRIDOR_PROFILE, describe_reference, scheduled_landings
from stair_mission_route import load, planner_source_hashes, route, sample_walk, source_hashes
from stair_mission_trace import (
    PAD_PAIRS,
    SLIDES,
    audit_metadata,
    measured_walk,
    observe,
    trace_arrays,
)
from stair_open_station import STATION_SECONDS, evaluate_station

from go2_wall import grasp_motion, mission_verdict, stair_valve, stairs, transition
from go2_wall.robot import ROOT
from go2_wall.stair_loads import StairLoads
from go2_wall.stair_walking_control import StairWalkingControl
from go2_wall.statics import torque_limits

PROTOCOL = ROOT / "evidence/stair-mission-20261121/protocol.json"
TRACK_FIELDS = ("qpos", "target", "torque", "tip", "stance", "wall", "cap", "feet")
SHIFT_SECONDS = 3.0
SOURCE_PATHS = (
    "tools/replay_stair_mission.py",
    "tools/stair_mission_route.py",
    "tools/stair_mission_reference.py",
    "tools/stair_mission_control_profile.py",
    "tools/stair_mission_trace.py",
    "tools/replay_stair_sequence.py",
    "src/go2_wall/grasp_motion.py",
    "src/go2_wall/stair_valve.py",
    "src/go2_wall/transition.py",
    "src/go2_wall/gripper_control.py",
    "src/go2_wall/stair_loads.py",
    "src/go2_wall/mission_verdict.py",
    "tools/stair_mission_limits.py",
)


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def join_tracks(pieces):
    result = {key: [] for key in ("time", *TRACK_FIELDS, "gain_scale")}
    clock = 0.0
    for track, gain_start, gain_end in pieces:
        times = track["time"] - track["time"][0]
        include = slice(1, None) if result["time"] else slice(None)
        result["time"].append(clock + times[include])
        scales = gain_start + (gain_end - gain_start) * times / times[-1]
        result["gain_scale"].append(scales[include])
        for key in TRACK_FIELDS:
            result[key].append(track[key][include])
        clock += times[-1]
    return {key: np.concatenate(arrays) for key, arrays in result.items()}


def subset(track, row_slice):
    return {key: track[key][row_slice] for key in ("time", *TRACK_FIELDS)}


def make_mission(geometry, sequence, study, segments, ends, anchor, walk, args, case):
    saved = dict(study["candidates"][4])
    opening = args.extent == "open-at-station"
    if opening:
        from go2_wall.open_valve_grasp import opening_candidate

        saved = opening_candidate(saved)
        if getattr(args, "opening_stance_feedback", False):
            from go2_wall.stair_open_balance import OpeningStanceControl

            saved["base_pose"] = OpeningStanceControl.work_pose(saved["base_pose"])
            saved["opening_load_relief"] = {"delay_s": 2.0, "ramp_s": 0.8, "hold_fraction": 0.2}
            for leg in ("RL", "RR"):
                saved["feet_xyz_m"][leg][0] -= OpeningStanceControl.REAR_OFFSET_M
    saved["handle_length_m"] = case["handle_length_m"]
    anchor_t, stance, _ = ends[anchor]
    boundary = int(np.argmin(np.abs(walk["time"] - anchor_t)))
    if abs(walk["time"][boundary] - anchor_t) > 1e-7:
        raise ValueError("Missing walking sample at the manipulation stance")
    walk_gain = sequence["gain_scale"]
    pre = subset(walk, slice(0, boundary + 1))
    limits = torque_limits("rb1")
    stairs.ARM = "rb1"
    stairs.use_stow()
    planning_model, planning_data = stairs.build_stairs(geometry["slope_deg"], geometry["rise_m"])
    stow = geometry["arm_stow_rad"]
    if args.extent == "approach":
        pre["gain_scale"] = np.full(len(pre["time"]), walk_gain)

        def open_hand(t, pos, vel):
            return grasp_motion.finger_command(
                -1.0, pos, vel, boundaries={"approach": 0.0}, force_limit_n=case["finger_force_n"]
            )

        return pre, lambda t: "prewalk", open_hand, saved, ends[: anchor + 1], {}, None, None

    nominal = study["candidates"][4]["base_pose"].copy()
    nominal[2] += 0.04
    target = saved["base_pose"]
    shifts = []
    shift_targets = [(nominal, target)] if opening else [(nominal, target), (target, nominal)]
    for start, end in shift_targets:
        moves = [
            {"kind": "shift", "from": start, "to": end, "seconds": SHIFT_SECONDS, "stance": stance}
        ]
        shifts.append(
            transition.sample_segments(planning_model, planning_data, moves, limits, stow)
        )
    task_track, details = grasp_motion.make_grasp_track(
        planning_model,
        planning_data,
        {**study, "arm_stow_rad": stow},
        saved,
        target_deg=90.0,
        motion_s=args.motion_seconds,
        resistance_nm=10.0,
        jaw_load_bias_m=args.jaw_load_bias_m,
        terminal_overtravel_deg=getattr(args, "terminal_overtravel_deg", 0.0),
        durations_s={
            "settle": grasp_motion.DEFAULT_DURATIONS_S["settle"] + args.extra_settle_seconds
        },
    )
    boundaries = details["boundaries_s"]
    during = float(task_track["time"][-1])
    pieces = [
        (pre, walk_gain, walk_gain),
        (shifts[0], walk_gain, args.gain),
        (task_track, args.gain, args.gain),
    ]
    if opening:
        parked = {key: np.repeat(task_track[key][-1:], 2, axis=0) for key in TRACK_FIELDS}
        parked["time"] = np.array([0.0, STATION_SECONDS])
        pieces.append((parked, args.gain, args.gain))
        new_ends = ends[: anchor + 1]
    else:
        post_end = (
            None
            if args.extent == "full"
            else int(np.searchsorted(walk["time"], ends[anchor + 1][0] + 1e-8, side="right"))
        )
        post = subset(walk, slice(boundary, post_end))
        pieces.extend([(shifts[1], args.gain, walk_gain), (post, walk_gain, walk_gain)])
        extra = 2 * SHIFT_SECONDS + during
        new_ends = [(t if i <= anchor else t + extra, s, leg) for i, (t, s, leg) in enumerate(ends)]
    track = join_tracks(pieces)
    start_task = anchor_t + SHIFT_SECONDS
    finish_task = start_task + during

    def phase(t):
        if t < anchor_t:
            return "prewalk"
        if t < start_task:
            return "position" if opening else "lower"
        if t < finish_task:
            return grasp_motion.stage_at(t - start_task, boundaries, "operate")
        return "station" if opening else "return"

    def command(t, pos, vel):
        # The local controller opens the fingers outside its contact and turning interval.
        return grasp_motion.finger_command(
            t - start_task, pos, vel, boundaries=boundaries, force_limit_n=case["finger_force_n"]
        )

    feedback = grasp_motion.make_arm_target_command(
        saved, boundaries, 90.0, args.motion_seconds, jaw_load_bias_m=args.jaw_load_bias_m
    )

    def arm_command(t, model, data, nominal_arm):
        if start_task <= t < finish_task:
            return feedback(t - start_task, model, data, nominal_arm)
        return nominal_arm

    return track, phase, command, saved, new_ends, details, arm_command, feedback


def first_component_failure(verdicts, *, first_fields_only=False):
    failures = []
    for verdict in verdicts:
        if not verdict:
            continue
        if not first_fields_only and verdict.get("failures"):
            failures.extend(verdict["failures"])
        elif verdict.get("first_failure") is not None:
            failures.append(verdict["first_failure"])

    def time_order(failure):
        time_s = failure.get("time_s")
        return time_s if isinstance(time_s, (int, float)) and math.isfinite(time_s) else math.inf

    return min(failures, key=time_order, default=None)


def run(args):
    use_feedback = getattr(args, "corridor_feedback", False)
    files = (
        (*SOURCE_PATHS, "src/go2_wall/stair_walking_control.py") if use_feedback else SOURCE_PATHS
    )
    if args.extent == "open-at-station":
        files = (*files, "src/go2_wall/open_valve_grasp.py", "tools/stair_open_station.py")
        if getattr(args, "opening_stance_feedback", False):
            files = (*files, "src/go2_wall/stair_open_balance.py", "tools/stair_open_approach.py")
    versions = source_hashes(files)
    protocol = load(PROTOCOL)
    settings = dict(
        slope_deg=50.0,
        rise_m=0.19,
        floor_mu=0.6,
        pad_mu=0.8,
        resistance_nm=10.0,
        handle_length_m=0.12,
        finger_force_n=100.0,
        slide_loss_n=30.0,
    )
    settings.update(protocol["single_factor_cases"][args.case])
    for key in settings:
        override = getattr(args, key, None)
        if override is not None:
            settings[key] = override
    geometry, seq, study, segments, ends, anchor = route(settings["slope_deg"], settings["rise_m"])
    walking_reference = None
    windows = None
    if args.walk:
        with np.load(args.walk, allow_pickle=False) as cache:
            dimensions = (float(cache["slope_deg"]), float(cache["rise_m"]))
            if dimensions != (settings["slope_deg"], settings["rise_m"]):
                raise ValueError("Walking reference has different stair geometry")
            ends, profile = scheduled_landings(cache, ends, anchor, *dimensions)
            walking_reference = describe_reference(args.walk, profile)
            walk = {key: cache[key] for key in ("time", *TRACK_FIELDS)}
            if use_feedback and profile == CORRIDOR_PROFILE:
                windows = np.interp(
                    [89.6, 189.6, 544.6, 651.7], cache["reference_time"], walk["time"]
                )
            planner_version = (
                json.loads(str(cache["planner_source_sha256"]))
                if "planner_source_sha256" in cache
                else None
            )
    else:
        if use_feedback:
            raise ValueError("Corridor feedback requires a fingerprinted lateral walking reference")
        planner_version = planner_source_hashes()
        walk = sample_walk(geometry, segments)
    if use_feedback and (windows is None or args.extent == "approach"):
        raise ValueError(
            "Corridor feedback requires the post-manipulation portion of the lateral walk"
        )
    opening_approach = None
    if getattr(args, "opening_stance_feedback", False):
        from stair_open_approach import approach_with_rear_foot_offset

        from go2_wall.stair_open_balance import OpeningStanceControl

        feet = study["candidates"][4]["feet_xyz_m"]
        walk, ends, opening_approach = approach_with_rear_foot_offset(
            geometry, segments, ends, anchor, walk, feet, OpeningStanceControl.REAR_OFFSET_M
        )
    control_profile = None
    if getattr(args, "control_profile", None) == RL_REPLANT_PROFILE:
        walk, ends, control_profile = apply_replant(geometry, segments, ends, walk)
        anchor += 1
        if windows is not None:
            windows += control_profile["duration_s"]
    track, phase, command, saved, new_ends, details, arm_command, feedback = make_mission(
        geometry, seq, study, segments, ends, anchor, walk, args, settings
    )
    stance_spring_active_s = None
    if details:
        boundaries = details["boundaries_s"]
        start_task = ends[anchor][0] + SHIFT_SECONDS
        stance_spring_active_s = (
            start_task + boundaries["contact"],
            start_task + boundaries["hold"],
        )
    model, data = stair_valve.build_grasp_scene(
        settings["slope_deg"],
        settings["rise_m"],
        saved["centre_m"],
        saved["start_deg"],
        saved["sense"],
        settings["handle_length_m"],
        settings["resistance_nm"],
        floor_mu=settings["floor_mu"],
        pad_mu=settings["pad_mu"],
        finger_force_n=settings["finger_force_n"],
        slide_loss_n=settings["slide_loss_n"],
    )
    limits = configure_mission_torque_limits(model, args.arm_limits)
    walking_control = None
    if use_feedback:
        up_start, up_end, down_start, down_end = windows + new_ends[-1][0] - ends[-1][0]
        active = [(up_start, up_end, 0.016), (down_start, down_end, 0.0)]
        walking_control = StairWalkingControl(
            model, track, active, TIMESTEP_S, settings["floor_mu"]
        )
    elif getattr(args, "opening_stance_feedback", False):
        from go2_wall.stair_open_balance import OpeningStanceControl

        walking_control = OpeningStanceControl(
            model, ends[anchor][0], start_task + boundaries["grip_start"]
        )
    for name in SLIDES:
        track["qpos"][0, model.joint(name).qposadr[0]] = grasp_motion.OPEN_FINGER_M
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "extent": args.extent,
        "case": args.case,
        "settings": settings,
        "centre_m": saved["centre_m"],
        "start_deg": saved["start_deg"],
        "sense": saved["sense"],
        "timestep_s": args.timestep,
        "actuation_period_s": TIMESTEP_S,
        "impratio": args.impratio or float(model.opt.impratio),
        "gain": args.gain,
        "jaw_load_bias_m": args.jaw_load_bias_m,
        "stance_spring_gain_n_per_m": args.stance_spring_gain_n_per_m,
        "stance_spring_active_s": stance_spring_active_s,
        "extra_settle_s": args.extra_settle_seconds,
        "arm_limits": args.arm_limits,
        "joint_limits_nm": limits.tolist(),
        "anchor_time_s": ends[anchor][0],
        "motion": details,
        "source_sha256": versions,
        "planner_source_sha256": planner_version,
    }
    if walking_reference is not None:
        manifest["walking_reference"] = walking_reference
    if opening_approach is not None:
        manifest["opening_approach"] = opening_approach
    if control_profile is not None:
        manifest["control_profile"] = control_profile
    if walking_control:
        manifest["walking_control"] = walking_control.describe()
    if args.extent == "open-at-station":
        manifest["valve_operation"] = "open"
        manifest["post_retract_station_s"] = STATION_SECONDS
        manifest["station_base_pose"] = saved["base_pose"]
    write_json(args.out / "scene.json", manifest)
    manifest_hash = hashlib.sha256((args.out / "scene.json").read_bytes()).hexdigest()
    physics = {"timestep": args.timestep}
    if args.impratio is not None:
        physics["impratio"] = args.impratio
    load_arrays = {}
    load_observer = StairLoads(model) if args.capture_loads else None
    step_count = transition.physics_step_count(track["time"][-1], args.timestep)
    observed = 0
    progress_every = max(1, round(120.0 / args.timestep))

    def observer(m, d):
        nonlocal observed
        if load_observer is not None:
            for name, values in load_observer.sample(d).items():
                if name not in load_arrays:
                    load_arrays[name] = np.empty((step_count, *values.shape), dtype=values.dtype)
                load_arrays[name][observed] = values
        observed += 1
        if args.extent == "full" and observed % progress_every == 0:
            print(f"{observed}/{step_count} physics steps, {d.time:.1f} s", flush=True)
        return observe(m, d)

    metrics, log = transition.execute(
        track,
        None,
        floor_mu=settings["floor_mu"],
        physics=physics,
        builder=lambda: (model, data),
        watch=observer,
        limits_nm=limits,
        gains_scale=seq["gain_scale"],
        swing_from_plan=seq["swing_from_plan"],
        log_every_step=True,
        actuator_command=command,
        arm_target_command=arm_command,
        actuation_period_s=TIMESTEP_S,
        task_contact_pairs=PAD_PAIRS,
        stance_xy_spring_n_per_m=args.stance_spring_gain_n_per_m,
        stance_xy_spring_active_s=stance_spring_active_s,
        stance_slip_spring_n_per_m=(
            SLIP_SPRING_GAIN_N_PER_M if control_profile is not None else 0.0
        ),
        stance_slip_spring_active_s=(SLIP_SPRING_ACTIVE_S if control_profile is not None else None),
        stance_slip_spring_legs=(("RL",) if control_profile is not None else ()),
        walking_control=walking_control,
    )
    stages = [phase(t) for t in log["time"]]
    trace, loads = trace_arrays(model, log, stages, settings["floor_mu"])
    metadata = audit_metadata(
        model, metrics, resistance_nm=settings["resistance_nm"], every_step=True
    )
    if args.extent == "open-at-station":
        metadata["valve_operation"] = "open"
        metadata["stage_map"] = {"lower": ["position"]}
    if observed != step_count:
        raise ValueError(f"Expected {step_count} physical observations, got {observed}")
    with (args.out / "mission.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            **trace,
            qpos=log["qpos"],
            qvel=log["qvel"],
            ctrl=log["ctrl"],
            torque_nm=log["torque"],
            foot_force_world_n=log["force"],
            tread_load_n=loads,
            requested_finger_force_n=log["requested_finger_force_n"],
            **{k: log[k] for k in ("arm_target_rad", "nominal_arm_target_rad") if k in log},
            **load_arrays,
        )
    walking = measured_walk(
        geometry, model, new_ends, log, loads, trace, limits, full=args.extent == "full"
    )
    with (args.out / "mission.npz").open("rb") as raw:
        raw_hash = hashlib.file_digest(raw, "sha256").hexdigest()
    result = {
        "extent": args.extent,
        "metrics": metrics,
        "audit": metadata,
        "walking": walking,
        "settings": settings,
        "scene_sha256": manifest_hash,
        "raw_sha256": raw_hash,
    }
    if control_profile is not None:
        replant_time_s = control_profile["replant_time_s"]
        trigger_time_s = control_profile["trigger_time_s"]
        profile_landing = new_ends[control_profile["trigger_landing_index"] + 1]
        frames = (log["time"] >= trigger_time_s) & (log["time"] < replant_time_s)
        leg_index = 2
        start = int(np.flatnonzero(log["time"] < trigger_time_s)[-1])
        replant_log = {**log, "extra": loads}
        physical_replant = {
            "original_scheduled_landings": len(new_ends) - 1,
            "auxiliary_replant_landings": 1,
            "total_scheduled_landings": len(new_ends),
            "minimum_total_replant_leg_load_n": float(loads[frames, leg_index].sum(axis=1).min()),
            "maximum_foot_lift_m": float(
                log["feet"][frames, leg_index, 2].max() - log["feet"][start, leg_index, 2]
            ),
            "released_from_support": bool(loads[frames, leg_index].sum(axis=1).min() < 1.0),
            "relanding": measure_milestone(replant_log, geometry, *profile_landing),
        }
        physical_replant["verified"] = bool(
            physical_replant["released_from_support"]
            and physical_replant["maximum_foot_lift_m"] > 0.0
            and physical_replant["relanding"]["support"]["all_expected_feet_ge_1n"]
            and physical_replant["relanding"]["terminal_foot_xy_error_m"] <= MAX_FOOT_XY_ERROR_M
        )
        result["control_profile"] = {
            **control_profile,
            "physical_replant": physical_replant,
        }
        if not physical_replant["verified"]:
            walking = with_failed_replant(walking, replant_time_s)
            result["walking"] = walking
    if feedback is not None:
        result["arm_feedback"] = feedback.details()
    if args.extent != "approach":
        result["valve"] = mission_verdict.evaluate_valve(trace, metadata)
    if args.extent == "open-at-station":
        result["station"] = evaluate_station(
            model,
            study["candidates"][4],
            saved["base_pose"],
            trace,
            log["qpos"],
            log["qvel"],
            loads,
            walking["landings_evaluated"],
            len(new_ends),
        )
    if args.extent == "full":
        n1_log = {**log, "extra": loads}
        result["route"] = evaluate_sequence(
            geometry, segments, new_ends, model, metrics, n1_log, round_trip=True
        )
        result["route"]["walking_failures"] = walking["failures"]
        result["mission"] = mission_verdict.evaluate_mission(trace, metadata, result["route"])
        if args.low_run is not None:
            if args.low_run.suffix == ".gz":
                with gzip.open(args.low_run, "rt", encoding="utf-8") as stream:
                    low_result = json.load(stream)
            else:
                low_result = load(args.low_run)
            result["validation"] = mission_verdict.evaluate_validation(
                {k: np.asarray(v) for k, v in low_result["trace"].items()},
                trace,
                result["route"],
                low_load_metadata=low_result["metadata"],
                mission_metadata=metadata,
            )
    first_failure = first_component_failure(
        result.get(component)
        for component in ("walking", "valve", "mission", "validation", "station")
    )
    if first_failure is not None:
        result["outcome"] = "physical_failed"
        result["first_failure"] = first_failure
    else:
        result["outcome"] = "success"
    write_json(args.out / "result.json", result)
    print(
        f"{len(stages)} physics steps, {metrics['seconds']:.2f}s, "
        f"fell={metrics['fell']}, outcome={result['outcome']}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--walk", type=Path)
    parser.add_argument("--low-run", type=Path, help="separate 1 N m replay_contact_valve JSON")
    parser.add_argument("--case", choices=load(PROTOCOL)["single_factor_cases"], default="nominal")
    parser.add_argument(
        "--extent", choices=("approach", "through-task", "full", "open-at-station"), default="full"
    )
    parser.add_argument("--gain", type=float, default=2.5)
    parser.add_argument("--motion-seconds", type=float)
    parser.add_argument("--timestep", type=float)
    parser.add_argument("--impratio", type=float)
    parser.add_argument("--arm-limits", choices=("rb1", "lims-ex"), default="rb1")
    parser.add_argument("--capture-loads", action="store_true")
    parser.add_argument(
        "--corridor-feedback",
        action="store_true",
        help="Use measured whole-body motor feedback alongside the closed valve",
    )
    parser.add_argument("--jaw-load-bias-m", type=float)
    parser.add_argument(
        "--terminal-overtravel-deg",
        type=float,
        help="Extra planned tool angle at the rated stroke endpoint, unwound during settling",
    )
    parser.add_argument("--stance-spring-gain-n-per-m", type=float)
    parser.add_argument("--extra-settle-seconds", type=float)
    parser.add_argument("--opening-stance-feedback", action=argparse.BooleanOptionalAction)
    parser.add_argument("--control-profile", choices=(RL_REPLANT_PROFILE,))
    for field in (
        "slope_deg",
        "rise_m",
        "floor_mu",
        "pad_mu",
        "resistance_nm",
        "handle_length_m",
        "finger_force_n",
        "slide_loss_n",
    ):
        parser.add_argument("--" + field.replace("_", "-"), type=float)
    args = parser.parse_args()
    opening = args.extent == "open-at-station"
    if args.opening_stance_feedback is None:
        args.opening_stance_feedback = opening
    if args.opening_stance_feedback and not opening:
        parser.error("Opening stance feedback requires the open-at-station endpoint")
    if args.opening_stance_feedback and args.control_profile:
        parser.error("The two profiles prescribe different rear-foot landings")
    for name, standard, at_station in (
        ("motion_seconds", 8.75, 7.5 if args.opening_stance_feedback else 9.0),
        ("jaw_load_bias_m", 0.0, 0.0005),
        ("terminal_overtravel_deg", 0.0, 1.2),
        ("stance_spring_gain_n_per_m", 0.0, 2500.0),
        ("extra_settle_seconds", 0.0, 1.4 if args.opening_stance_feedback else 5.0),
    ):
        if getattr(args, name) is None:
            setattr(args, name, at_station if opening else standard)
    if args.out.exists():
        parser.error(f"Output exists: {args.out}")
    if args.extent == "open-at-station" and (args.corridor_feedback or args.low_run):
        parser.error(
            "The station endpoint does not use the post-task corridor or closing validation"
        )
    if not np.isfinite(args.jaw_load_bias_m):
        parser.error("Jaw load bias must be finite")
    if not np.isfinite(args.stance_spring_gain_n_per_m) or args.stance_spring_gain_n_per_m < 0:
        parser.error("Stance spring gain must be finite and non-negative")
    if args.extent == "approach" and args.stance_spring_gain_n_per_m > 0:
        parser.error("Stance spring requires a task extent")
    if not np.isfinite(args.extra_settle_seconds) or args.extra_settle_seconds < 0:
        parser.error("Extra settle duration must be finite and non-negative")
    if args.control_profile == RL_REPLANT_PROFILE:
        if args.walk is None or args.extent == "approach":
            parser.error("The RL replant profile requires a through-task or full walk")
        args.motion_seconds = 9.0
        args.timestep = 0.001 if args.timestep is None else args.timestep
        args.jaw_load_bias_m = 0.0005
        args.stance_spring_gain_n_per_m = 2500.0
        args.extra_settle_seconds = 0.0
        args.terminal_overtravel_deg = 1.3
    elif args.timestep is None:
        args.timestep = 0.001 if opening else 0.005
    run(args)


if __name__ == "__main__":
    main()
