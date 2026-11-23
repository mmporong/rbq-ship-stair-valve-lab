"""Replay a contact-only RB1 grasp and passive stair-valve operation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import stair_mission_trace
from replay_stair_transfer import TIMESTEP_S, expected_surfaces

from go2_wall import grasp_motion, stair_valve, transition
from go2_wall.robot import LEGS
from go2_wall.statics import torque_limits

SOURCE_FILES = (
    "src/go2_wall/grasp_motion.py",
    "tools/replay_contact_valve.py",
    "src/go2_wall/transition.py",
    "src/go2_wall/gripper_control.py",
    "src/go2_wall/stair_valve.py",
    "src/go2_wall/mission_verdict.py",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def source_sha256_lf() -> dict[str, str]:
    """Hash the control and verdict sources after normalizing line endings to LF."""
    root = Path(__file__).resolve().parents[1]
    hashes = {}
    for relative in SOURCE_FILES:
        normalized = (root / relative).read_text(encoding="utf-8").replace("\r\n", "\n")
        hashes[relative] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return hashes


def run(
    study: dict,
    geometry: dict,
    *,
    candidate: int,
    target_deg: float,
    resistance_nm: float,
    motion_s: float,
    gain: float,
    timestep: float,
    finger_force_n: float,
    pad_mu: float,
    slide_loss_n: float,
    open_hand_control: bool = False,
    jaw_load_bias_m: float = 0.0,
    stance_spring_gain_n_per_m: float = 0.0,
    extra_settle_s: float = 0.0,
) -> dict:
    version_at_start = source_sha256_lf()
    saved = {**study["candidates"][candidate], "resistance_nm": resistance_nm}
    model, data = stair_valve.build_grasp_scene(
        study["slope_deg"],
        study["rise_m"],
        saved["centre_m"],
        saved["start_deg"],
        saved["sense"],
        saved["handle_length_m"],
        resistance_nm,
        floor_mu=stair_valve.FLOOR_MU,
        pad_mu=pad_mu,
        finger_force_n=finger_force_n,
        slide_loss_n=slide_loss_n,
    )
    track_study = {**study, "arm_stow_rad": geometry["arm_stow_rad"]}
    track, planned = grasp_motion.make_grasp_track(
        model,
        data,
        track_study,
        saved,
        target_deg=target_deg,
        motion_s=motion_s,
        resistance_nm=resistance_nm,
        jaw_load_bias_m=jaw_load_bias_m,
        durations_s={
            "settle": grasp_motion.DEFAULT_DURATIONS_S["settle"] + extra_settle_s
        },
    )
    stance = {
        leg: (np.asarray(saved["feet_xyz_m"][leg]), "floor") for leg in LEGS
    }
    treads = tuple(
        expected_surfaces(stance, study["rise_m"], study["foot_radius_m"])
    )
    boundaries = planned["boundaries_s"]
    arm_command = grasp_motion.make_arm_target_command(
        saved,
        boundaries,
        planned["command_target_deg"],
        motion_s,
        planned["hold_target_deg"],
        jaw_load_bias_m,
    )

    def command(time_s: float, position_m: float, velocity_mps: float) -> float:
        if open_hand_control:
            return grasp_motion.open_finger_command(
                position_m, velocity_mps, force_limit_n=finger_force_n
            )
        return grasp_motion.finger_command(
            time_s,
            position_m,
            velocity_mps,
            boundaries=boundaries,
            force_limit_n=finger_force_n,
        )

    metrics, log = transition.execute(
        track,
        None,
        physics={"timestep": timestep},
        builder=lambda: (model, data),
        watch=stair_mission_trace.observe,
        gains_scale=gain,
        limits_nm=torque_limits(study["arm"]),
        log_every_step=True,
        actuator_command=command,
        finger_actuator=stair_valve.GRIPPER_ACTUATOR,
        task_contact_pairs=stair_mission_trace.PAD_PAIRS,
        arm_target_command=arm_command,
        actuation_period_s=TIMESTEP_S,
        stance_xy_spring_n_per_m=stance_spring_gain_n_per_m,
        stance_xy_spring_active_s=(boundaries["contact"], boundaries["hold"]),
    )
    stages = [
        grasp_motion.stage_at(float(time), boundaries, boundaries["operation_name"])
        for time in log["time"]
    ]
    trace, tread_loads = stair_mission_trace.trace_arrays(
        model, log, stages, stair_valve.FLOOR_MU
    )
    trace["phase"] = trace["stage"].copy()
    trace["finger_ctrl_n"] = log["ctrl"][
        :, model.actuator(stair_valve.GRIPPER_ACTUATOR).id
    ]
    trace["robot_ctrl_nm"] = log["ctrl"][:, :18]
    trace["nominal_arm_target_rad"] = log["nominal_arm_target_rad"]
    trace["arm_target_rad"] = log["arm_target_rad"]
    trace["arm_feedback_raw_target_rad"] = np.asarray(
        arm_command.raw_target_history_rad
    )
    trace["arm_feedback_applied_target_rad"] = np.asarray(
        arm_command.applied_target_history_rad
    )
    metadata = stair_mission_trace.audit_metadata(
        model, metrics, resistance_nm=resistance_nm, every_step=True
    )
    metadata["valve_resistance_nm"] = resistance_nm
    metadata["robot_saturated_steps"] = metrics["saturated_steps"]
    metadata["finger_saturated_steps"] = metadata["finger_clipped_steps"]
    metadata["arm_target_feedback"] = arm_command.details()
    metadata["control_source"] = (
        "open_hand_falsification" if open_hand_control else "closed_loop_two_pad_grasp"
    )
    metadata["actuation_period_s"] = TIMESTEP_S
    metadata["jaw_load_bias_m"] = jaw_load_bias_m
    metadata["stance_spring_gain_n_per_m"] = stance_spring_gain_n_per_m
    metadata["extra_settle_s"] = extra_settle_s
    if source_sha256_lf() != version_at_start:
        raise ValueError("Replay sources changed during the trial")
    metadata["source_sha256_lf"] = version_at_start
    return {
        "candidate": candidate,
        "target_deg": target_deg,
        "resistance_nm": resistance_nm,
        "foot_order": LEGS,
        "foot_treads": treads,
        "stage_alias": "phase is an exact alias of stage",
        "actuation_period_s": TIMESTEP_S,
        "physics_timestep_s": timestep,
        "jaw_load_bias_m": jaw_load_bias_m,
        "stance_spring_gain_n_per_m": stance_spring_gain_n_per_m,
        "extra_settle_s": extra_settle_s,
        "source_sha256_lf": metadata["source_sha256_lf"],
        "plan": json_ready(planned),
        "controller_metrics": metrics,
        "metadata": json_ready(metadata),
        "tread_loads_n": tread_loads.tolist(),
        "trace": json_ready(trace),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--candidate", type=int, default=4)
    parser.add_argument("--target-deg", type=float, choices=(20.0, 90.0), default=20.0)
    parser.add_argument("--resistance-nm", type=float, default=1.0)
    parser.add_argument("--motion-seconds", type=float, default=4.0)
    parser.add_argument("--gain", type=float, default=2.5)
    parser.add_argument("--timestep", type=float, default=0.005)
    parser.add_argument("--finger-force-n", type=float, default=100.0)
    parser.add_argument("--pad-mu", type=float, default=0.8)
    parser.add_argument("--slide-loss-n", type=float, default=30.0)
    parser.add_argument("--open-hand-control", action="store_true")
    parser.add_argument("--jaw-load-bias-m", type=float, default=0.0)
    parser.add_argument("--stance-spring-gain-n-per-m", type=float, default=0.0)
    parser.add_argument("--extra-settle-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    positive = (
        args.resistance_nm,
        args.motion_seconds,
        args.gain,
        args.timestep,
        args.finger_force_n,
        args.pad_mu,
    )
    if any(not np.isfinite(value) or value <= 0 for value in positive):
        parser.error("Resistance, durations, gains, timestep, force and friction must be positive")
    if not np.isfinite(args.slide_loss_n) or args.slide_loss_n < 0:
        parser.error("Slide loss must be finite and non-negative")
    if not np.isfinite(args.jaw_load_bias_m):
        parser.error("Jaw load bias must be finite")
    if (
        not np.isfinite(args.stance_spring_gain_n_per_m)
        or args.stance_spring_gain_n_per_m < 0
    ):
        parser.error("Stance spring gain must be finite and non-negative")
    if not np.isfinite(args.extra_settle_seconds) or args.extra_settle_seconds < 0:
        parser.error("Extra settle duration must be finite and non-negative")
    result = run(
        read(args.study),
        read(args.geometry),
        candidate=args.candidate,
        target_deg=args.target_deg,
        resistance_nm=args.resistance_nm,
        motion_s=args.motion_seconds,
        gain=args.gain,
        timestep=args.timestep,
        finger_force_n=args.finger_force_n,
        pad_mu=args.pad_mu,
        slide_loss_n=args.slide_loss_n,
        open_hand_control=args.open_hand_control,
        jaw_load_bias_m=args.jaw_load_bias_m,
        stance_spring_gain_n_per_m=args.stance_spring_gain_n_per_m,
        extra_settle_s=args.extra_settle_seconds,
    )
    save(args.out, result)
    print(
        f"Maximum angle: {max(result['trace']['valve_angle_deg']):.3f} deg; "
        f"maximum pad torque: {max(result['trace']['pad_closing_torque_nm']):.3f} N m",
        flush=True,
    )


if __name__ == "__main__":
    main()
