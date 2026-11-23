"""Translate observations of the live contact scene into the shared mission verdict schema."""

import mujoco
import numpy as np
from replay_stair_sequence import (
    DWELL_S,
    MAX_FOOT_XY_ERROR_M,
    RETURN_ANGULAR_SPEED_LIMIT_RADPS,
    RETURN_HOLD_S,
    RETURN_LINEAR_SPEED_LIMIT_MPS,
    RETURN_TILT_LIMIT_DEG,
    RETURN_XY_LIMIT_M,
    START_HOLD_S,
    TURN_TOLERANCE_DEG,
)
from replay_stair_transfer import LOAD_THRESHOLD_N, SURFACES, expected_surfaces, support_loads

from go2_wall import stair_valve, transition
from go2_wall.robot import LEGS

PAD_PAIRS = tuple((name, stair_valve.VALVE_LEVER) for name in stair_valve.GRIPPER_PADS)
SLIDES = ("RB1_finger_upper_joint", "RB1_finger_lower_joint")
OBSERVATIONS = (
    ("valve_angle_deg", 1),
    ("valve_speed_rad_s", 1),
    ("hinge_friction_torque_nm", 1),
    ("signed_hinge_friction_torque_nm", 1),
    ("pad_closing_torque_nm", 1),
    ("pad_closing_tangent_force_n", 1),
    ("pad_normal_force_n", 2),
    ("pad_surface_gap_m", 2),
    ("handle_axis_offset_m", 1),
    ("handle_axial_position_m", 1),
    ("max_unexpected_lever_contact_n", 1),
)


def observe(model, data):
    sample = stair_valve.observe_grasp(model, data)
    fields = [np.atleast_1d(sample[name]) for name, _ in OBSERVATIONS]
    return np.concatenate([*fields, support_loads(model, data).ravel()])


def trace_arrays(model, log, stages, floor_mu):
    """Use 4×12 actual surface loads. M1 refers to the four named manipulation treads."""
    times = log["physics_time_s"]
    stages = np.asarray(stages, dtype=str)
    if stages.shape != times.shape or floor_mu <= 0:
        raise ValueError("Inconsistent stage labels or nonpositive ground friction")
    trace = {"time_s": times, "stage": stages}
    col = 0
    for name, width in OBSERVATIONS:
        trace[name] = log["extra"][:, col] if width == 1 else log["extra"][:, col : col + width]
        col += width
    loads = log["extra"][:, col:].reshape(-1, len(LEGS), len(SURFACES))
    trace["handle_nonpad_force_n"] = trace.pop("max_unexpected_lever_contact_n")
    trace["base_xyz_m"] = log["base"][:, :3]
    trace["feet_xyz_m"] = log["feet"]
    trace["foot_normal_force_n"] = loads[:, np.arange(4), [5, 5, 3, 3]]
    force = log["force"]
    trace["foot_friction_use"] = (
        np.linalg.norm(force[..., :2], axis=-1) / np.maximum(force[..., 2], 1.0) / floor_mu
    )
    trace["unexpected_contact_force_n"] = log["unexpected_contact_n"]
    for key in ("base_track_m", "stance_slip_m", "robot_torque_requested_nm", "fell"):
        if key in log:
            trace[key] = log[key]
    trace["qp_attempt"] = log["attempt"]
    slide_positions = [model.joint(name).qposadr[0] for name in SLIDES]
    trace["finger_slide_m"] = log["qpos"][:, slide_positions]
    # The coupled slides separate the pads by twice their travel. 30 mm clears the 20 mm lever.
    trace["gripper_open"] = trace["finger_slide_m"].sum(axis=1) >= 0.03
    return trace, loads


def audit_metadata(model, metrics, *, resistance_nm, every_step):
    audit = dict(metrics["gripper_audit"])
    fingers = {model.joint(name).id for name in SLIDES}
    equality_objects = zip(model.eq_type, model.eq_obj1id, model.eq_obj2id, strict=True)
    linked_fingers = sum(
        eq_type == mujoco.mjtEq.mjEQ_JOINT and {first, second} == fingers
        for eq_type, first, second in equality_objects
    )
    hinge = model.joint(stair_valve.VALVE_HINGE)
    measured_setting = float(model.dof_frictionloss[hinge.dofadr[0]])
    if not np.isclose(measured_setting, resistance_nm):
        raise ValueError("The specified trial resistance differs from the compiled hinge")
    hinge_motors = np.count_nonzero(
        (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
        & (model.actuator_trnid[:, 0] == hinge.id)
    )
    audit.update(
        physics_timestep_s=float(model.opt.timestep),
        record_every_step=every_step,
        finger_equality_count=int(linked_fingers),
        other_equality_count=model.neq - int(linked_fingers),
        hinge_actuator_count=int(hinge_motors),
        valve_resistance_nm=measured_setting,
        robot_saturated_steps=metrics["saturated_steps"],
        finger_saturated_steps=audit["finger_clipped_steps"],
        qp_failed_ticks=metrics["qp_failed_ticks"],
        qp_relaxed_ticks=metrics["qp_relaxed_ticks"],
        fell=metrics["fell"],
    )
    return audit


def measured_walk(geometry, model, ends, log, loads, trace, limits, *, full=False):
    """Check the observed part of the walk, retaining the first offending physical sample."""
    time = trace["time_s"]
    stages = trace["stage"]
    faults = []
    n = np.size(time)
    shapes = {
        "time_s": (n,),
        "time": (n,),
        "base": (n, 7),
        "feet": (n, 4, 3),
        "tread_load_n": (n, 4, len(SURFACES)),
        "robot_torque_requested_nm": (n, 18),
        "torque_limit_nm": (18,),
        "pad_normal_force_n": (n, 2),
    }

    for name, values in {
        "time_s": time,
        "tread_load_n": loads,
        "torque_limit_nm": limits,
        **{key: log[key] for key in ("base", "feet", "qpos", "qvel", "time")},
        **{
            key: trace[key]
            for key in (
                "base_track_m",
                "stance_slip_m",
                "fell",
                "qp_attempt",
                "robot_torque_requested_nm",
                "unexpected_contact_force_n",
                "pad_normal_force_n",
            )
        },
    }.items():
        values = np.asarray(values)
        if name in ("qpos", "qvel"):
            valid_shape = (
                values.ndim == 2
                and values.shape[0] == n
                and values.shape[1] >= (7 if name == "qpos" else 6)
            )
        else:
            valid_shape = values.shape == shapes.get(name, (n,))
        if not n or not valid_shape:
            faults.append(
                dict(
                    stage="input",
                    time_s=None,
                    metric="shape_" + name,
                    measured=list(values.shape),
                    limit="per-sample array with expected dimensions",
                )
            )
            continue
        invalid = np.argwhere(~np.isfinite(values))
        if invalid.size:
            i = int(invalid[0, 0]) if name != "torque_limit_nm" else 0
            stamp = (
                float(time[i])
                if np.ndim(time) == 1 and i < len(time) and np.isfinite(time[i])
                else None
            )
            faults.append(
                dict(
                    stage="input",
                    time_s=stamp,
                    metric="nonfinite_" + name,
                    measured=str(values[tuple(invalid[0])]),
                    limit="finite",
                )
            )
    if faults:
        faults.sort(key=lambda f: (f["time_s"] is None, f["time_s"] or 0.0))
        return {
            "landings_evaluated": 0,
            "verified": False,
            "first_failure": faults[0],
            "failures": faults,
        }

    def first(metric, values, mask, limit):
        indices = np.flatnonzero(mask)
        if len(indices):
            k = indices[0]
            faults.append(
                dict(
                    stage=str(stages[k]),
                    time_s=float(time[k]),
                    metric=metric,
                    measured=float(values[k]),
                    limit=limit,
                )
            )

    checks = [
        ("base_track_m", trace["base_track_m"], transition.BASE_TRACK_LIMIT_M),
        ("stance_slip_m", trace["stance_slip_m"], transition.STANCE_SLIP_LIMIT_M),
        (
            "requested_torque_ratio",
            np.max(np.abs(trace["robot_torque_requested_nm"]) / limits, axis=1),
            1.0,
        ),
        ("fell", trace["fell"], 0),
    ]
    geom = geometry["stair_geometry"]
    positions = np.concatenate((log["feet"], log["base"][:, None, :3]), axis=1)
    platform = model.geom("platform")
    start_x = platform.pos[0] - platform.size[0]
    on_stairs = (positions[..., 0] >= 0) & (positions[..., 0] < start_x)
    lateral = np.max(np.where(on_stairs, np.abs(positions[..., 1]), 0), axis=1)
    checks.append(("tread_lateral_position_m", lateral, geom["clear_width_m"] / 2))
    outside = np.max(np.abs(positions[..., :2] - platform.pos[:2]) - platform.size[:2], axis=2)
    beyond = np.max(np.where(positions[..., 0] >= start_x, outside, 0.0), axis=1)
    checks.append(("platform_boundary_excess_m", beyond, 0))
    for metric, values, upper in checks:
        first(metric, values, values > upper, f"<= {upper}")
    attempts = trace["qp_attempt"]
    first("qp_attempt", attempts, attempts != 0, "== 0")
    start = (log["time"] >= 0.2) & (log["time"] < START_HOLD_S)
    on_floor = loads[:, :, 0].min(axis=1)
    first("start_floor_support_n", on_floor, start & (on_floor < LOAD_THRESHOLD_N), ">= 1 N")
    first(
        "unexpected_contact_force_n",
        trace["unexpected_contact_force_n"],
        trace["unexpected_contact_force_n"] > 10,
        "<= 10 N",
    )
    floor = np.max(np.where(log["feet"][..., 0] > geom["tread_depth_m"], loads[..., 0], 0), axis=1)
    first("floor_support_past_tread_n", floor, floor >= LOAD_THRESHOLD_N, "< 1 N")
    unscheduled = np.isin(stages, ("prewalk", "lower", "position", "return", "station"))
    pad = np.max(trace["pad_normal_force_n"], axis=1)
    first("pad_contact_during_walk_n", pad, unscheduled & (pad > 1e-6), "<= 1e-6 N")
    evaluated = 0
    platform_headings = []
    for end, stance, leg in ends:
        if end > time[-1] + 1e-7:
            continue
        # Use the same command-time window as the frozen N1 judge; observations are post-step.
        window = (log["time"] >= end - DWELL_S) & (log["time"] < end)
        indices = np.flatnonzero(window)
        if not len(indices):
            continue
        evaluated += 1
        surfaces = expected_surfaces(stance, geometry["rise_m"], geometry["foot_radius_m"])
        supporting = loads[indices][:, np.arange(4), surfaces].min(axis=1)
        missed = np.flatnonzero(supporting < LOAD_THRESHOLD_N)
        if len(missed):
            k = indices[missed[0]]
            faults.append(
                dict(
                    stage=str(stages[k]),
                    time_s=float(time[k]),
                    metric="landing_support_n",
                    measured=float(supporting[missed[0]]),
                    limit=">= 1 N",
                    landing=evaluated,
                )
            )
        k = indices[-1]
        if all(s == len(SURFACES) - 1 for s in surfaces) and not len(missed):
            rot = np.empty(9)
            mujoco.mju_quat2Mat(rot, log["qpos"][k, 3:7])
            platform_headings.append((k, np.degrees(np.arctan2(rot[3], rot[0]))))
        xy = np.linalg.norm(log["feet"][k, LEGS.index(leg), :2] - stance[leg][0][:2])
        if xy > MAX_FOOT_XY_ERROR_M:
            faults.append(
                dict(
                    stage=str(stages[k]),
                    time_s=float(time[k]),
                    metric="landing_xy_m",
                    measured=float(xy),
                    limit=f"<= {MAX_FOOT_XY_ERROR_M}",
                    landing=evaluated,
                )
            )
    if full:
        final = log["time"] >= time[-1] - RETURN_HOLD_S - 1e-8
        poses = log["qpos"]
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (poses[:, 4] ** 2 + poses[:, 5] ** 2), -1, 1)))
        for name, values, limit in (
            ("return_xy_m", np.linalg.norm(poses[:, :2] - poses[0, :2], axis=1), RETURN_XY_LIMIT_M),
            ("return_tilt_deg", tilt, RETURN_TILT_LIMIT_DEG),
            (
                "return_speed_mps",
                np.linalg.norm(log["qvel"][:, :3], axis=1),
                RETURN_LINEAR_SPEED_LIMIT_MPS,
            ),
            (
                "return_angular_speed_radps",
                np.linalg.norm(log["qvel"][:, 3:6], axis=1),
                RETURN_ANGULAR_SPEED_LIMIT_RADPS,
            ),
        ):
            first(name, values, final & (values > limit), f"<= {limit}")
        first("return_floor_support_n", on_floor, final & (on_floor < LOAD_THRESHOLD_N), ">= 1 N")
        smallest_turn_error = min(
            (abs((h - platform_headings[0][1]) % 360 - 180) for _, h in platform_headings),
            default=180.0,
        )
        if smallest_turn_error > TURN_TOLERANCE_DEG:
            k = platform_headings[-1][0] if platform_headings else len(time) - 1
            faults.append(
                dict(
                    stage="return",
                    time_s=float(time[k]),
                    metric="platform_turn_error_deg",
                    measured=float(smallest_turn_error),
                    limit=f"<= {TURN_TOLERANCE_DEG}",
                )
            )
        if evaluated != 156:
            faults.append(
                dict(
                    stage="return",
                    time_s=float(time[-1]),
                    metric="landing_count",
                    measured=evaluated,
                    limit="== 156",
                )
            )
    faults.sort(key=lambda item: item["time_s"])
    return {
        "landings_evaluated": evaluated,
        "verified": not faults,
        "first_failure": faults[0] if faults else None,
        "failures": faults,
    }
