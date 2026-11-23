"""Score an open valve and a free-standing robot at the prescribed work stance."""

import mujoco
import numpy as np
from replay_stair_sequence import RETURN_ANGULAR_SPEED_LIMIT_RADPS, RETURN_LINEAR_SPEED_LIMIT_MPS

from go2_wall import mission_verdict, stair_valve, transition
from go2_wall.pose_search import set_base

STATION_SECONDS = mission_verdict.POST_RELEASE_OBSERVATION_S


def lever_direction_error_deg(model, pose, target_deg):
    """Measure the tip direction in world XZ relative to the original valve installation."""
    data = mujoco.MjData(model)
    data.qpos[:] = pose
    mujoco.mj_kinematics(model, data)
    radial = (
        data.site(stair_valve.VALVE_HANDLE).xpos
        - data.xanchor[model.joint(stair_valve.VALVE_HINGE).id]
    )
    angle = np.deg2rad(target_deg)
    desired = np.array([np.cos(angle), 0.0, np.sin(angle)])
    if not np.all(np.isfinite(radial)) or np.linalg.norm(radial) <= np.finfo(float).eps:
        raise ValueError("Cannot determine the physical direction of the valve lever")
    return float(
        np.rad2deg(np.arctan2(np.linalg.norm(np.cross(radial, desired)), radial @ desired))
    )


def evaluate_station(
    model, installed, target_pose, trace, poses, velocities, loads, observed_landings, scheduled
):
    time = np.asarray(trace["time_s"])
    stage = np.asarray(trace["stage"])
    station = np.flatnonzero(stage == "station")
    failures = []
    measurements = {}

    def failure(metric, index, measured, limit):
        failures.append(
            dict(
                stage=str(stage[index]),
                time_s=float(time[index]),
                metric=metric,
                measured=float(measured),
                limit=limit,
            )
        )

    dt = float(model.opt.timestep)
    span = len(station) * dt
    measurements["station_s"] = span
    if (
        not len(station)
        or len(station) != len(time) - station[0]
        or abs(span - STATION_SECONDS) > 1e-7
    ):
        failure("terminal_station_s", len(time) - 1, span, f"trailing {STATION_SECONDS} s")
        return dict(
            verified=False, first_failure=failures[0], failures=failures, measurements=measurements
        )

    def check(name, values, limit, *, minimum=False):
        values = np.asarray(values)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite station data: {name}")
        measurements[name] = float(values.min() if minimum else values.max())
        outside = np.flatnonzero(values < limit if minimum else values > limit)
        if outside.size:
            i = outside[0]
            failure(name, station[i], values[i], f"{'>=' if minimum else '<='} {limit}")

    closed_deg = installed["start_deg"] + installed["sense"] * mission_verdict.VALVE_TARGET_DEG
    for label, index, target in (
        ("closed_start_error_deg", 0, closed_deg),
        ("open_finish_error_deg", -1, installed["start_deg"]),
    ):
        error = lever_direction_error_deg(model, poses[index], target)
        measurements[label] = error
        if error > mission_verdict.ANGLE_TOLERANCE_DEG:
            failure(label, index, error, f"<= {mission_verdict.ANGLE_TOLERANCE_DEG}")
    hinge = model.joint(stair_valve.VALVE_HINGE)
    if not np.allclose(
        np.rad2deg(poses[:, hinge.qposadr[0]]), trace["valve_angle_deg"], rtol=0, atol=1e-8
    ):
        raise ValueError("Recorded valve angle differs from the raw hinge position")
    measurements["scheduled_landings"] = int(scheduled)
    measurements["evaluated_landings"] = int(observed_landings)
    if observed_landings != scheduled:
        failure("approach_landings", station[0], observed_landings, f"== {scheduled}")

    ref = mujoco.MjData(model)
    set_base(
        model,
        ref,
        target_pose[3],
        np.asarray(target_pose[:3]),
        yaw=target_pose[4] if len(target_pose) >= 5 else 0,
        roll=target_pose[5] if len(target_pose) >= 6 else 0,
    )
    measured = poses[station]
    check(
        "max_base_position_error_m",
        np.linalg.norm(measured[:, :3] - ref.qpos[:3], axis=1),
        transition.BASE_TRACK_LIMIT_M,
    )
    alignment = np.abs(measured[:, 3:7] @ ref.qpos[3:7])
    check(
        "max_station_attitude_error_deg", np.rad2deg(2 * np.arccos(np.clip(alignment, 0, 1))), 10.0
    )
    check(
        "max_linear_speed_mps",
        np.linalg.norm(velocities[station, :3], axis=1),
        RETURN_LINEAR_SPEED_LIMIT_MPS,
    )
    check(
        "max_angular_speed_radps",
        np.linalg.norm(velocities[station, 3:6], axis=1),
        RETURN_ANGULAR_SPEED_LIMIT_RADPS,
    )
    support = loads[station][:, np.arange(4), [5, 5, 3, 3]]
    check(
        "min_each_prescribed_tread_load_n",
        support.min(axis=1),
        mission_verdict.FOOT_LOAD_N,
        minimum=True,
    )
    check(
        "max_open_angle_error_deg",
        abs(trace["valve_angle_deg"][station] - mission_verdict.VALVE_TARGET_DEG),
        mission_verdict.ANGLE_TOLERANCE_DEG,
    )
    measurements["gripper_open_during_station"] = bool(np.all(trace["gripper_open"][station]))
    closed = np.flatnonzero(~np.asarray(trace["gripper_open"][station], dtype=bool))
    if closed.size:
        failure("gripper_open", station[closed[0]], 0, "== 1")
    check(
        "min_pad_clearance_m",
        trace["pad_surface_gap_m"][station].min(axis=1),
        mission_verdict.RELEASE_GAP_M,
        minimum=True,
    )
    check(
        "max_pad_contact_n",
        trace["pad_normal_force_n"][station].max(axis=1),
        mission_verdict.ZERO_CONTACT_N,
    )
    check(
        "max_nonpad_contact_n",
        trace["handle_nonpad_force_n"][station],
        mission_verdict.ZERO_CONTACT_N,
    )
    failures.sort(key=lambda entry: entry["time_s"])
    return dict(
        verified=not failures,
        first_failure=failures[0] if failures else None,
        failures=failures,
        measurements=measurements,
    )
