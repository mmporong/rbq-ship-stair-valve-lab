"""Verdicts for contact-gripped stair-valve validation traces.

The functions consume saved observations. Missing measurements are never inferred,
so an incomplete trace cannot be reported as successful.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

PAD_CONTACT_N = 1.0
CONTACT_HOLD_S = 0.2
LOW_LOAD_TARGET_DEG = 20.0
LOW_LOAD_TORQUE_NM = 0.98
LOW_LOAD_SLIP_M = 0.002
VALVE_TARGET_DEG = 90.0
ANGLE_TOLERANCE_DEG = 0.25
MOTION_MIN_SPEED_RADPS = 0.02
PAD_LOSS_LIMIT_S = 0.02
HOLD_S = 5.0
BASE_SHIFT_M = 0.010
FOOT_SHIFT_M = 0.005
FRICTION_USE_LIMIT = 0.8
FOOT_LOAD_N = 1.0
RELEASE_GAP_M = 0.005
RELEASE_HOLD_S = 0.5
POST_RELEASE_OBSERVATION_S = 5.0
UNEXPECTED_CONTACT_N = 10.0
ZERO_CONTACT_N = 1e-6

COMMON_FIELDS = (
    "time_s",
    "stage",
    "gripper_open",
    "pad_normal_force_n",
    "valve_angle_deg",
    "valve_speed_rad_s",
    "hinge_friction_torque_nm",
    "pad_closing_torque_nm",
    "handle_axial_position_m",
    "pad_surface_gap_m",
    "handle_nonpad_force_n",
)
M1_FIELDS = (
    "base_xyz_m",
    "feet_xyz_m",
    "foot_normal_force_n",
    "foot_friction_use",
    "unexpected_contact_force_n",
)
METADATA_FIELDS = (
    "physics_timestep_s",
    "record_every_step",
    "state_reset_count",
    "qpos_write_count_after_start",
    "qvel_write_count_after_start",
    "act_write_count_after_start",
    "mocap_write_count_after_start",
    "max_qfrc_applied_n",
    "max_xfrc_applied_n",
    "finger_equality_count",
    "other_equality_count",
    "hinge_actuator_count",
    "valve_resistance_nm",
    "robot_saturated_steps",
    "finger_saturated_steps",
    "qp_failed_ticks",
    "qp_relaxed_ticks",
    "fell",
)


class _Verdict:
    def __init__(self) -> None:
        self.failures: list[dict[str, Any]] = []
        self.measurements: dict[str, Any] = {}

    def fail(self, stage, metric, measured, limit, message, time_s=None) -> None:
        self.failures.append(
            {
                "stage": stage,
                "time_s": time_s,
                "metric": metric,
                "measured": _plain(measured),
                "limit": _plain(limit),
                "message": message,
            }
        )

    def result(self) -> dict[str, Any]:
        indexed = list(enumerate(self.failures))
        indexed.sort(key=lambda item: (item[1]["time_s"] is None, item[1]["time_s"] or 0, item[0]))
        failures = [failure for _, failure in indexed]
        return {
            "verified": not failures,
            "first_failure": failures[0] if failures else None,
            "failures": failures,
            "measurements": self.measurements,
        }


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _load_trace(trace, required, verdict):
    missing = [name for name in required if name not in trace]
    if missing:
        verdict.fail(
            "input", "missing_trace_fields", missing, [], "required observations are missing"
        )
        return None
    arrays = {name: np.asarray(trace[name]) for name in required}
    time = arrays["time_s"]
    if time.ndim != 1 or time.size < 2:
        verdict.fail(
            "input", "time_s", time.shape, "one-dimensional with >=2 samples", "invalid time axis"
        )
        return None
    bad_length = [name for name, value in arrays.items() if len(value) != len(time)]
    if bad_length:
        verdict.fail(
            "input", "trace_lengths", bad_length, len(time), "trace fields have unequal lengths"
        )
        return None
    numeric = [name for name in required if name not in ("stage", "gripper_open")]
    nonfinite = [name for name in numeric if not np.all(np.isfinite(arrays[name]))]
    if nonfinite:
        verdict.fail(
            "input", "nonfinite_trace_fields", nonfinite, [], "trace contains non-finite data"
        )
        return None
    shapes = {
        "pad_normal_force_n": (len(time), 2),
        "pad_surface_gap_m": (len(time), 2),
        "base_xyz_m": (len(time), 3),
        "feet_xyz_m": (len(time), 4, 3),
        "foot_normal_force_n": (len(time), 4),
        "foot_friction_use": (len(time), 4),
    }
    wrong = {
        name: arrays[name].shape
        for name, shape in shapes.items()
        if name in arrays and arrays[name].shape != shape
    }
    if wrong:
        verdict.fail("input", "trace_shapes", wrong, shapes, "trace field shape mismatch")
        return None
    if not np.all(np.diff(time) > 0):
        verdict.fail("input", "time_monotonic", False, True, "time must increase strictly")
        return None
    arrays["stage"] = arrays["stage"].astype(str)
    arrays["gripper_open"] = arrays["gripper_open"].astype(bool)
    return arrays


def _check_metadata(metadata, time, verdict):
    missing = [name for name in METADATA_FIELDS if name not in metadata]
    if missing:
        verdict.fail(
            "input", "missing_metadata_fields", missing, [], "required run metadata is missing"
        )
        return
    timestep = metadata["physics_timestep_s"]
    if not isinstance(timestep, (int, float)) or not np.isfinite(timestep) or timestep <= 0:
        verdict.fail("input", "physics_timestep_s", timestep, "> 0", "invalid physics timestep")
        return
    delta = np.diff(time)
    verdict.measurements["max_record_step_s"] = float(delta.max())
    close = np.isclose(delta, timestep, rtol=1e-6, atol=1e-9)
    if not metadata["record_every_step"] or not np.all(close):
        first = int(np.flatnonzero(~close)[0]) if not np.all(close) else 0
        verdict.fail(
            "continuity",
            "physics_step_record",
            float(delta[first]),
            float(timestep),
            "every physics step must be recorded",
            float(time[first]),
        )
    zero_fields = (
        "state_reset_count",
        "qpos_write_count_after_start",
        "qvel_write_count_after_start",
        "act_write_count_after_start",
        "mocap_write_count_after_start",
        "max_qfrc_applied_n",
        "max_xfrc_applied_n",
        "other_equality_count",
        "hinge_actuator_count",
        "robot_saturated_steps",
        "finger_saturated_steps",
        "qp_failed_ticks",
        "qp_relaxed_ticks",
    )
    for field in zero_fields:
        if metadata[field] != 0:
            verdict.fail(
                "continuity",
                field,
                metadata[field],
                0,
                "run metadata violates the mission contract",
            )
    verdict.measurements["total_saturated_steps"] = (
        metadata["robot_saturated_steps"] + metadata["finger_saturated_steps"]
    )
    if metadata["finger_equality_count"] != 1:
        verdict.fail(
            "continuity",
            "finger_equality_count",
            metadata["finger_equality_count"],
            1,
            "only the coupled-finger equality is required",
        )
    if bool(metadata["fell"]):
        verdict.fail("continuity", "fell", True, False, "the robot fell")
    resistance = metadata["valve_resistance_nm"]
    if not isinstance(resistance, (int, float)) or not np.isfinite(resistance) or resistance < 0:
        verdict.fail(
            "input",
            "valve_resistance_nm",
            resistance,
            ">= 0",
            "invalid valve resistance",
        )


def _stage_mask(arrays, metadata, name):
    aliases = metadata.get("stage_map", {}).get(name, [name])
    aliases = [aliases] if isinstance(aliases, str) else aliases
    return np.isin(arrays["stage"], list(aliases))


def _require_stage_order(arrays, metadata, names, verdict):
    previous = -1
    for name in names:
        indices = np.flatnonzero(_stage_mask(arrays, metadata, name))
        if not indices.size:
            verdict.fail(name, "stage_present", False, True, f"required {name} stage is absent")
            continue
        if indices[0] <= previous:
            verdict.fail(
                name,
                "stage_order",
                int(indices[0]),
                f"> {previous}",
                "mission stages are out of order",
                float(arrays["time_s"][indices[0]]),
            )
        previous = int(indices[-1])


def _first_window(condition, time, duration_s):
    timestep = float(np.median(np.diff(time)))
    start = None
    for index, value in enumerate(condition):
        if value and start is None:
            start = index
        if not value:
            start = None
        elif time[index] - time[start] + timestep >= duration_s - 1e-12:
            return start, index
    return None


def _longest_window(condition, time):
    timestep = float(np.median(np.diff(time)))
    start = None
    best, best_start = 0.0, None
    for index, value in enumerate(condition):
        if value and start is None:
            start = index
        if value:
            duration = float(time[index] - time[start] + timestep)
            if duration > best:
                best, best_start = duration, start
        else:
            start = None
    return best, best_start


def _check_contact(arrays, metadata, verdict):
    approach = _stage_mask(arrays, metadata, "approach")
    contact = _stage_mask(arrays, metadata, "contact")
    bad = approach & ~arrays["gripper_open"]
    if np.any(bad):
        index = int(np.flatnonzero(bad)[0])
        verdict.fail(
            "approach",
            "gripper_open",
            False,
            True,
            "approach did not remain open",
            float(arrays["time_s"][index]),
        )
    secured = contact & np.all(arrays["pad_normal_force_n"] >= PAD_CONTACT_N, axis=1)
    window = _first_window(secured, arrays["time_s"], CONTACT_HOLD_S)
    longest, _ = _longest_window(secured, arrays["time_s"])
    verdict.measurements["contact_hold_s"] = longest
    if window is None:
        verdict.fail(
            "contact",
            "two_pad_contact_hold_s",
            longest,
            f">= {CONTACT_HOLD_S}",
            "both pads did not secure the handle",
            float(arrays["time_s"][np.flatnonzero(contact)[-1]]) if contact.any() else None,
        )
    return window


def evaluate_low_load(trace: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Verify open approach, two-pad contact, and the 1 N m / 20 degree grip test."""
    verdict = _Verdict()
    arrays = _load_trace(trace, COMMON_FIELDS, verdict)
    if arrays is None:
        return verdict.result()
    _check_metadata(metadata, arrays["time_s"], verdict)
    _require_stage_order(arrays, metadata, ("approach", "contact", "low_load"), verdict)
    if metadata.get("valve_resistance_nm") != 1.0:
        verdict.fail(
            "input",
            "valve_resistance_nm",
            metadata.get("valve_resistance_nm"),
            1.0,
            "the low-load proof requires the 1 N m hinge setting",
        )
    contact_window = _check_contact(arrays, metadata, verdict)
    low = _stage_mask(arrays, metadata, "low_load")
    if not np.any(low):
        return verdict.result()
    angle = arrays["valve_angle_deg"]
    first_low = int(np.flatnonzero(low)[0])
    if abs(angle[first_low]) > ANGLE_TOLERANCE_DEG:
        verdict.fail(
            "low_load",
            "start_angle_deg",
            float(angle[first_low]),
            f"abs <= {ANGLE_TOLERANCE_DEG}",
            "low-load valve did not start open",
            float(arrays["time_s"][first_low]),
        )
    maximum = float(angle[low].max())
    verdict.measurements["low_load_max_angle_deg"] = maximum
    if maximum < LOW_LOAD_TARGET_DEG:
        index = int(np.flatnonzero(low)[-1])
        verdict.fail(
            "low_load",
            "max_angle_deg",
            maximum,
            f">= {LOW_LOAD_TARGET_DEG}",
            "low-load rotation did not reach its target",
            float(arrays["time_s"][index]),
        )
    for lower in (5.0, 10.0):
        sample = low & (angle >= lower) & (angle < lower + 5.0) & (arrays["valve_speed_rad_s"] > 0)
        if not np.any(sample):
            verdict.fail(
                "low_load",
                f"angle_bin_{lower:g}_{lower + 5:g}_deg",
                0,
                ">= 1 sample",
                "low-load angle coverage is missing",
                float(arrays["time_s"][np.flatnonzero(low)[-1]]),
            )
            continue
        ungripped = sample & (arrays["pad_normal_force_n"].min(axis=1) < PAD_CONTACT_N)
        if ungripped.any():
            index = int(np.flatnonzero(ungripped)[0])
            verdict.fail(
                "low_load",
                "two_loaded_pads_n",
                float(arrays["pad_normal_force_n"][index].min()),
                PAD_CONTACT_N,
                "both pads must carry the rotating handle",
                float(arrays["time_s"][index]),
            )
        for field in ("hinge_friction_torque_nm", "pad_closing_torque_nm"):
            minimum = float(arrays[field][sample].min())
            if minimum < LOW_LOAD_TORQUE_NM:
                index = int(np.flatnonzero(sample & (arrays[field] < LOW_LOAD_TORQUE_NM))[0])
                verdict.fail(
                    "low_load",
                    field,
                    float(arrays[field][index]),
                    f">= {LOW_LOAD_TORQUE_NM}",
                    "measured low-load torque is insufficient",
                    float(arrays["time_s"][index]),
                )
    if contact_window is not None:
        reference = float(arrays["handle_axial_position_m"][contact_window[1]])
        slip_values = np.abs(arrays["handle_axial_position_m"] - reference)
        slip = float(slip_values[low].max())
        verdict.measurements["low_load_max_axial_slip_m"] = slip
        if slip > LOW_LOAD_SLIP_M:
            index = int(np.flatnonzero(low & (slip_values > LOW_LOAD_SLIP_M))[0])
            verdict.fail(
                "low_load",
                "handle_axial_slip_m",
                float(slip_values[index]),
                f"<= {LOW_LOAD_SLIP_M}",
                "handle slipped after two-pad contact was established",
                float(arrays["time_s"][index]),
            )
    before_end = np.arange(len(low)) <= np.flatnonzero(low)[-1]
    nonpad = before_end & (arrays["handle_nonpad_force_n"] > ZERO_CONTACT_N)
    if nonpad.any():
        index = int(np.flatnonzero(nonpad)[0])
        verdict.fail(
            str(arrays["stage"][index]),
            "handle_nonpad_force_n",
            float(arrays["handle_nonpad_force_n"][index]),
            ZERO_CONTACT_N,
            "a non-pad body contacted the low-load handle",
            float(arrays["time_s"][index]),
        )
    return verdict.result()


def _check_m1(arrays, manipulation, first_operate, verdict):
    base_shift = np.linalg.norm(arrays["base_xyz_m"] - arrays["base_xyz_m"][first_operate], axis=1)
    foot_shift = np.linalg.norm(arrays["feet_xyz_m"] - arrays["feet_xyz_m"][first_operate], axis=2)
    for field, values, limit, minimum in (
        ("max_base_shift_m", base_shift, BASE_SHIFT_M, False),
        ("max_foot_shift_m", foot_shift.max(axis=1), FOOT_SHIFT_M, False),
        ("max_friction_use", arrays["foot_friction_use"].max(axis=1), FRICTION_USE_LIMIT, False),
        ("min_foot_normal_force_n", arrays["foot_normal_force_n"].min(axis=1), FOOT_LOAD_N, True),
    ):
        window = values[manipulation]
        verdict.measurements[field] = float(window.min() if minimum else window.max())
        bad = manipulation & ((values < limit) if minimum else (values > limit))
        if bad.any():
            index = int(np.flatnonzero(bad)[0])
            verdict.fail(
                str(arrays["stage"][index]),
                field,
                float(values[index]),
                f"{'>=' if minimum else '<='} {limit}",
                "M1 stability limit exceeded",
                float(arrays["time_s"][index]),
            )


def evaluate_valve(trace: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Verify 10 N m operation through release, without requiring the N1 return."""
    verdict = _Verdict()
    arrays = _load_trace(trace, COMMON_FIELDS + M1_FIELDS, verdict)
    if arrays is None:
        return verdict.result()
    _check_metadata(metadata, arrays["time_s"], verdict)
    _require_stage_order(
        arrays, metadata, ("approach", "contact", "operate", "hold", "release"), verdict
    )
    _check_contact(arrays, metadata, verdict)
    time, angle = arrays["time_s"], arrays["valve_angle_deg"]
    opening = metadata.get("valve_operation") == "open"
    lower = _stage_mask(arrays, metadata, "lower")
    support = arrays["foot_normal_force_n"].min(axis=1)
    lost_support = lower & (support < FOOT_LOAD_N)
    if lost_support.any():
        index = int(np.flatnonzero(lost_support)[0])
        verdict.fail(
            "lower",
            "min_foot_normal_force_n",
            float(support[index]),
            f">= {FOOT_LOAD_N}",
            "the lowering stance lost its expected tread load",
            float(time[index]),
        )
    operate = _stage_mask(arrays, metadata, "operate")
    settle = _stage_mask(arrays, metadata, "settle")
    hold = _stage_mask(arrays, metadata, "hold")
    if not np.any(operate) or not np.any(hold):
        return verdict.result()
    first_operate = int(np.flatnonzero(operate)[0])
    if abs(angle[first_operate]) > ANGLE_TOLERANCE_DEG:
        verdict.fail(
            "operate",
            "start_angle_deg",
            float(angle[first_operate]),
            f"abs <= {ANGLE_TOLERANCE_DEG}",
            "valve did not start closed" if opening else "valve did not start open",
            float(time[first_operate]),
        )
    maximum = float(angle[operate | settle | hold].max())
    verdict.measurements["max_angle_deg"] = maximum
    if maximum < VALVE_TARGET_DEG:
        verdict.fail(
            "operate",
            "max_angle_deg",
            maximum,
            f">= {VALVE_TARGET_DEG}",
            "valve did not reach the open stop"
            if opening
            else "valve did not reach the closed stop",
            float(time[np.flatnonzero(hold)[-1]]),
        )
    hold_span, _ = _longest_window(hold, time)
    verdict.measurements["hold_s"] = hold_span
    if hold_span < HOLD_S - 1e-9:
        verdict.fail(
            "hold",
            "hold_s",
            hold_span,
            f">= {HOLD_S}",
            "open hold is too short" if opening else "closed hold is too short",
            float(time[np.flatnonzero(hold)[-1]]),
        )
    hold_bad = hold & (np.abs(angle - VALVE_TARGET_DEG) > ANGLE_TOLERANCE_DEG)
    if np.any(hold_bad):
        index = int(np.flatnonzero(hold_bad)[0])
        verdict.fail(
            "hold",
            "angle_deg",
            float(angle[index]),
            f"{VALVE_TARGET_DEG} +/- {ANGLE_TOLERANCE_DEG}",
            "hold angle left the allowed band",
            float(time[index]),
        )
    moving = (
        operate
        & (angle >= 5)
        & (angle < 85)
        & (arrays["valve_speed_rad_s"] > MOTION_MIN_SPEED_RADPS)
    )
    resistance = metadata.get("valve_resistance_nm")
    if not isinstance(resistance, (int, float)) or not np.isfinite(resistance):
        return verdict.result()
    resistance_limit = float(resistance) - 0.02
    verdict.measurements["resistance_limit_nm"] = resistance_limit
    covered = 0
    pad_torque_by_bin = []
    for lower in np.arange(5.0, 85.0, 5.0):
        samples = moving & (angle >= lower) & (angle < lower + 5)
        if not np.any(samples):
            verdict.fail(
                "operate",
                f"loaded_bin_{lower:g}_{lower + 5:g}_deg",
                0,
                ">= 1 sample",
                "loaded rotation bin is missing",
                float(time[np.flatnonzero(operate)[-1]]),
            )
            continue
        covered += 1
        minimum = float(arrays["hinge_friction_torque_nm"][samples].min())
        if minimum < resistance_limit:
            index = int(
                np.flatnonzero(samples & (arrays["hinge_friction_torque_nm"] < resistance_limit))[0]
            )
            verdict.fail(
                "operate",
                "hinge_friction_torque_nm",
                float(arrays["hinge_friction_torque_nm"][index]),
                f">= {resistance_limit}",
                "measured valve resistance is insufficient",
                float(time[index]),
            )
        pad_torque = arrays["pad_closing_torque_nm"][samples]
        pad_torque_by_bin.append(
            {
                "angle_deg": [float(lower), float(lower + 5)],
                "min_nm": float(pad_torque.min()),
                "max_nm": float(pad_torque.max()),
            }
        )
        if pad_torque.max() <= 0:
            index = int(np.flatnonzero(samples)[0])
            verdict.fail(
                "operate",
                "pad_closing_torque_nm",
                float(pad_torque.max()),
                "> 0 in each loaded angle bin",
                "pads did not provide torque in the opening direction"
                if opening
                else "pads did not provide a measured closing-direction torque",
                float(time[index]),
            )
    verdict.measurements["loaded_angle_bins"] = covered
    verdict.measurements["pad_closing_torque_by_bin_nm"] = pad_torque_by_bin
    manipulation = operate | settle | hold
    lost = manipulation & np.any(arrays["pad_normal_force_n"] < PAD_CONTACT_N, axis=1)
    loss_s, _ = _longest_window(lost, time)
    verdict.measurements["max_pad_loss_s"] = loss_s
    if loss_s > PAD_LOSS_LIMIT_S + 1e-12:
        step = float(np.median(np.diff(time)))
        first, index = _first_window(lost, time, PAD_LOSS_LIMIT_S + step * 1e-4)
        verdict.fail(
            str(arrays["stage"][index]),
            "pad_contact_loss_s",
            float(time[index] - time[first] + step),
            f"<= {PAD_LOSS_LIMIT_S}",
            "a pad lost contact too long",
            float(time[index]),
        )
    _check_m1(arrays, manipulation, first_operate, verdict)
    unexpected = float(arrays["unexpected_contact_force_n"].max())
    if unexpected > UNEXPECTED_CONTACT_N:
        index = int(np.flatnonzero(arrays["unexpected_contact_force_n"] > UNEXPECTED_CONTACT_N)[0])
        verdict.fail(
            str(arrays["stage"][index]),
            "unexpected_contact_force_n",
            float(arrays["unexpected_contact_force_n"][index]),
            f"<= {UNEXPECTED_CONTACT_N}",
            "unexpected contact exceeded its limit",
            float(time[index]),
        )
    nonpad = float(arrays["handle_nonpad_force_n"].max())
    if nonpad > ZERO_CONTACT_N:
        index = int(np.flatnonzero(arrays["handle_nonpad_force_n"] > ZERO_CONTACT_N)[0])
        verdict.fail(
            str(arrays["stage"][index]),
            "handle_nonpad_force_n",
            float(arrays["handle_nonpad_force_n"][index]),
            f"<= {ZERO_CONTACT_N}",
            "a non-pad body pushed the handle",
            float(time[index]),
        )
    release = _stage_mask(arrays, metadata, "release") | _stage_mask(arrays, metadata, "retract")
    released = (
        release
        & np.all(arrays["pad_surface_gap_m"] >= RELEASE_GAP_M, axis=1)
        & np.all(arrays["pad_normal_force_n"] <= ZERO_CONTACT_N, axis=1)
    )
    release_window = _first_window(released, time, RELEASE_HOLD_S)
    release_span, release_start = _longest_window(released, time)
    verdict.measurements["release_hold_s"] = release_span
    if release_window is None:
        verdict.fail(
            "release",
            "release_hold_s",
            release_span,
            f">= {RELEASE_HOLD_S}",
            "pads did not establish a measured release",
            None if release_start is None else float(time[release_start]),
        )
        return verdict.result()
    post_start = release_window[1] + 1
    step = float(np.median(np.diff(time)))
    post_span = float(time[-1] - time[post_start] + step) if post_start < len(time) else 0.0
    verdict.measurements["post_release_observation_s"] = post_span
    if post_span < POST_RELEASE_OBSERVATION_S:
        verdict.fail(
            "release",
            "post_release_observation_s",
            post_span,
            f">= {POST_RELEASE_OBSERVATION_S}",
            "post-release observation is too short",
            float(time[release_window[1]]),
        )
    separated = np.arange(len(time)) >= release_window[0]
    angle_bad = (np.arange(len(time)) >= np.flatnonzero(release)[0]) & (
        np.abs(angle - VALVE_TARGET_DEG) > ANGLE_TOLERANCE_DEG
    )
    if np.any(angle_bad):
        index = int(np.flatnonzero(angle_bad)[0])
        verdict.fail(
            "release",
            "post_release_angle_deg",
            float(angle[index]),
            f"{VALVE_TARGET_DEG} +/- {ANGLE_TOLERANCE_DEG}",
            "valve moved after release",
            float(time[index]),
        )
    recontact = separated & (
        np.any(arrays["pad_normal_force_n"] > ZERO_CONTACT_N, axis=1)
        | (arrays["handle_nonpad_force_n"] > ZERO_CONTACT_N)
    )
    if np.any(recontact):
        index = int(np.flatnonzero(recontact)[0])
        verdict.fail(
            "release",
            "post_release_contact",
            True,
            False,
            "robot contacted the handle after release",
            float(time[index]),
        )
    return verdict.result()


def evaluate_mission(
    trace: Mapping[str, Any], metadata: Mapping[str, Any], n1_result: Mapping[str, Any]
) -> dict[str, Any]:
    """Add post-release return and saved N1 checks to the standalone valve verdict."""
    result = evaluate_valve(trace, metadata)
    verdict = _Verdict()
    verdict.failures.extend(result["failures"])
    verdict.measurements.update(result["measurements"])
    verdict.failures.extend(n1_result.get("walking_failures", ()))
    if "time_s" in trace and "stage" in trace:
        time = np.asarray(trace["time_s"])
        stage = np.asarray(trace["stage"]).astype(str)
        _require_stage_order(
            {"stage": stage, "time_s": time},
            metadata,
            (
                "prewalk",
                "lower",
                "approach",
                "contact",
                "operate",
                "settle",
                "hold",
                "release",
                "retract",
                "return",
            ),
            verdict,
        )
        aliases = metadata.get("stage_map", {}).get("return", ["return"])
        aliases = [aliases] if isinstance(aliases, str) else aliases
        return_mask = np.isin(stage, aliases)
        if not np.any(return_mask):
            verdict.fail(
                "return", "stage_present", False, True, "whole-mission trace has no return stage"
            )
        elif "valve_angle_deg" in trace:
            angle = np.asarray(trace["valve_angle_deg"])
            bad = return_mask & (np.abs(angle - VALVE_TARGET_DEG) > ANGLE_TOLERANCE_DEG)
            if np.any(bad):
                index = int(np.flatnonzero(bad)[0])
                verdict.fail(
                    "return",
                    "valve_angle_deg",
                    float(angle[index]),
                    f"{VALVE_TARGET_DEG} +/- {ANGLE_TOLERANCE_DEG}",
                    "valve did not remain closed through return",
                    float(time[index]),
                )
    if not n1_result.get("physical_round_trip_verified", False):
        verdict.fail(
            "return", "physical_round_trip_verified", False, True, "N1 round trip did not pass"
        )
    milestones = n1_result.get("milestones")
    valid_sequence = isinstance(milestones, Sequence) and not isinstance(milestones, (str, bytes))
    count = len(milestones) if valid_sequence else None
    if count != 156:
        verdict.fail("return", "landing_count", count, 156, "saved N1 landing count is incomplete")
    controller = n1_result.get("controller_metrics")
    if not isinstance(controller, Mapping):
        verdict.fail(
            "return", "controller_metrics", None, "mapping", "N1 controller metrics are missing"
        )
    else:
        for name in ("saturated_steps", "qp_failed_ticks", "qp_relaxed_ticks"):
            if controller.get(name) != 0:
                verdict.fail("return", name, controller.get(name), 0, "N1 controller metric failed")
        if controller.get("fell") is not False:
            verdict.fail(
                "return", "fell", controller.get("fell"), False, "N1 controller recorded a fall"
            )
    return verdict.result()


def evaluate_validation(
    low_load_trace: Mapping[str, Any],
    mission_trace: Mapping[str, Any],
    n1_result: Mapping[str, Any],
    *,
    low_load_metadata: Mapping[str, Any],
    mission_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate component evidence while preserving each component's failures."""
    low_load = evaluate_low_load(low_load_trace, low_load_metadata)
    mission = evaluate_mission(mission_trace, mission_metadata, n1_result)
    first = [
        failure for failure in (low_load["first_failure"], mission["first_failure"]) if failure
    ]
    return {
        "verified": low_load["verified"] and mission["verified"],
        "first_failure": first[0] if first else None,
        "low_load": low_load,
        "mission": mission,
    }
