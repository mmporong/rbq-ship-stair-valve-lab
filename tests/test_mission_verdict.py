import copy

import numpy as np
import pytest

from go2_wall import mission_verdict as mv

DT = 0.005


def metadata(resistance=10.0):
    return {
        "physics_timestep_s": DT,
        "record_every_step": True,
        "state_reset_count": 0,
        "qpos_write_count_after_start": 0,
        "qvel_write_count_after_start": 0,
        "act_write_count_after_start": 0,
        "mocap_write_count_after_start": 0,
        "max_qfrc_applied_n": 0,
        "max_xfrc_applied_n": 0,
        "finger_equality_count": 1,
        "other_equality_count": 0,
        "hinge_actuator_count": 0,
        "valve_resistance_nm": resistance,
        "robot_saturated_steps": 0,
        "finger_saturated_steps": 0,
        "qp_failed_ticks": 0,
        "qp_relaxed_ticks": 0,
        "fell": False,
    }


def trace(stages):
    labels = np.concatenate([np.repeat(name, count) for name, count in stages])
    count = len(labels)
    values = {
        "time_s": np.arange(count) * DT,
        "stage": labels,
        "gripper_open": labels == "approach",
        "pad_normal_force_n": np.full((count, 2), 20.0),
        "valve_angle_deg": np.zeros(count),
        "valve_speed_rad_s": np.zeros(count),
        "hinge_friction_torque_nm": np.full(count, 10.0),
        "pad_closing_torque_nm": np.full(count, 10.0),
        "handle_axial_position_m": np.zeros(count),
        "pad_surface_gap_m": np.zeros((count, 2)),
        "handle_nonpad_force_n": np.zeros(count),
        "base_xyz_m": np.zeros((count, 3)),
        "feet_xyz_m": np.zeros((count, 4, 3)),
        "foot_normal_force_n": np.full((count, 4), 100.0),
        "foot_friction_use": np.full((count, 4), 0.4),
        "unexpected_contact_force_n": np.zeros(count),
    }
    values["pad_normal_force_n"][labels == "approach"] = 0
    return values


def low_load_trace():
    values = trace((("approach", 50), ("contact", 50), ("low_load", 401)))
    mask = values["stage"] == "low_load"
    values["handle_axial_position_m"][:] = 0.021
    values["valve_angle_deg"][mask] = np.linspace(0, 20, mask.sum())
    values["valve_speed_rad_s"][mask] = 0.2
    values["hinge_friction_torque_nm"][:] = 1.0
    values["pad_closing_torque_nm"][:] = 1.0
    return values


def valve_trace(include_return=False):
    stages = [
        ("prewalk", 20),
        ("lower", 100),
        ("approach", 50),
        ("contact", 50),
        ("operate", 1601),
        ("settle", 100),
        ("hold", 1000),
        ("release", 100),
        ("retract", 1100),
    ]
    if include_return:
        stages.append(("return", 200))
    values = trace(stages)
    operate = values["stage"] == "operate"
    settled = np.isin(values["stage"], ["settle", "hold", "release", "retract", "return"])
    released = np.isin(values["stage"], ["release", "retract", "return"])
    values["valve_angle_deg"][operate] = np.linspace(0, 90, operate.sum())
    values["valve_speed_rad_s"][operate] = 0.2
    values["valve_angle_deg"][settled] = 90
    values["pad_normal_force_n"][released] = 0
    values["pad_surface_gap_m"][released] = 0.006
    return values


def n1_result():
    return {
        "physical_round_trip_verified": True,
        "milestones": [{} for _ in range(156)],
        "controller_metrics": {
            "saturated_steps": 0,
            "qp_failed_ticks": 0,
            "qp_relaxed_ticks": 0,
            "fell": False,
        },
    }


def test_low_load_passes_measured_contact_torque_and_slip():
    result = mv.evaluate_low_load(low_load_trace(), metadata(1.0))
    assert result["verified"], result["failures"]
    assert result["measurements"]["low_load_max_angle_deg"] == pytest.approx(20)


def test_low_load_rejects_one_pad_and_unintended_support_at_first_contact():
    values = low_load_trace()
    misleading = np.flatnonzero(values["valve_angle_deg"] >= 7)[0]
    values["pad_normal_force_n"][misleading, 0] = 0.0
    values["handle_nonpad_force_n"][30] = 2.0
    result = mv.evaluate_low_load(values, metadata(1.0))
    assert {f["metric"] for f in result["failures"]} == {
        "two_loaded_pads_n",
        "handle_nonpad_force_n",
    }
    assert result["first_failure"]["time_s"] == values["time_s"][30]
    assert not mv.evaluate_low_load(low_load_trace(), metadata(0.2))["verified"]


@pytest.mark.parametrize(
    ("change", "metric"),
    [
        (lambda data: data.pop("pad_normal_force_n"), "missing_trace_fields"),
        (lambda data: data["valve_angle_deg"].fill(90), "start_angle_deg"),
        (
            lambda data: data["handle_axial_position_m"].__setitem__(
                data["stage"] == "low_load", 0.024
            ),
            "handle_axial_slip_m",
        ),
        (lambda data: data["pad_closing_torque_nm"].fill(0.5), "pad_closing_torque_nm"),
    ],
)
def test_low_load_rejects_fake_success(change, metric):
    values = low_load_trace()
    change(values)
    result = mv.evaluate_low_load(values, metadata(1.0))
    assert not result["verified"]
    assert any(failure["metric"] == metric for failure in result["failures"])


def test_low_load_slip_reference_is_first_secured_contact_not_last_contact_sample():
    values = low_load_trace()
    contact = np.flatnonzero(values["stage"] == "contact")
    values["handle_axial_position_m"][contact[40:]] = 0.024
    values["handle_axial_position_m"][values["stage"] == "low_load"] = 0.024
    result = mv.evaluate_low_load(values, metadata(1.0))
    assert not result["verified"]
    assert any(failure["metric"] == "handle_axial_slip_m" for failure in result["failures"])


def test_valve_passes_settle_hold_release_and_observation():
    result = mv.evaluate_valve(valve_trace(), metadata())
    assert result["verified"], result["failures"]
    assert result["measurements"]["loaded_angle_bins"] == 16
    assert result["measurements"]["hold_s"] == pytest.approx(5)


def test_mission_rejects_unloaded_lowering_and_angle_loss_during_the_release_dwell():
    values = valve_trace(include_return=True)
    lower = np.flatnonzero(values["stage"] == "lower")[20]
    release = np.flatnonzero(values["stage"] == "release")[4]
    values["foot_normal_force_n"][lower, 1] = 0
    values["valve_angle_deg"][release] = 70
    result = mv.evaluate_mission(values, metadata(), n1_result())
    assert result["first_failure"]["time_s"] == values["time_s"][lower]
    assert any(f["time_s"] == values["time_s"][release] for f in result["failures"])


@pytest.mark.parametrize("resistance", [5.0, 10.0, 15.0])
def test_valve_uses_each_run_resistance_for_hinge_and_pad_torque(resistance):
    values, info = valve_trace(), metadata()
    values["hinge_friction_torque_nm"][:] = resistance
    values["pad_closing_torque_nm"][:] = resistance
    info["valve_resistance_nm"] = resistance
    result = mv.evaluate_valve(values, info)
    assert result["verified"], result["failures"]
    assert result["measurements"]["resistance_limit_nm"] == pytest.approx(resistance - 0.02)

    values["pad_closing_torque_nm"][values["stage"] == "operate"] = resistance * 0.3
    result = mv.evaluate_valve(values, info)
    assert result["verified"], result["failures"]

    values["pad_closing_torque_nm"][values["stage"] == "operate"] = -0.01
    result = mv.evaluate_valve(values, info)
    assert not result["verified"]
    assert any(failure["metric"] == "pad_closing_torque_nm" for failure in result["failures"])


def test_valve_reports_first_timed_failure_and_all_failures():
    values = valve_trace()
    operate = np.flatnonzero(values["stage"] == "operate")
    values["pad_normal_force_n"][operate[100:106], 0] = 0
    values["unexpected_contact_force_n"][operate[200]] = 11
    result = mv.evaluate_valve(values, metadata())
    assert not result["verified"]
    assert result["first_failure"]["metric"] == "pad_contact_loss_s"
    assert any(failure["metric"] == "unexpected_contact_force_n" for failure in result["failures"])


@pytest.mark.parametrize(
    ("mutate_trace", "mutate_metadata", "metric"),
    [
        (lambda data: data["handle_nonpad_force_n"].fill(0.1), None, "handle_nonpad_force_n"),
        (lambda data: data["pad_surface_gap_m"].fill(0), None, "release_hold_s"),
        (
            lambda data: data["valve_angle_deg"].__setitem__(data["stage"] == "retract", 89),
            None,
            "post_release_angle_deg",
        ),
        (
            None,
            lambda info: info.__setitem__("qpos_write_count_after_start", 1),
            "qpos_write_count_after_start",
        ),
        (None, lambda info: info.__setitem__("hinge_actuator_count", 1), "hinge_actuator_count"),
        (
            None,
            lambda info: info.__setitem__("finger_saturated_steps", 1),
            "finger_saturated_steps",
        ),
    ],
)
def test_valve_rejects_nonpad_help_missing_release_motion_and_state_injection(
    mutate_trace, mutate_metadata, metric
):
    values, info = valve_trace(), metadata()
    if mutate_trace:
        mutate_trace(values)
    if mutate_metadata:
        mutate_metadata(info)
    result = mv.evaluate_valve(values, info)
    assert not result["verified"]
    assert any(failure["metric"] == metric for failure in result["failures"])


def test_whole_mission_requires_return_and_complete_n1():
    values = valve_trace(include_return=True)
    result = mv.evaluate_mission(values, metadata(), n1_result())
    assert result["verified"], result["failures"]

    broken = copy.deepcopy(n1_result())
    broken["physical_round_trip_verified"] = False
    broken["milestones"].pop()
    result = mv.evaluate_mission(values, metadata(), broken)
    assert not result["verified"]
    assert {failure["metric"] for failure in result["failures"]} >= {
        "physical_round_trip_verified",
        "landing_count",
    }


def test_aggregate_preserves_component_failures():
    low = low_load_trace()
    low.pop("handle_nonpad_force_n")
    result = mv.evaluate_validation(
        low,
        valve_trace(include_return=True),
        n1_result(),
        low_load_metadata=metadata(1.0),
        mission_metadata=metadata(),
    )
    assert not result["verified"]
    assert not result["low_load"]["verified"]
    assert result["mission"]["verified"]
