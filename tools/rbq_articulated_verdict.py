"""Score an articulated RBQ stair-valve trajectory from recorded MuJoCo states.

The scorer advances no physics.  It restores every post-step state, calls
``mj_forward`` with the recorded motor command, and measures contacts from that
state.  The reference file supplies the frozen clock, stages and nominal path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from render_stair_mission import build_model
from replay_stair_mission import TRACK_FIELDS, first_component_failure, make_mission
from replay_stair_sequence import DWELL_S, MAX_FOOT_XY_ERROR_M
from replay_stair_transfer import LOAD_THRESHOLD_N, SURFACES, expected_surfaces, support_loads
from stair_mission_reference import scheduled_landings
from stair_mission_route import route
from stair_mission_trace import PAD_PAIRS, SLIDES, measured_walk
from stair_open_approach import approach_with_rear_foot_offset, rear_landing_stance
from stair_open_station import evaluate_station

from go2_wall import mission_verdict, stair_valve, transition
from go2_wall.open_valve_grasp import opening_candidate
from go2_wall.robot import LEGS

REQUIRED_STATE_FIELDS = (
    "time_s",
    "qpos",
    "qvel",
    "ctrl",
    "requested_robot_nm",
    "requested_finger_force_n",
)
REFERENCE_FIELDS = ("time_s", "stage", "qpos", "stance")


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        missing = set(required) - set(source.files)
        if missing:
            raise ValueError(f"{path} is missing {', '.join(sorted(missing))}")
        return {name: source[name] for name in source.files}


def _validate_inputs(
    model: mujoco.MjModel,
    state: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> int:
    n = len(state["time_s"])
    shapes = {
        "time_s": (n,),
        "qpos": (n, model.nq),
        "qvel": (n, model.nv),
        "ctrl": (n, model.nu),
        "requested_robot_nm": (n, 18),
        "requested_finger_force_n": (n,),
    }
    for name, shape in shapes.items():
        values = np.asarray(state[name])
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"invalid state array {name}: {values.shape}, expected {shape}")
    if n == 0 or n > len(reference["time_s"]):
        raise ValueError("state clock must be a non-empty prefix of the frozen reference")
    for name in REFERENCE_FIELDS:
        if len(reference[name]) < n:
            raise ValueError(f"reference {name} is shorter than the state trajectory")
    dt = float(model.opt.timestep)
    expected = np.arange(1, n + 1) * dt
    if not np.allclose(state["time_s"], expected, atol=1e-7, rtol=0):
        raise ValueError("state time_s must be the uninterrupted post-step physics clock")
    if not np.allclose(reference["time_s"][:n], expected, atol=1e-7, rtol=0):
        raise ValueError("reference time_s differs from the physical clock")
    return n


def planned_stance(reference: dict[str, np.ndarray], walk_path: Path, count: int) -> np.ndarray:
    """Map the official walking stance flags to each 1 ms post-step sample."""
    if "stance" in reference:
        values = np.asarray(reference["stance"][:count], dtype=bool)
        if values.shape != (count, 4):
            raise ValueError("reference stance must have shape (physics_steps, 4)")
        return values
    with np.load(walk_path, allow_pickle=False) as walk:
        times = np.asarray(walk["time"], dtype=float)
        stance = np.asarray(walk["stance"], dtype=bool)
    clock = np.asarray(reference["time_s"][:count])
    indices = np.searchsorted(times, np.minimum(clock, times[-1]), side="right") - 1
    indices = np.maximum(indices, 0)
    result = stance[indices]
    result[np.asarray(reference["stage"][:count]).astype(str) != "prewalk"] = True
    return result


def write_reference_cache(
    scene_path: Path, nominal_path: Path, walk_path: Path, output_path: Path
) -> None:
    """Build the exact per-physics-step planned pose and stance used by the frozen replay."""
    manifest = json.loads(scene_path.read_text(encoding="utf-8"))
    settings = manifest["settings"]
    geometry, sequence, study, segments, ends, anchor = route(
        settings["slope_deg"], settings["rise_m"]
    )
    with np.load(walk_path, allow_pickle=False) as saved_walk:
        walk = {name: saved_walk[name] for name in ("time", *TRACK_FIELDS)}
        ends, _ = scheduled_landings(
            saved_walk,
            ends,
            anchor,
            float(saved_walk["slope_deg"]),
            float(saved_walk["rise_m"]),
        )
    from go2_wall.stair_open_balance import OpeningStanceControl

    walk, ends, _ = approach_with_rear_foot_offset(
        geometry,
        segments,
        ends,
        anchor,
        walk,
        study["candidates"][4]["feet_xyz_m"],
        OpeningStanceControl.REAR_OFFSET_M,
    )
    boundaries = manifest["motion"]["boundaries_s"]
    args = SimpleNamespace(
        extent="open-at-station",
        gain=float(manifest["gain"]),
        motion_seconds=float(boundaries["operate"] - boundaries["operate_motion_start"]),
        extra_settle_seconds=float(manifest["extra_settle_s"]),
        jaw_load_bias_m=float(manifest["jaw_load_bias_m"]),
        terminal_overtravel_deg=float(
            manifest["motion"]["command_target_deg"] - manifest["motion"]["hold_target_deg"]
        ),
        opening_stance_feedback=True,
    )
    track, _, _, _, _, _, _, _ = make_mission(
        geometry, sequence, study, segments, ends, anchor, walk, args, settings
    )
    nominal = _load_npz(nominal_path, ("time_s", "stage", "qpos"))
    count = len(nominal["time_s"])
    command_time = np.arange(count) * float(manifest["timestep_s"])
    left = np.searchsorted(track["time"], command_time, side="right") - 1
    left = np.clip(left, 0, len(track["time"]) - 2)
    right = left + 1
    weight = (command_time - track["time"][left]) / (
        track["time"][right] - track["time"][left]
    )
    planned = np.asarray(nominal["qpos"], dtype=float).copy()
    planned[:, :7] = (
        (1 - weight[:, None]) * track["qpos"][left, :7]
        + weight[:, None] * track["qpos"][right, :7]
    )
    planned[:, 3:7] /= np.linalg.norm(planned[:, 3:7], axis=1)[:, None]
    stance = np.asarray(track["stance"][left] & track["stance"][right], dtype=bool)
    np.savez(
        output_path,
        time_s=nominal["time_s"],
        stage=nominal["stage"],
        qpos=planned,
        stance=stance,
        scene_sha256=file_sha256(scene_path),
        nominal_sha256=file_sha256(nominal_path),
        walk_sha256=file_sha256(walk_path),
    )


def _material_slip(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_ids: list[int],
    stance: np.ndarray,
    accumulated: np.ndarray,
    maximum: np.ndarray,
    jacobian: np.ndarray,
) -> None:
    for index, geom in enumerate(foot_ids):
        if not stance[index]:
            accumulated[index] = 0.0
            continue
        velocity = transition.contact_velocity(
            model, data, geom, transition.FLOOR_NORMAL, jacobian
        )
        accumulated[index] += velocity * model.opt.timestep
        maximum[index] = max(maximum[index], np.linalg.norm(accumulated[index]))


def reconstruct_trace(
    model: mujoco.MjModel,
    state: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    stance: np.ndarray,
    floor_mu: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Measure contacts from saved states without stepping the model."""
    count = len(state["time_s"])
    data = mujoco.MjData(model)
    feet_ids = [model.geom(name).id for name in LEGS]
    world = set(transition.world_geoms(model))
    task_pairs = {
        (model.geom(first).id, model.geom(second).id)
        for first, second in PAD_PAIRS
        for first, second in ((first, second), (second, first))
    }
    slide_positions = [model.joint(name).qposadr[0] for name in SLIDES]
    trace: dict[str, np.ndarray] = {
        "time_s": np.asarray(state["time_s"], dtype=float),
        "stage": np.asarray(reference["stage"][:count]).astype(str),
        "base_xyz_m": np.empty((count, 3)),
        "feet_xyz_m": np.empty((count, 4, 3)),
        "foot_normal_force_n": np.empty((count, 4)),
        "foot_friction_use": np.empty((count, 4)),
        "unexpected_contact_force_n": np.empty(count),
        "base_track_m": np.linalg.norm(
            state["qpos"][:, :3] - reference["qpos"][:count, :3], axis=1
        ),
        "stance_slip_m": np.empty(count),
        "robot_torque_requested_nm": np.asarray(state["requested_robot_nm"], dtype=float),
        "fell": np.empty(count, dtype=bool),
        "finger_slide_m": np.asarray(state["qpos"][:, slide_positions], dtype=float),
        "gripper_open": np.asarray(state["qpos"][:, slide_positions].sum(axis=1) >= 0.03),
    }
    for field, width in (
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
        ("handle_nonpad_force_n", 1),
    ):
        trace[field] = np.zeros(count) if width == 1 else np.zeros((count, width))
    loads = np.empty((count, 4, len(SURFACES)))
    foot_force = np.empty((count, 4, 3))
    accumulated_slip = np.zeros((4, 3))
    maximum_slip = np.zeros(4)
    jacobian = np.zeros((3, model.nv))
    worst_contacts: dict[str, float] = {}
    tilt_error = np.empty(3)
    limits = np.asarray(model.actuator_ctrlrange[:18, 1], dtype=float)
    for index in range(count):
        data.qpos[:] = state["qpos"][index]
        data.qvel[:] = state["qvel"][index]
        data.ctrl[:] = state["ctrl"][index]
        data.time = state["time_s"][index]
        data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(model, data)
        loads[index] = support_loads(model, data)
        contacts: dict[str, float] = {}
        totals = transition._foot_forces(
            model, data, feet_ids, world, set(feet_ids), contacts, task_pairs
        )
        foot_force[index] = totals
        trace["unexpected_contact_force_n"][index] = max(contacts.values(), default=0.0)
        for name, force in contacts.items():
            worst_contacts[name] = max(worst_contacts.get(name, 0.0), force)
        trace["base_xyz_m"][index] = data.qpos[:3]
        trace["feet_xyz_m"][index] = [data.geom_xpos[geom] for geom in feet_ids]
        trace["foot_normal_force_n"][index] = loads[index, np.arange(4), [5, 5, 3, 3]]
        normals = np.maximum(foot_force[index, :, 2], 1.0)
        trace["foot_friction_use"][index] = (
            np.linalg.norm(foot_force[index, :, :2], axis=1) / normals / floor_mu
        )
        _material_slip(
            model,
            data,
            feet_ids,
            stance[index],
            accumulated_slip,
            maximum_slip,
            jacobian,
        )
        trace["stance_slip_m"][index] = maximum_slip.max()
        mujoco.mju_subQuat(tilt_error, data.qpos[3:7], reference["qpos"][index, 3:7])
        trace["fell"][index] = bool(
            np.degrees(np.linalg.norm(tilt_error)) > transition.FALL_TILT_DEG
            or data.qpos[2] < 0.12
        )
        if trace["stage"][index] != "prewalk":
            sample = stair_valve.observe_grasp(model, data)
            for name in (
                "valve_angle_deg",
                "valve_speed_rad_s",
                "hinge_friction_torque_nm",
                "signed_hinge_friction_torque_nm",
                "pad_closing_torque_nm",
                "pad_closing_tangent_force_n",
                "pad_normal_force_n",
                "pad_surface_gap_m",
                "handle_axis_offset_m",
                "handle_axial_position_m",
            ):
                trace[name][index] = sample[name]
            trace["handle_nonpad_force_n"][index] = sample["max_unexpected_lever_contact_n"]
        else:
            hinge = model.joint(stair_valve.VALVE_HINGE)
            trace["valve_angle_deg"][index] = np.degrees(data.qpos[hinge.qposadr[0]])
            trace["valve_speed_rad_s"][index] = data.qvel[hinge.dofadr[0]]
    maximum_ratio = float(np.max(np.abs(state["requested_robot_nm"]) / limits))
    metrics = {
        "maximum_requested_torque_ratio": maximum_ratio,
        "maximum_material_slip_m": dict(zip(LEGS, maximum_slip.tolist(), strict=True)),
        "unexpected_contacts_n": worst_contacts,
        "fell": bool(trace["fell"].any()),
        "controller": {"qp_applicable": False, "qp_metrics": None},
    }
    return {**trace, "tread_load_n": loads, "foot_force_world_n": foot_force}, metrics


def _walking_verdict(
    geometry: dict,
    model: mujoco.MjModel,
    ends: list,
    trace: dict[str, np.ndarray],
    metrics: dict[str, object],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    time = trace["time_s"]
    loads = trace["tread_load_n"]

    def fail(metric: str, index: int, measured: object, limit: str, **extra: object) -> None:
        failures.append(
            {
                "stage": str(trace["stage"][index]),
                "time_s": float(time[index]),
                "metric": metric,
                "measured": measured,
                "limit": limit,
                **extra,
            }
        )

    if np.any(trace["fell"]):
        index = int(np.flatnonzero(trace["fell"])[0])
        fail("fell", index, True, "== false")
    for name, limit in (
        ("base_track_m", transition.BASE_TRACK_LIMIT_M),
        ("stance_slip_m", transition.STANCE_SLIP_LIMIT_M),
        ("unexpected_contact_force_n", mission_verdict.UNEXPECTED_CONTACT_N),
    ):
        bad = np.asarray(trace[name]) > limit
        if np.any(bad):
            index = int(np.flatnonzero(bad)[0])
            fail(name, index, float(trace[name][index]), f"<= {limit}")
    ratio = float(metrics["maximum_requested_torque_ratio"])
    if ratio > 1.0:
        index = int(
            np.argmax(np.max(np.abs(trace["robot_torque_requested_nm"]), axis=1))
        )
        fail("requested_torque_ratio", index, ratio, "<= 1")
    positions = np.concatenate((trace["feet_xyz_m"], trace["base_xyz_m"][:, None]), axis=1)
    stair = geometry["stair_geometry"]
    platform = model.geom("platform")
    start_x = platform.pos[0] - platform.size[0]
    on_stairs = (positions[..., 0] >= 0) & (positions[..., 0] < start_x)
    outside_width = on_stairs & (np.abs(positions[..., 1]) > stair["clear_width_m"] / 2)
    if np.any(outside_width):
        index = int(np.argwhere(outside_width)[0, 0])
        fail(
            "tread_lateral_position_m",
            index,
            float(np.abs(positions[index, :, 1]).max()),
            f"<= {stair['clear_width_m'] / 2}",
        )
    evaluated = 0
    for landing, (end, stance, leg) in enumerate(ends, 1):
        if end > time[-1] + 1e-7:
            break
        command_time = time - model.opt.timestep
        frames = np.flatnonzero((command_time >= end - DWELL_S) & (command_time < end))
        if not len(frames):
            continue
        evaluated += 1
        surfaces = expected_surfaces(stance, geometry["rise_m"], geometry["foot_radius_m"])
        supporting = loads[frames][:, np.arange(4), surfaces].min(axis=1)
        missed = np.flatnonzero(supporting < LOAD_THRESHOLD_N)
        if len(missed):
            index = int(frames[missed[0]])
            fail("landing_support_n", index, float(supporting[missed[0]]), ">= 1", landing=landing)
        last = int(frames[-1])
        error = float(
            np.linalg.norm(
                trace["feet_xyz_m"][last, LEGS.index(leg), :2] - stance[leg][0][:2]
            )
        )
        if error > MAX_FOOT_XY_ERROR_M:
            fail("landing_xy_m", last, error, f"<= {MAX_FOOT_XY_ERROR_M}", landing=landing)
    if evaluated != len(ends):
        fail("landing_count", len(time) - 1, evaluated, f"== {len(ends)}")
    failures.sort(key=lambda failure: failure["time_s"])
    return {
        "landings_evaluated": evaluated,
        "scheduled_landings": len(ends),
        "verified": not failures,
        "first_failure": failures[0] if failures else None,
        "failures": failures,
    }


def _installed_task(settings: dict) -> tuple[dict, dict, list, int]:
    geometry, _, study, segments, ends, anchor = route(settings["slope_deg"], settings["rise_m"])
    installed = study["candidates"][4]
    saved = opening_candidate(installed)
    from go2_wall.stair_open_balance import OpeningStanceControl

    saved["base_pose"] = OpeningStanceControl.work_pose(saved["base_pose"])
    ends = [
        (
            time,
            rear_landing_stance(
                stance, installed["feet_xyz_m"], OpeningStanceControl.REAR_OFFSET_M
            ),
            leg,
        )
        for time, stance, leg in ends
    ]
    return geometry, installed, saved, ends[: anchor + 1]


def score_state_file(
    scene_path: Path,
    state_path: Path,
    reference_path: Path,
    walk_path: Path,
    *,
    trace_out: Path | None = None,
) -> dict[str, object]:
    manifest = json.loads(scene_path.read_text(encoding="utf-8"))
    scene_digest = file_sha256(scene_path)
    model, _ = build_model(manifest)
    state = _load_npz(state_path, REQUIRED_STATE_FIELDS)
    reference = _load_npz(reference_path, REFERENCE_FIELDS)
    for source, label in ((state, "state"), (reference, "reference")):
        if "scene_sha256" in source and str(source["scene_sha256"]) != scene_digest:
            raise ValueError(f"{label} scene_sha256 differs from the selected scene")
    count = _validate_inputs(model, state, reference)
    stance = planned_stance(reference, walk_path, count)
    trace, metrics = reconstruct_trace(
        model, state, reference, stance, float(manifest["settings"]["floor_mu"])
    )
    geometry, installed, saved, ends = _installed_task(manifest["settings"])
    # The established walk judge requires this field as part of its historical QP
    # trace schema. This controller has no QP. Zeros are supplied only to reuse its
    # unrelated physical checks; the public controller record remains explicitly N/A.
    trace["qp_attempt"] = np.zeros(count, dtype=int)
    log = {
        "time": state["time_s"] - model.opt.timestep,
        "base": state["qpos"][:, :7],
        "feet": trace["feet_xyz_m"],
        "qpos": state["qpos"],
        "qvel": state["qvel"],
    }
    walking = measured_walk(
        geometry,
        model,
        ends,
        log,
        trace["tread_load_n"],
        trace,
        model.actuator_ctrlrange[:18, 1],
        full=False,
    )
    walking["scheduled_landings"] = len(ends)
    if walking["landings_evaluated"] != len(ends):
        failure = {
            "stage": str(trace["stage"][-1]),
            "time_s": float(trace["time_s"][-1]),
            "metric": "landing_count",
            "measured": walking["landings_evaluated"],
            "limit": f"== {len(ends)}",
        }
        walking["failures"].append(failure)
        walking["failures"].sort(key=lambda item: item["time_s"])
        walking["verified"] = False
        walking["first_failure"] = walking["failures"][0]
    finger_joints = {model.joint(name).id for name in SLIDES}
    equality_objects = zip(model.eq_type, model.eq_obj1id, model.eq_obj2id, strict=True)
    finger_equalities = sum(
        eq_type == mujoco.mjtEq.mjEQ_JOINT and {first, second} == finger_joints
        for eq_type, first, second in equality_objects
    )
    hinge = model.joint(stair_valve.VALVE_HINGE)
    hinge_actuators = int(
        np.count_nonzero(
            (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
            & (model.actuator_trnid[:, 0] == hinge.id)
        )
    )
    measured_resistance = float(model.dof_frictionloss[hinge.dofadr[0]])
    if not np.isclose(measured_resistance, manifest["settings"]["resistance_nm"]):
        raise ValueError("compiled valve resistance differs from scene.json")
    metadata = {
        "physics_timestep_s": float(model.opt.timestep),
        "record_every_step": True,
        "state_reset_count": 0,
        "qpos_write_count_after_start": 0,
        "qvel_write_count_after_start": 0,
        "act_write_count_after_start": 0,
        "mocap_write_count_after_start": 0,
        "max_qfrc_applied_n": 0.0,
        "max_xfrc_applied_n": 0.0,
        "finger_equality_count": int(finger_equalities),
        "other_equality_count": model.neq - int(finger_equalities),
        "hinge_actuator_count": hinge_actuators,
        "valve_resistance_nm": measured_resistance,
        "robot_saturated_steps": int(
            np.count_nonzero(
                np.any(
                    np.abs(state["requested_robot_nm"])
                    > model.actuator_ctrlrange[:18, 1],
                    axis=1,
                )
            )
        ),
        "finger_saturated_steps": int(
            np.count_nonzero(
                np.abs(state["requested_finger_force_n"])
                > model.actuator_ctrlrange[18, 1]
            )
        ),
        # Compatibility inputs for the established valve-only judge. The public result
        # reports that this controller has no QP, rather than exposing these as metrics.
        "qp_failed_ticks": 0,
        "qp_relaxed_ticks": 0,
        "fell": metrics["fell"],
        "valve_operation": "open",
        "stage_map": {},
    }
    valve = mission_verdict.evaluate_valve(trace, metadata)
    station = evaluate_station(
        model,
        installed,
        saved["base_pose"],
        trace,
        state["qpos"],
        state["qvel"],
        trace["tread_load_n"],
        walking["landings_evaluated"],
        len(ends),
    )
    complete = count == len(reference["time_s"])
    if not complete:
        station = {
            "verified": False,
            "first_failure": {
                "stage": str(trace["stage"][-1]),
                "time_s": float(trace["time_s"][-1]),
                "metric": "complete_horizon",
                "measured": count,
                "limit": f"== {len(reference['time_s'])}",
            },
            "failures": [],
            "measurements": {
                "station_s": float(
                    np.count_nonzero(trace["stage"] == "station") * model.opt.timestep
                )
            },
        }
        station["failures"] = [station["first_failure"]]
    first = first_component_failure((walking, valve, station))
    result = {
        "schema": "rbq_articulated_verdict_v1",
        "outcome": "success" if first is None and complete else "physical_failed",
        "strict_success": bool(first is None and complete),
        "complete_horizon": complete,
        "physics_steps": count,
        "state_sha256": file_sha256(state_path),
        "reference_sha256": file_sha256(reference_path),
        "scene_sha256": scene_digest,
        "walk_sha256": file_sha256(walk_path),
        "controller": metrics.pop("controller"),
        "continuity_audit": {
            "state_samples_are_post_step_prefix": True,
            "state_write_counters": "runner_contract_not_encoded_in_state_file",
            "legacy_valve_judge_qp_compatibility_value": 0,
            "qp_metrics_are_publicly_reported_as": "not_applicable",
        },
        "physical_metrics": metrics,
        "walking": walking,
        "valve": valve,
        "station": station,
        "first_failure": first,
    }
    result["trial_identity"] = {
        name: str(state[name]) for name in ("case_id", "mode") if name in state
    }
    if trace_out is not None:
        np.savez_compressed(trace_out, **trace)
        result["reconstructed_trace_sha256"] = file_sha256(trace_out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-reference")
    prepare.add_argument("scene", type=Path)
    prepare.add_argument("nominal", type=Path)
    prepare.add_argument("walk", type=Path)
    prepare.add_argument("out", type=Path)
    score = subparsers.add_parser("score")
    score.add_argument("scene", type=Path)
    score.add_argument("state", type=Path)
    score.add_argument("reference", type=Path)
    score.add_argument("walk", type=Path)
    score.add_argument("out", type=Path)
    score.add_argument("--trace-out", type=Path)
    args = parser.parse_args()
    if args.command == "prepare-reference":
        if args.out.exists():
            parser.error("output path already exists")
        write_reference_cache(args.scene, args.nominal, args.walk, args.out)
        print(file_sha256(args.out))
        return
    if args.out.exists() or (args.trace_out is not None and args.trace_out.exists()):
        parser.error("output path already exists")
    result = score_state_file(
        args.scene, args.state, args.reference, args.walk, trace_out=args.trace_out
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    brief = {key: result[key] for key in ("outcome", "strict_success", "first_failure")}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
