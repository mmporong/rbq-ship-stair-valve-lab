"""Try a real RR-before-RL landing reorder beside the shut candidate-4 valve.

The nominal N1 route swings RL at 112.12 s and RR at 117.02 s.  This probe keeps both
landings and their exact targets, but replans the two transfer blocks in the opposite order.
Every accepted path is checked by the existing 20 ms IK, collision and static-QP sampler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
from probe_stair_transfer import transfer_segments
from stair_bypass_inputs import provenance
from stair_mission_route import route

from go2_wall import grasp_motion, stair_valve, stairs, transition
from go2_wall.robot import LEGS, ROOT
from go2_wall.statics import joint_names, torque_limits

PUBLIC_EVIDENCE = ROOT / "evidence/stair-mission-20261121/bypass/swing-order"
DEFAULT_WALK = PUBLIC_EVIDENCE / "walk50.npz"
DEFAULT_REPORT = ROOT / ".omx/mission-closeout-20261121/swing-order/result.json"
DEFAULT_FRONTIER = PUBLIC_EVIDENCE / "incoming-frontier.npz"
SLOPE_DEG = 50.0
RISE_M = 0.19
QUERY_TIME_S = 112.20
BASELINE_COMMIT = "632fcc21a2c6f0ceed236cfe83cef5837736c791"
OPEN_APERTURE_TOLERANCE_M = 0.00025
EXECUTE_TRACK_KEYS = (
    "time",
    "qpos",
    "target",
    "torque",
    "stance",
    "feet",
    "wall",
    "cap",
    "tip",
)


def support_signature(stance: dict) -> Counter:
    """Return the target support-surface multiset (surface and tread-top height)."""
    return Counter(
        (surface, round((float(point[2]) - 0.03) / RISE_M)) for point, surface in stance.values()
    )


def reorder_stances(before: dict, rl_after: dict, rr_after: dict) -> tuple[dict, dict]:
    """Move RR to its nominal target first, then RL to the same nominal final stance."""
    rr_first = {**before, "RR": rr_after["RR"]}
    final = {**rr_first, "RL": rl_after["RL"]}
    return rr_first, final


def closed_valve_model(study: dict):
    saved = study["candidates"][4]
    model, data = stair_valve.build_grasp_scene(
        study["slope_deg"],
        study["rise_m"],
        saved["centre_m"],
        saved["start_deg"],
        saved["sense"],
        saved["handle_length_m"],
        saved["resistance_nm"],
    )
    hinge_q = model.joint(stair_valve.VALVE_HINGE).qposadr[0]
    data.qpos[hinge_q] = np.pi / 2
    mujoco.mj_forward(model, data)
    return model, data, int(hinge_q)


def set_open_hand(model, qpos: np.ndarray) -> None:
    for name in stair_valve.GRIPPER_JOINTS:
        qpos[model.joint(name).qposadr[0]] = grasp_motion.OPEN_FINGER_M


def finger_trace(model, qpos: np.ndarray) -> dict[str, np.ndarray]:
    upper, lower = (qpos[:, model.joint(name).qposadr[0]] for name in stair_valve.GRIPPER_JOINTS)
    return {"upper_m": upper, "lower_m": lower, "aperture_m": upper + lower}


def unique_bases(items):
    seen = set()
    out = []
    for item in items:
        base = tuple(float(value) for value in item)
        key = tuple(round(value, 7) for value in base)
        if key not in seen:
            seen.add(key)
            out.append(base)
    return out


def candidate_bases(anchors: list[tuple], stance: dict, *, full_grid: bool) -> list[tuple]:
    """Local bases around actual neighbouring N1 poses, optionally plus the planner grid."""
    local = []
    for anchor in anchors:
        for dx in (-0.06, 0.0, 0.06):
            for dy in (-0.03, 0.0, 0.03, 0.06, 0.09):
                for dz in (-0.03, 0.0, 0.03):
                    for pitch in (-5, 0, 5):
                        local.append(
                            (
                                anchor[0] + float(dx),
                                anchor[1] + float(dy),
                                anchor[2] + float(dz),
                                anchor[3] + float(np.radians(pitch)),
                            )
                        )
    grid = []
    if full_grid:
        grid = [
            base
            for offset in stairs.GRID["pitch_offsets"]
            for base in stairs.candidates(stance, SLOPE_DEG, offset)
        ]
    return unique_bases([*anchors, *local, *grid])


def lever_collision_geoms(model) -> list[int]:
    lever = model.geom(stair_valve.VALVE_LEVER).id
    robot_root = model.body_rootid[model.body("base").id]
    explicit = {
        frozenset((int(first), int(second)))
        for first, second in zip(model.pair_geom1, model.pair_geom2, strict=True)
    }
    excluded = set(map(int, model.exclude_signature))

    def enabled(geom):
        if frozenset((lever, geom)) in explicit:
            return True
        bodies = sorted((int(model.geom_bodyid[lever]), int(model.geom_bodyid[geom])))
        if (bodies[0] << 16) + bodies[1] in excluded:
            return False
        return bool(
            (model.geom_contype[lever] & model.geom_conaffinity[geom])
            or (model.geom_contype[geom] & model.geom_conaffinity[lever])
        )

    return [
        geom
        for geom in range(model.ngeom)
        if model.body_rootid[model.geom_bodyid[geom]] == robot_root and enabled(geom)
    ]


def lever_clearance_m(model, data) -> float:
    lever = model.geom(stair_valve.VALVE_LEVER).id
    segment = np.empty(6)
    return float(
        min(
            mujoco.mj_geomDistance(model, data, lever, geom, 1.0, segment)
            for geom in lever_collision_geoms(model)
        )
    )


def quick_score(model, data, limits, before, after, leg, bases, arm, warm, *, robust):
    feet = {name: item[0] for name, item in after.items()}
    feet[leg] = stairs.swing_over_tread(before[leg], after[leg], 0.5)
    contacts = transition.contacts_for(before, skip=leg)
    passing = []
    for base in bases:
        nominal = transition.evaluate(
            model,
            data,
            base,
            feet,
            contacts,
            arm,
            limits,
            warm=warm,
        )
        if nominal is None or nominal.ratio > 1.0:
            continue
        solution = nominal
        if robust:
            solution = transition.evaluate(
                model,
                data,
                base,
                feet,
                contacts,
                arm,
                limits,
                robust=True,
                warm=warm,
            )
        if solution is not None and solution.ratio <= 1.0 and lever_clearance_m(model, data) >= 0.0:
            passing.append((float(solution.ratio), base))
    return sorted(passing)


def path_failure(model, data, limits, start, base, before, after, leg):
    """Locate and classify the first failed sample of the same robust path contract."""

    def classify(base_pose, feet, contacts, index):
        robust = transition.evaluate(
            model,
            data,
            base_pose,
            feet,
            contacts,
            transition.tuck_arm(),
            limits,
            robust=True,
            disturb=index % 3 == 0,
        )
        if robust is not None:
            return None
        clearance_only = transition.evaluate(
            model,
            data,
            base_pose,
            feet,
            contacts,
            transition.tuck_arm(),
            limits,
            robust=True,
            disturb=False,
        )
        if clearance_only is not None:
            return "disturbance_qp"
        nominal = transition.evaluate(
            model,
            data,
            base_pose,
            feet,
            contacts,
            transition.tuck_arm(),
            limits,
        )
        if nominal is not None:
            return "clearance"
        feet_only = transition.place(model, data, base_pose, feet, transition.tuck_arm())
        return "static_qp" if feet_only else "ik_or_collision"

    stand = {name: item[0] for name, item in before.items()}
    shift = np.linspace(0, 1, int(round(transition.SHIFT_S / transition.SAMPLE_S)) + 1)
    for index, progress in enumerate(shift):
        pose = transition.interpolate_base(start, base, progress)
        reason = classify(pose, stand, transition.contacts_for(before), index)
        if reason:
            return {
                "phase": "shift",
                "sample": index,
                "time_s": float(index * transition.SAMPLE_S),
                "progress": float(progress),
                "reason": reason,
                "base": list(pose),
            }

    swing = np.linspace(0, 1, int(round(transition.SWING_S / transition.SAMPLE_S)) + 1)
    for index, progress in enumerate(swing):
        feet = {name: item[0] for name, item in after.items()}
        feet[leg] = stairs.swing_over_tread(before[leg], after[leg], progress)
        contacts = transition.contacts_for(
            before if progress < 1 else after, skip=leg if progress < 1 else None
        )
        reason = classify(base, feet, contacts, index)
        if reason:
            return {
                "phase": "swing",
                "sample": index,
                "time_s": float(transition.SHIFT_S + index * transition.SAMPLE_S),
                "progress": float(progress),
                "reason": reason,
                "base": list(base),
            }
    return {"reason": "not_reproduced"}


def sample_transfer(model, data, limits, before, after, leg, path, initial_qpos, arm):
    """Sample one existing transfer block from a known feasible warm-start pose."""
    waypoints = [tuple(base) for base in path.get("waypoints", [])]
    if waypoints:
        plan = [
            {
                "kind": "hold",
                "base": tuple(path["start_base"]),
                "stance": before,
                "seconds": 0.4,
            }
        ]
        origin = tuple(path["start_base"])
        for waypoint in waypoints:
            plan.append(
                {
                    "kind": "shift",
                    "from": origin,
                    "to": waypoint,
                    "stance": before,
                    "seconds": transition.SHIFT_S,
                }
            )
            origin = waypoint
        inner = {**path, "start_base": origin}
        plan.extend(
            transfer_segments(before, after, leg, inner, swing_curve=stairs.swing_over_tread)[1:]
        )
    else:
        plan = transfer_segments(before, after, leg, path, swing_curve=stairs.swing_over_tread)
    addresses = np.array([model.joint(name).qposadr[0] for name in joint_names()])
    rows = {
        "time": [],
        "qpos": [],
        "target": [],
        "torque": [],
        "ratio": [],
        "stance": [],
        "feet": [],
        "wall": [],
        "cap": [],
        "tip": [],
        "phase": [],
    }
    warm = np.asarray(initial_qpos).copy()
    clock = 0.0
    for segment_index, segment in enumerate(plan):
        count = int(round(segment["seconds"] / transition.SAMPLE_S))
        handoff = None
        if segment["kind"] in ("unload", "load"):
            stand = {name: item[0] for name, item in segment["stance"].items()}
            full = transition.evaluate(
                model,
                data,
                segment["base"],
                stand,
                transition.contacts_for(segment["stance"]),
                arm,
                limits,
                warm=warm,
            )
            if full is None:
                return None, {
                    "phase": segment["kind"],
                    "time_s": clock,
                    "progress": 0.0,
                    "reason": "full_support_pose",
                }
            handoff = float(full.forces[leg] @ transition.normal_of(segment["stance"][leg][1]))
        for sample in range(0 if segment_index == 0 else 1, count + 1):
            progress = sample / count
            cap = None
            if segment["kind"] in ("hold", "unload", "load"):
                base = segment["base"]
                stance = segment["stance"]
                feet = {name: item[0] for name, item in stance.items()}
                contacts = transition.contacts_for(stance)
                if segment["kind"] == "unload":
                    cap = {leg: (1 - progress) * handoff}
                elif segment["kind"] == "load":
                    cap = {leg: progress * handoff}
                flags = [True] * 4
            elif segment["kind"] == "shift":
                base = transition.interpolate_base(segment["from"], segment["to"], progress)
                stance = segment["stance"]
                feet = {name: item[0] for name, item in stance.items()}
                contacts = transition.contacts_for(stance)
                flags = [True] * 4
            else:
                base = segment["base"]
                feet = {name: item[0] for name, item in segment["after"].items()}
                feet[leg] = stairs.swing_over_tread(
                    segment["before"][leg], segment["after"][leg], progress
                )
                stance = segment["before"] if progress < 1 else segment["after"]
                contacts = transition.contacts_for(stance, skip=leg if progress < 1 else None)
                cap = None if progress < 1 else {leg: 0.0}
                flags = [name != leg or progress >= 1 for name in LEGS]
            solution = transition.evaluate(
                model,
                data,
                base,
                feet,
                contacts,
                arm,
                limits,
                max_normal=cap,
                warm=warm,
            )
            clearance = None if solution is None else lever_clearance_m(model, data)
            if solution is not None and clearance < 0.0:
                return None, {
                    "phase": segment["kind"],
                    "time_s": float(clock + progress * segment["seconds"]),
                    "progress": float(progress),
                    "reason": "lever_geometric_overlap",
                    "lever_clearance_m": clearance,
                    "base": list(base),
                }
            if solution is None:
                placed = transition.place(
                    model,
                    data,
                    base,
                    feet,
                    arm,
                    warm=warm,
                )
                foot_errors = {
                    name: float(np.linalg.norm(data.geom(name).xpos - feet[name])) for name in LEGS
                }
                overlaps = []
                for contact in data.contact[: data.ncon]:
                    if contact.dist >= -1e-4:
                        continue
                    overlaps.append(
                        {
                            "geoms": [
                                model.geom(int(contact.geom1)).name
                                or model.body(model.geom_bodyid[int(contact.geom1)]).name,
                                model.geom(int(contact.geom2)).name
                                or model.body(model.geom_bodyid[int(contact.geom2)]).name,
                            ],
                            "penetration_m": float(-contact.dist),
                        }
                    )
                return None, {
                    "phase": segment["kind"],
                    "time_s": float(clock + progress * segment["seconds"]),
                    "progress": float(progress),
                    "reason": "static_qp" if placed else "ik_or_collision",
                    "base": list(base),
                    "max_foot_ik_error_m": max(foot_errors.values()),
                    "foot_ik_errors_m": foot_errors,
                    "overlaps": overlaps,
                }
            warm = data.qpos.copy()
            rows["time"].append(clock + progress * segment["seconds"])
            rows["qpos"].append(warm.copy())
            rows["target"].append(warm[addresses].copy())
            rows["torque"].append(solution.torque.copy())
            rows["ratio"].append(float(solution.ratio))
            rows["stance"].append(flags)
            rows["feet"].append([data.geom(name).xpos.copy() for name in LEGS])
            rows["wall"].append([False] * 4)
            rows["cap"].append([(cap or {}).get(name, np.inf) for name in LEGS])
            rows["tip"].append(np.zeros(3))
            rows["phase"].append(segment["kind"])
        clock += segment["seconds"]
    return {key: np.asarray(value) for key, value in rows.items()}, None


def connect(
    model,
    data,
    limits,
    start,
    before,
    after,
    leg,
    scored,
    max_paths,
    initial_qpos,
    arm,
    via_bases=(),
):
    attempts = []
    ranked = sorted(
        scored,
        key=lambda item: (
            np.linalg.norm(np.subtract(start[:3], item[1][:3])),
            abs(start[3] - item[1][3]),
            item[0],
        ),
    )
    for ratio, base in ranked[:max_paths]:
        for waypoint in [None, *via_bases]:
            path = {"start_base": start, "swing_base": base}
            if waypoint is not None:
                path["waypoints"] = [waypoint]
            track, failure = sample_transfer(
                model, data, limits, before, after, leg, path, initial_qpos, arm
            )
            cost = None if track is None else float(np.max(track["ratio"]))
            attempt = {
                "base": list(base),
                "waypoint": None if waypoint is None else list(waypoint),
                "midpoint_ratio": ratio,
                "physical_max_ratio": cost,
            }
            if failure is not None:
                attempt["first_failure"] = failure
            attempts.append(attempt)
            if cost is not None:
                path["physical_max_ratio"] = cost
                return path, attempts, track
    return None, attempts, None


def json_stance(stance):
    return {
        leg: {"xyz_m": np.asarray(point).tolist(), "surface": surface}
        for leg, (point, surface) in stance.items()
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report_path(path: Path) -> str:
    """Render repository files without leaking a machine-specific checkout path."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def thread_environment() -> dict[str, str | None]:
    return {
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }


def compare_raw(first: Path, second: Path, out: Path) -> None:
    """Record bitwise repeatability without making the second raw log public evidence."""
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    with np.load(first, allow_pickle=False) as left, np.load(second, allow_pickle=False) as right:
        keys_match = left.files == right.files
        arrays = {}
        for key in left.files:
            same = key in right.files and np.array_equal(left[key], right[key])
            arrays[key] = {
                "equal": bool(same),
                "shape": list(left[key].shape),
                "dtype": str(left[key].dtype),
                "content_sha256": hashlib.sha256(left[key].tobytes()).hexdigest(),
            }
        report = {
            "bitwise_identical": bool(
                keys_match and all(item["equal"] for item in arrays.values())
            ),
            "first_raw_sha256": sha256(first),
            "repeat_raw_sha256": sha256(second),
            "repeat_raw_preserved": False,
            "thread_environment": thread_environment(),
            "arrays": arrays,
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"bitwise_identical": report["bitwise_identical"], "report": str(out)}))


def write_fingerprint(paths: list[Path], out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    report = {
        "baseline_commit": BASELINE_COMMIT,
        "executor_sources_changed_from_baseline": ["tools/stair_valve_swing_order.py"],
        "normalization": (
            "source_sha256_lf replaces CRLF with LF before SHA-256; artifact_sha256 hashes "
            "the exact stored bytes"
        ),
        "provenance": provenance("tools/stair_valve_swing_order.py"),
        "artifact_sha256": {path.as_posix(): sha256(path) for path in paths},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifacts": len(paths), "report": str(out)}))


def execute_track(track_path: Path, out: Path) -> None:
    """Run the nine-key reorder track through transition.execute without state rewrites."""
    raw_path = out.with_suffix(".raw.npz")
    for path in (out, raw_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    track_hash_at_start = sha256(track_path)
    provenance_at_start = provenance("tools/stair_valve_swing_order.py")
    _, _, study, _, _, _ = route(SLOPE_DEG, RISE_M)
    saved = study["candidates"][4]
    built = {}

    def builder():
        model, data = stair_valve.build_grasp_scene(
            study["slope_deg"],
            study["rise_m"],
            saved["centre_m"],
            saved["start_deg"],
            saved["sense"],
            saved["handle_length_m"],
            saved["resistance_nm"],
        )
        built["model"] = model
        return model, data

    def watch(model, data):
        hinge = model.joint(stair_valve.VALVE_HINGE)
        lever = model.geom(stair_valve.VALVE_LEVER).id
        world = {geom for geom in range(model.ngeom) if model.geom_bodyid[geom] == 0}
        foot_ids = [model.geom(name).id for name in LEGS]
        surface = np.full(4, -1.0)
        surface_force = np.zeros(4)
        lever_count = 0
        lever_force = 0.0
        finger = model.joint(stair_valve.GRIPPER_JOINTS[0])
        lower = model.joint(stair_valve.GRIPPER_JOINTS[1])
        finger_actuator = model.actuator(stair_valve.GRIPPER_ACTUATOR).id
        wrench = np.empty(6)
        for index, contact in enumerate(data.contact[: data.ncon]):
            pair = {int(contact.geom1), int(contact.geom2)}
            mujoco.mj_contactForce(model, data, index, wrench)
            force = float(np.linalg.norm(wrench[:3]))
            if lever in pair:
                lever_count += 1
                lever_force = max(lever_force, force)
            for foot_index, foot in enumerate(foot_ids):
                if foot not in pair or not pair & world or force <= surface_force[foot_index]:
                    continue
                other = next(geom for geom in pair if geom != foot)
                surface[foot_index] = other
                surface_force[foot_index] = force
        return np.r_[
            data.qpos[hinge.qposadr[0]],
            data.qvel[hinge.dofadr[0]],
            lever_count,
            lever_force,
            surface,
            data.qpos[finger.qposadr[0]],
            data.qpos[lower.qposadr[0]],
            data.qpos[finger.qposadr[0]] + data.qpos[lower.qposadr[0]],
            data.qvel[finger.dofadr[0]],
            data.actuator_force[finger_actuator],
            data.ctrl[finger_actuator],
        ]

    with np.load(track_path, allow_pickle=False) as source:
        if tuple(source.files) != EXECUTE_TRACK_KEYS:
            raise ValueError(
                f"Expected exact nine-key track {EXECUTE_TRACK_KEYS}, got {tuple(source.files)}"
            )
        track = {key: source[key] for key in source.files}
    if not np.isclose(track["qpos"][0, -1], np.pi / 2):
        raise ValueError("Initial track sample must set the valve hinge to 90 degrees")
    track_fingers = finger_trace(builder()[0], track["qpos"])
    if not all(
        np.allclose(values, grasp_motion.OPEN_FINGER_M)
        for name, values in track_fingers.items()
        if name != "aperture_m"
    ):
        raise ValueError("Every planning sample must keep both fingers at OPEN_FINGER_M")
    built.clear()

    def open_command(time_s, position_m, velocity_mps):
        force_limit = float(built["model"].actuator(stair_valve.GRIPPER_ACTUATOR).ctrlrange[1])
        return grasp_motion.open_finger_command(position_m, velocity_mps, force_limit_n=force_limit)

    metrics, log = transition.execute(
        track,
        wall_x=None,
        builder=builder,
        watch=watch,
        limits_nm=torque_limits("rb1"),
        mode="force",
        log_every_step=True,
        actuator_command=open_command,
        actuation_period_s=0.005,
    )
    model = built["model"]
    track_hash_at_end = sha256(track_path)
    if track_hash_at_end != track_hash_at_start:
        raise RuntimeError("Execution track changed during replay")
    provenance_at_end = provenance("tools/stair_valve_swing_order.py")
    if provenance_at_end != provenance_at_start:
        raise RuntimeError("Controller, assets or route inputs changed during replay")
    times = log["time"]
    indices = np.minimum(
        np.searchsorted(track["time"], times, side="right") - 1,
        len(track["time"]) - 2,
    )
    expected_stance = track["stance"][indices].astype(bool)
    caps = track["cap"][indices]
    required_support = expected_stance & ((caps > 1.0) | ~np.isfinite(caps))
    normal_force = np.abs(log["force"][:, :, 2])
    supported = normal_force > 1.0
    support_failures = required_support & ~supported

    extra = log["extra"]
    actual_surfaces = extra[:, 4:8].astype(int)
    foot_radius = model.geom("FL").size[0]
    expected_surfaces = np.empty_like(actual_surfaces)
    for sample, track_index in enumerate(indices):
        for foot_index, point in enumerate(track["feet"][track_index]):
            level = round((float(point[2]) - foot_radius) / RISE_M)
            name = "floor" if level <= 0 else f"tread{level}"
            expected_surfaces[sample, foot_index] = model.geom(name).id
    surface_failures = required_support & supported & (actual_surfaces != expected_surfaces)

    terminal_xy = np.linalg.norm(log["feet"][-1, :, :2] - track["feet"][-1, :, :2], axis=1)
    movement = np.linalg.norm(track["qpos"][:, :7] - track["qpos"][0, :7], axis=1)
    moving = np.flatnonzero(movement > 1e-8)
    first_motion_s = float(track["time"][moving[0]]) if len(moving) else None
    finger_actuator = model.actuator(stair_valve.GRIPPER_ACTUATOR).id
    hinge_angle = extra[:, 0]
    hinge_speed = extra[:, 1]
    lever_contact_force = extra[:, 3]
    dynamic_fingers = finger_trace(model, log["qpos"])
    measured_aperture = extra[:, 10]
    planned_base = (
        1
        - ((times - track["time"][indices]) / (track["time"][indices + 1] - track["time"][indices]))
    )[:, None] * track["qpos"][indices, :3] + (
        (times - track["time"][indices]) / (track["time"][indices + 1] - track["time"][indices])
    )[:, None] * track["qpos"][indices + 1, :3]
    base_error = np.linalg.norm(log["base"][:, :3] - planned_base, axis=1)

    def first_failure(mask, *, legged=False):
        rows = np.argwhere(mask)
        if not len(rows):
            return None
        sample = int(rows[0, 0])
        result = {"time_s": float(times[sample]), "sample": sample}
        if legged:
            result["leg"] = LEGS[int(rows[0, 1])]
        return result

    first_support = first_failure(support_failures, legged=True)
    if first_support:
        leg_index = LEGS.index(first_support["leg"])
        first_support["normal_force_n"] = float(normal_force[first_support["sample"], leg_index])
    first_surface = first_failure(surface_failures, legged=True)
    if first_surface:
        sample, leg_index = first_surface["sample"], LEGS.index(first_surface["leg"])
        first_surface.update(
            actual_geom=model.geom(int(actual_surfaces[sample, leg_index])).name,
            expected_geom=model.geom(int(expected_surfaces[sample, leg_index])).name,
        )
    first_lever = first_failure(extra[:, 2] > 0)
    if first_lever:
        first_lever["force_n"] = float(lever_contact_force[first_lever["sample"]])
    first_qp = first_failure(log["attempt"] < 0)
    first_base = first_failure(base_error > transition.BASE_TRACK_LIMIT_M)
    if first_base:
        first_base["error_m"] = float(base_error[first_base["sample"]])
    first_hinge = first_failure(np.degrees(hinge_angle) < 89.0)

    settling = times <= (first_motion_s or 0.0)
    settling_metrics = {
        "end_s": first_motion_s,
        "maximum_base_displacement_m": float(
            np.linalg.norm(log["base"][settling, :3] - log["base"][0, :3], axis=1).max()
        ),
        "minimum_foot_normal_n": {
            leg: float(normal_force[settling, index].min()) for index, leg in enumerate(LEGS)
        },
    }

    support = {}
    for index, leg in enumerate(LEGS):
        mask = required_support[:, index]
        support[leg] = {
            "required_samples": int(mask.sum()),
            "samples_over_1n": int((mask & supported[:, index]).sum()),
            "minimum_required_normal_n": float(normal_force[mask, index].min())
            if np.any(mask)
            else None,
            "end_normal_n": float(normal_force[-1, index]),
            "surface_mismatch_samples": int(surface_failures[:, index].sum()),
            "terminal_xy_error_m": float(terminal_xy[index]),
        }

    dynamic_pass = bool(
        not metrics["fell"]
        and metrics["saturated_steps"] == 0
        and metrics["qp_failed_ticks"] == 0
        and metrics["max_ratio"] <= 1.0
        and metrics["max_base_track_mm"] <= transition.BASE_TRACK_LIMIT_M * 1e3
        and max(metrics["max_stance_slip_mm"].values()) <= transition.STANCE_SLIP_LIMIT_M * 1e3
        and not np.any(support_failures)
        and not np.any(surface_failures)
        and dynamic_fingers["aperture_m"].min()
        >= 2 * grasp_motion.OPEN_FINGER_M - OPEN_APERTURE_TOLERANCE_M
    )

    inputs = [
        track_path,
        Path(__file__),
        ROOT / "src/go2_wall/transition.py",
        ROOT / "src/go2_wall/stair_valve.py",
        ROOT / "evidence/t3-valve-20261120/fixed-body-candidates.json",
    ]
    report = {
        "outcome": "dynamic_pass" if dynamic_pass else "dynamic_failed",
        "scope": "independent cold-start RR-before-RL segment; not reached from the floor",
        "entry_point": (
            f"python tools/stair_valve_swing_order.py --execute-track {track_path} "
            f"--execute-out {out}"
        ),
        "track_contract": {"keys": list(EXECUTE_TRACK_KEYS), "samples": len(track["time"])},
        "input_audit": {
            "track_sha256_at_start": track_hash_at_start,
            "track_sha256_at_end": track_hash_at_end,
            "unchanged_during_run": True,
        },
        "provenance": provenance_at_start,
        "thread_environment": thread_environment(),
        "baseline_commit": BASELINE_COMMIT,
        "executor_sources_changed_from_baseline": ["tools/stair_valve_swing_order.py"],
        "initial_conditions": {
            "qpos": track["qpos"][0].tolist(),
            "qvel": [0.0] * model.nv,
            "hinge_initialized_once_rad": float(track["qpos"][0, -1]),
            "runtime_state_overwrite": False,
            "open_hand_physical_drive": True,
            "finger_joint_qpos": {
                name: float(track["qpos"][0, model.joint(name).qposadr[0]])
                for name in stair_valve.GRIPPER_JOINTS
            },
            "finger_aperture_m": float(track_fingers["aperture_m"][0]),
            "settling_hold_s": first_motion_s,
        },
        "physics": {
            "model_timestep_s": float(model.opt.timestep),
            "controller": "transition.execute force mode; measured-state QP + PD",
            "physics_overrides": {},
            "actuation_period_s": 0.005,
            "finger_actuator_max_abs_ctrl_n": float(
                np.max(np.abs(log["ctrl"][:, finger_actuator]))
            ),
            "physics_clock_max_error_s": float(
                np.max(np.abs(log["physics_time_s"] - (log["time"] + model.opt.timestep)))
            ),
        },
        "finger_drive": {
            "target_per_finger_m": grasp_motion.OPEN_FINGER_M,
            "upper_slide_range_m": [
                float(dynamic_fingers["upper_m"].min()),
                float(dynamic_fingers["upper_m"].max()),
            ],
            "lower_slide_range_m": [
                float(dynamic_fingers["lower_m"].min()),
                float(dynamic_fingers["lower_m"].max()),
            ],
            "aperture_range_m": [
                float(dynamic_fingers["aperture_m"].min()),
                float(dynamic_fingers["aperture_m"].max()),
            ],
            "minimum_aperture_criterion_m": (
                2 * grasp_motion.OPEN_FINGER_M - OPEN_APERTURE_TOLERANCE_M
            ),
            "minimum_aperture_criterion_met": bool(
                dynamic_fingers["aperture_m"].min()
                >= 2 * grasp_motion.OPEN_FINGER_M - OPEN_APERTURE_TOLERANCE_M
            ),
            "watch_aperture_matches_qpos": bool(
                np.array_equal(measured_aperture, dynamic_fingers["aperture_m"])
            ),
            "requested_force_range_n": [
                float(log["requested_finger_force_n"].min()),
                float(log["requested_finger_force_n"].max()),
            ],
            "applied_ctrl_range_n": [
                float(extra[:, 13].min()),
                float(extra[:, 13].max()),
            ],
            "actuator_force_range_n": [
                float(extra[:, 12].min()),
                float(extra[:, 12].max()),
            ],
            "upper_velocity_range_mps": [
                float(extra[:, 11].min()),
                float(extra[:, 11].max()),
            ],
            "audit": metrics["gripper_audit"],
        },
        "controller_metrics": metrics,
        "support": support,
        "support_failure_samples": int(support_failures.sum()),
        "surface_failure_samples": int(surface_failures.sum()),
        "settling": settling_metrics,
        "failure_timeline": {
            "first_required_support_below_1n": first_support,
            "first_surface_mismatch": first_surface,
            "first_base_tracking_over_30mm": first_base,
            "first_lever_contact": first_lever,
            "first_qp_failure": first_qp,
            "first_hinge_below_89deg": first_hinge,
        },
        "hinge": {
            "angle_range_deg": np.degrees([hinge_angle.min(), hinge_angle.max()]).tolist(),
            "maximum_speed_rad_s": float(np.max(np.abs(hinge_speed))),
        },
        "lever_contacts": {
            "samples": int(np.count_nonzero(extra[:, 2])),
            "maximum_force_n": float(lever_contact_force.max()),
        },
        "terminal_max_foot_xy_error_m": float(terminal_xy.max()),
        "terminal_actual_surface_geom": {
            leg: None
            if actual_surfaces[-1, index] < 0
            else model.geom(int(actual_surfaces[-1, index])).name
            for index, leg in enumerate(LEGS)
        },
        "hashes_sha256": {report_path(path): sha256(path) for path in inputs},
        "limitation": (
            "The segment starts from the accepted 112.02 s bypass state with zero velocity; "
            "it does not prove physical reachability from the floor to 112.02 s."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, **log)
    report["raw_log"] = {
        "path": str(raw_path),
        "sha256": sha256(raw_path),
        "samples": len(times),
        "keys": list(log),
    }
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"outcome": report["outcome"], "report": str(out)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk", type=Path, default=DEFAULT_WALK)
    parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-paths", type=int, default=120)
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--execute-track", type=Path)
    parser.add_argument("--execute-out", type=Path)
    parser.add_argument("--compare-raw", nargs=2, type=Path, metavar=("FIRST", "REPEAT"))
    parser.add_argument("--compare-out", type=Path)
    parser.add_argument("--fingerprint", nargs="+", type=Path)
    parser.add_argument("--fingerprint-out", type=Path)
    parser.add_argument(
        "--recheck", type=Path, help="recheck the RR bases already saved in a prior report"
    )
    args = parser.parse_args()

    if args.fingerprint is not None:
        if args.fingerprint_out is None:
            parser.error("--fingerprint requires --fingerprint-out")
        write_fingerprint(args.fingerprint, args.fingerprint_out)
        return
    if args.compare_raw is not None:
        if args.compare_out is None:
            parser.error("--compare-raw requires --compare-out")
        compare_raw(*args.compare_raw, args.compare_out)
        return
    if args.execute_track is not None:
        if args.execute_out is None:
            parser.error("--execute-track requires --execute-out")
        execute_track(args.execute_track, args.execute_out)
        return

    companions = (args.out, args.out.with_suffix(".npz"), args.out.with_suffix(".track.npz"))
    for path in companions:
        if path.exists():
            parser.error(f"Refusing to overwrite {path}")
    input_paths = [args.walk, args.frontier]
    if args.recheck is not None:
        input_paths.append(args.recheck)
    input_hashes_at_start = {str(path): sha256(path) for path in input_paths}
    provenance_at_start = provenance("tools/stair_valve_swing_order.py")

    geometry, sequence, study, segments, ends, _ = route(SLOPE_DEG, RISE_M)
    stairs.ARM = geometry["arm"]
    stairs.use_stow()
    limits = torque_limits(geometry["arm"])
    model, data, hinge_q = closed_valve_model(study)
    with np.load(args.walk, allow_pickle=False) as walk:
        row = int(np.argmin(np.abs(walk["time"] - QUERY_TIME_S)))
        observed = {
            "row": row,
            "time_s": float(walk["time"][row]),
            "segment": int(walk["segment"][row]),
            "stance": walk["stance"][row].astype(bool).tolist(),
            "feet_xyz_m": walk["feet"][row].tolist(),
            "base_xyz_m": walk["qpos"][row, :3].tolist(),
        }

        with np.load(args.frontier, allow_pickle=False) as frontier:
            frontier_time = float(frontier["time_s"][-1])
            frontier_row = int(np.argmin(np.abs(walk["time"] - frontier_time)))
            body_offset = frontier["body_offset"][-1]
            foot_x = frontier["foot_x_delta_m"][-1]
            foot_y = frontier["foot_y_delta_m"][-1]
            data.qpos[:] = model.qpos0
            data.qpos[: frontier["qpos"].shape[1]] = frontier["qpos"][-1]
            data.qpos[hinge_q] = np.pi / 2
            set_open_hand(model, data.qpos)
            mujoco.mj_forward(model, data)
            frontier_qpos = data.qpos.copy()
            arm_pose = {
                name: float(frontier_qpos[model.joint(name).qposadr[0]])
                for name in transition.tuck_arm()
            }
            frontier_feet = {leg: data.geom(leg).xpos.copy() for leg in LEGS}
            start = (
                *frontier["qpos"][-1, :3].tolist(),
                transition.base_of(walk["qpos"][frontier_row])[3]
                + float(np.radians(body_offset[3])),
                float(np.radians(body_offset[4])),
                float(np.radians(body_offset[5])),
            )
            frontier_info = {
                "path": str(args.frontier),
                "time_s": frontier_time,
                "body_offset": body_offset.tolist(),
                "foot_x_delta_m": foot_x.tolist(),
                "foot_y_delta_m": foot_y.tolist(),
                "ratio": float(frontier["ratio"][-1]),
                "lever_clearance_m": float(frontier["lever_clearance_m"][-1]),
                "start_base": list(start),
                "arm_pose_rad": arm_pose,
            }

    # Milestones 21, 22 and 23 are the actual FL-complete, RL-complete and RR-complete
    # supports surrounding the queried RL swing. Derive them from moved legs, not hard-coded rows.
    rl_index = next(
        i for i, (end_s, _, leg) in enumerate(ends) if leg == "RL" and end_s > QUERY_TIME_S
    )
    rr_index = next(i for i in range(rl_index + 1, len(ends)) if ends[i][2] == "RR")
    before = {
        leg: (np.asarray(point).copy(), surface)
        for leg, (point, surface) in ends[rl_index - 1][1].items()
    }
    for leg in LEGS:
        before[leg] = (frontier_feet[leg], before[leg][1])
    rl_after = ends[rl_index][1]
    rr_after = ends[rr_index][1]
    rr_first, final = reorder_stances(before, rl_after, rr_after)

    original_multiset = support_signature(rr_after)
    reordered_multiset = support_signature(final)
    if original_multiset != reordered_multiset:
        raise RuntimeError("Reorder changed the final support target multiset")

    nearby = unique_bases(
        seg[key] for seg in segments[107:120] for key in ("base", "from", "to") if key in seg
    )
    if args.recheck:
        prior = json.loads(args.recheck.read_text(encoding="utf-8"))
        first_scored = [
            (item["midpoint_ratio"], tuple(item["base"]))
            for item in prior["search"]["rr_first_path_attempts"]
        ]
    else:
        first_scored = quick_score(
            model,
            data,
            limits,
            before,
            rr_first,
            "RR",
            candidate_bases(nearby, before, full_grid=args.full_grid),
            arm_pose,
            frontier_qpos,
            robust=False,
        )
    first_path, first_attempts, first_track = connect(
        model,
        data,
        limits,
        start,
        before,
        rr_first,
        "RR",
        first_scored,
        args.max_paths,
        frontier_qpos,
        arm_pose,
    )

    report = {
        "scope": "candidate4 shut-valve real RR-before-RL reorder at the 112.20 s swing",
        "inputs": {
            "walk": str(args.walk),
            "slope_deg": SLOPE_DEG,
            "rise_m": RISE_M,
            "query_time_s": QUERY_TIME_S,
            "candidate": 4,
            "hinge_rad": float(frontier_qpos[hinge_q]),
        },
        "input_audit": {
            "sha256_at_start": input_hashes_at_start,
            "sha256_at_end": {str(path): sha256(path) for path in input_paths},
        },
        "provenance": provenance_at_start,
        "thread_environment": thread_environment(),
        "baseline_commit": BASELINE_COMMIT,
        "executor_sources_changed_from_baseline": ["tools/stair_valve_swing_order.py"],
        "observed_query": observed,
        "incoming_bypass_frontier": frontier_info,
        "original": {
            "first": {"leg": "RL", "end_s": ends[rl_index][0], "stance": json_stance(rl_after)},
            "second": {"leg": "RR", "end_s": ends[rr_index][0], "stance": json_stance(rr_after)},
        },
        "reordered": {
            "first": {"leg": "RR", "stance": json_stance(rr_first)},
            "second": {"leg": "RL", "stance": json_stance(final)},
        },
        "contracts": {
            # The cached mission has two entry landings before sequence["moves"].
            "nominal_intended_landings": len(sequence["moves"]) + 2,
            "reordered_intended_landings": len(sequence["moves"]) + 2,
            "landing_count_preserved": True,
            "target_support_multiset_preserved": True,
            "handoff_support_feet": 4,
            "swing_support_feet": 3,
            "sample_period_s": transition.SAMPLE_S,
        },
        "search": {
            "rr_first_quick_passes": len(first_scored),
            "rr_first_path_attempts": first_attempts,
        },
        "outcome": "rr_first_no_connected_path",
    }

    track = None
    if first_path is not None:
        if args.recheck and prior["search"].get("rl_second_path_attempts"):
            seen = set()
            second_scored = []
            for item in prior["search"]["rl_second_path_attempts"]:
                base = tuple(item["base"])
                if base not in seen:
                    seen.add(base)
                    second_scored.append((item["midpoint_ratio"], base))
        else:
            second_scored = quick_score(
                model,
                data,
                limits,
                rr_first,
                final,
                "RL",
                candidate_bases(
                    [first_path["swing_base"], *nearby], rr_first, full_grid=args.full_grid
                ),
                arm_pose,
                first_track["qpos"][-1],
                robust=False,
            )
        first_base = first_path["swing_base"]
        via_bases = unique_bases(
            (
                first_base[0] + dx,
                y,
                first_base[2] + dz,
                first_base[3] + np.radians(pitch),
            )
            for dz in (0.06, 0.09, 0.12, 0.15, 0.18)
            for dx in (0.0, -0.03, 0.03)
            for y in (0.06, 0.09)
            for pitch in (0, 5, 10)
        )
        second_path, second_attempts, second_track = connect(
            model,
            data,
            limits,
            tuple(first_path["swing_base"]),
            rr_first,
            final,
            "RL",
            second_scored,
            args.max_paths,
            first_track["qpos"][-1],
            arm_pose,
            via_bases,
        )
        report["search"].update(
            rl_second_quick_passes=len(second_scored), rl_second_path_attempts=second_attempts
        )
        if second_path is not None:
            track = {
                key: np.concatenate(
                    (
                        first_track[key],
                        second_track[key][1:]
                        if key != "time"
                        else second_track[key][1:] + first_track["time"][-1],
                    )
                )
                for key in first_track
            }
            swing_counts = {
                leg: int(
                    np.count_nonzero(
                        (track["stance"].sum(axis=1) == 3) & ~track["stance"][:, LEGS.index(leg)]
                    )
                )
                for leg in ("RR", "RL")
            }
            hinge_values = track["qpos"][:, hinge_q]
            lever = model.geom(stair_valve.VALVE_LEVER).id
            robot_geoms = lever_collision_geoms(model)
            segment = np.empty(6)
            lever_clearances = []
            closest = None
            lever_contact_samples = 0
            maximum_lever_contact_force = 0.0
            wrench = np.empty(6)
            for sample_index, qpos in enumerate(track["qpos"]):
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                distances = [
                    (mujoco.mj_geomDistance(model, data, lever, geom, 1.0, segment), geom)
                    for geom in robot_geoms
                ]
                distance, geom = min(distances)
                lever_clearances.append(distance)
                if closest is None or distance < closest["distance_m"]:
                    closest = {
                        "sample": sample_index,
                        "time_s": float(track["time"][sample_index]),
                        "distance_m": float(distance),
                        "geom_id": int(geom),
                        "geom": model.geom(geom).name or model.body(model.geom_bodyid[geom]).name,
                        "combined_margin_m": float(
                            model.geom_margin[lever] + model.geom_margin[geom]
                        ),
                    }
                sample_has_contact = False
                for contact_index, contact in enumerate(data.contact[: data.ncon]):
                    if lever not in contact.geom:
                        continue
                    sample_has_contact = True
                    mujoco.mj_contactForce(model, data, contact_index, wrench)
                    maximum_lever_contact_force = max(
                        maximum_lever_contact_force, float(np.linalg.norm(wrench[:3]))
                    )
                lever_contact_samples += int(sample_has_contact)
            terminal_feet = {name: data.geom(name).xpos.copy() for name in LEGS}
            terminal_errors = {
                name: float(np.linalg.norm(terminal_feet[name] - final[name][0])) for name in LEGS
            }
            report.update(
                outcome="verified_reorder",
                paths={"rr_first": first_path, "rl_second": second_path},
                verification={
                    "samples": len(track["time"]),
                    "duration_s": float(track["time"][-1]),
                    "max_torque_ratio": float(track["ratio"].max()),
                    "min_support_feet": int(track["stance"].sum(axis=1).min()),
                    "max_support_feet": int(track["stance"].sum(axis=1).max()),
                    "three_foot_swing_samples": swing_counts,
                    "hinge_min_rad": float(hinge_values.min()),
                    "hinge_max_rad": float(hinge_values.max()),
                    "minimum_lever_clearance_m": float(min(lever_clearances)),
                    "closest_collision_enabled_lever_pair": closest,
                    "lever_contact_samples": lever_contact_samples,
                    "maximum_lever_contact_force_n": maximum_lever_contact_force,
                    "collision_filter": {
                        "explicit_pairs": int(model.npair),
                        "excluded_body_pairs": int(model.nexclude),
                        "eligible_robot_geoms": len(robot_geoms),
                    },
                    "maximum_terminal_foot_error_m": max(terminal_errors.values()),
                    "terminal_foot_errors_m": terminal_errors,
                    "collision_contract": (
                        "every sample passed transition.place active-contact gate"
                    ),
                    "all_finite": bool(
                        all(np.all(np.isfinite(track[key])) for key in ("qpos", "ratio"))
                    ),
                },
            )
        else:
            report["outcome"] = "rr_first_connected_but_rl_second_no_connected_path"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if report["input_audit"]["sha256_at_start"] != report["input_audit"]["sha256_at_end"]:
        raise RuntimeError("Planning inputs changed during the run")
    if provenance("tools/stair_valve_swing_order.py") != provenance_at_start:
        raise RuntimeError("Planner, assets or route inputs changed during the run")
    report["input_audit"]["unchanged_during_run"] = True
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if track is not None:
        np.savez_compressed(args.out.with_suffix(".npz"), **track)
        np.savez_compressed(
            args.out.with_suffix(".track.npz"),
            **{key: track[key] for key in EXECUTE_TRACK_KEYS},
        )
    print(json.dumps({"outcome": report["outcome"], "report": str(args.out)}))


if __name__ == "__main__":
    main()
