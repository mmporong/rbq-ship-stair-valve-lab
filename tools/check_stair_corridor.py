"""Check a fixed stair walking reference against the compiled RBQ model and the closed lever."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from stair_mission_reference import (
    corridor_lateral_offsets,
    file_sha256,
    reference_samples,
    scheduled_landings,
)
from stair_mission_route import route

from go2_wall import grasp_motion, stair_valve, stairs, transition
from go2_wall.robot import LEGS, ROOT, world_geoms
from go2_wall.statics import joint_names, solve_static, torque_limits

BASELINE = ROOT / "evidence/stair-mission-20261121/full/walk50.npz"


def check_tip_forces(recorded, nominal, left, right, weight):
    """Keep the controller's assumed tip force tied to the prescribed walking route."""
    expected = nominal[left] + weight[:, None] * (nominal[right] - nominal[left])
    if recorded.shape != expected.shape or not np.allclose(recorded, expected, atol=1e-12, rtol=0):
        raise ValueError("Walking reference changes the prescribed tip force")


def check_statics(path: Path, nominal_path: Path) -> dict:
    """Re-solve the stored support phases with the stair planner's friction reserve."""
    with np.load(path, allow_pickle=False) as candidate, np.load(nominal_path) as nominal:
        poses = candidate["qpos"]
        flags, caps, tip = candidate["stance"], candidate["cap"], candidate["tip"]
        left, _, _, knot = reference_samples(candidate["reference_time"], nominal["time"])
        changed = np.flatnonzero(~knot | np.any(poses != nominal["qpos"][left], axis=1))
    previous_arm = stairs.ARM
    try:
        stairs.ARM = "rb1"
        model, data = stairs.build_stairs(50.0, 0.19)
    finally:
        stairs.ARM = previous_arm
    limits = torque_limits("rb1")
    floor = transition.contacts_for({leg: (None, "floor") for leg in LEGS})
    failures = []
    peak = 0.0
    for i in changed:
        data.qpos[:] = poses[i]
        contacts = [contact for contact, on in zip(floor, flags[i], strict=True) if on]
        cap = {
            leg: float(value)
            for leg, value in zip(LEGS, caps[i], strict=True)
            if np.isfinite(value)
        }
        solution = solve_static(
            model, data, contacts, limits, tip_force=tip[i], max_normal=cap, smooth=True
        )
        if solution is None:
            failures.append(int(i))
        else:
            peak = max(peak, solution.ratio)
            if solution.ratio > 1.0:
                failures.append(int(i))
    return {
        "changed_rows": len(changed),
        "friction_pyramid_mu": floor[0].mu,
        "failed_rows": failures,
        "peak_torque_ratio": peak,
        "verified": not failures,
    }


def check(walk_path: Path, nominal_path: Path = BASELINE, *, motor_ticks: bool = False) -> dict:
    geometry, _, study, _, ends, anchor = route(50.0, 0.19)
    cfg = study["candidates"][4]
    m, d = stair_valve.build_grasp_scene(
        geometry["slope_deg"],
        geometry["rise_m"],
        cfg["centre_m"],
        cfg["start_deg"],
        cfg["sense"],
        cfg["handle_length_m"],
        cfg["resistance_nm"],
    )
    toes = np.array([m.geom(leg).id for leg in LEGS])
    toe_set = set(toes)
    lever = m.geom(stair_valve.VALVE_LEVER).id
    terrain = set(world_geoms(m)) - {lever}
    root = m.body_rootid[m.body("base").id]
    robot = [
        i
        for i in range(m.ngeom)
        if m.body_rootid[m.geom_bodyid[i]] == root
        and (
            m.geom_conaffinity[i] & m.geom_contype[lever]
            or m.geom_contype[i] & m.geom_conaffinity[lever]
        )
    ]
    fingers = [m.joint(j).qposadr[0] for j in stair_valve.GRIPPER_JOINTS]
    hinge = m.joint(stair_valve.VALVE_HINGE).qposadr[0]
    motor_positions = [m.joint(name).qposadr[0] for name in joint_names()]
    with (
        np.load(walk_path, allow_pickle=False) as saved,
        np.load(nominal_path, allow_pickle=False) as nominal,
    ):
        times, parent_times = saved["time"], saved["reference_time"]
        poses, foot_targets = saved["qpos"], saved["feet"]
        parent_poses, source_feet = nominal["qpos"], nominal["feet"]
        ends_new, profile = scheduled_landings(
            saved, ends, anchor, geometry["slope_deg"], geometry["rise_m"]
        )
        if profile is None:
            raise ValueError("Walking plan does not contain the complete nominal route")
        grid = nominal["time"]
        left, right, weight, knot = reference_samples(parent_times, grid)
        check_tip_forces(saved["tip"], nominal["tip"], left, right, weight)
        if poses.shape != (len(times), parent_poses.shape[1]) or not np.all(np.isfinite(poses)):
            raise ValueError("Walking pose shape or values differ from the robot contract")
        if not np.array_equal(saved["target"], poses[:, motor_positions]):
            raise ValueError("Recorded motor targets differ from the prescribed robot poses")
        expected_stance = nominal["stance"][left]
        expected_stance[~knot] &= nominal["stance"][right[~knot]]
        if not np.array_equal(expected_stance, saved["stance"]):
            raise ValueError("Reference changes the scheduled support legs")
        if not np.array_equal(nominal["wall"][left], saved["wall"]):
            raise ValueError("Reference changes the scheduled support surfaces")
        cap_lo, cap_hi = nominal["cap"][left], nominal["cap"][right]
        finite = np.isfinite(cap_lo) & np.isfinite(cap_hi)
        cap = cap_lo.copy()
        row, leg = np.nonzero(finite)
        cap[row, leg] += weight[row] * (cap_hi[row, leg] - cap_lo[row, leg])
        if not np.allclose(cap, saved["cap"], atol=1e-8, rtol=0):
            raise ValueError("Reference changes the support load transfer")
        prefix = parent_times <= ends[anchor][0] + 1e-8
        moving = grid > ends[anchor][0] + 1e-8
        source_feet[moving, :, 1] += corridor_lateral_offsets(source_feet[moving, :, 0])
        planned_feet = source_feet[left] + weight[:, None, None] * (
            source_feet[right] - source_feet[left]
        )
        swing_peak = 0.0
        if "swing_offset_m" in saved:
            offset = saved["swing_offset_m"]
            if (
                offset.shape != planned_feet.shape
                or not np.all(np.isfinite(offset))
                or np.any(offset[expected_stance] != 0)
                or np.any(offset[prefix] != 0)
            ):
                raise ValueError(
                    "A swing offset changes a supporting foot or the original approach"
                )
            planned_feet += offset
            swing_peak = float(np.linalg.norm(offset, axis=2).max())
        if not np.allclose(planned_feet, foot_targets, atol=1e-9, rtol=0):
            raise ValueError("Recorded foot targets differ from the fixed lateral placement rule")
        if not np.array_equal(poses[prefix], parent_poses[left[prefix]]):
            raise ValueError("The pre-manipulation walking poses changed")
        checked = ~knot | np.any(poses != parent_poses[left], axis=1)
        reference_rows = len(poses)
        if motor_ticks:
            # transition.execute interpolates the pose, then normalizes its quaternion.
            ticks = np.arange(round(times[-1] / 0.005) + 1, dtype=float) * 0.005
            i = np.searchsorted(times, ticks, side="right").clip(1, len(times) - 1) - 1
            w = ((ticks - times[i]) / (times[i + 1] - times[i])).clip(0.0, 1.0)
            poses = (1 - w[:, None]) * poses[i] + w[:, None] * poses[i + 1]
            poses[:, 3:7] /= np.linalg.norm(poses[:, 3:7], axis=1, keepdims=True)
            planned_feet = (1 - w[:, None, None]) * planned_feet[i] + w[
                :, None, None
            ] * planned_feet[i + 1]
            parent_times = np.interp(ticks, times, parent_times)
            prefix = parent_times <= ends[anchor][0] + 1e-8
            times = ticks
        n = len(poses)
        errors = np.empty(n)
        obstacles = np.zeros(n)
        gaps = np.empty(n)
        for i, pose in enumerate(poses):
            d.qpos[: len(pose)] = pose
            d.qpos[fingers] = grasp_motion.OPEN_FINGER_M
            d.qpos[hinge] = 0.0 if prefix[i] else np.pi / 2
            mujoco.mj_fwdPosition(m, d)
            errors[i] = np.linalg.norm(d.geom_xpos[toes] - planned_feet[i], axis=1).max()
            gaps[i] = min(mujoco.mj_geomDistance(m, d, lever, geom, 0.08, None) for geom in robot)
            for c in d.contact:
                a, b = map(int, c.geom)
                if (a in toe_set and b in terrain) or (b in toe_set and a in terrain):
                    continue
                obstacles[i] = max(obstacles[i], -float(c.dist))
        report = {
            "scope": "kinematic walking samples before and after manipulation",
            "walking_sha256": file_sha256(walk_path),
            "parent_walk_sha256": file_sha256(nominal_path),
            "footprint_profile": profile,
            "samples": n,
            "inserted_samples": int(np.count_nonzero(~knot)),
            "max_swing_offset_m": swing_peak,
            "changed_pose_samples": int(checked.sum()),
            "scheduled_landings": len(ends_new),
            "walking_duration_s": float(times[-1]),
            "task_anchor_s": float(ends_new[anchor][0]),
            "max_foot_ik_error_m": float(errors.max()),
            "max_unexpected_penetration_m": float(obstacles.max()),
            "min_closed_valve_clearance_m": float(gaps[~prefix].min()),
            "nominal_time_at_closest_closed_valve_s": float(
                parent_times[~prefix][gaps[~prefix].argmin()]
            ),
            "ik_limit_m": transition.IK_TOLERANCE_M,
            "penetration_limit_m": 0.0001,
            "verified": bool(
                np.all(errors <= transition.IK_TOLERANCE_M)
                and np.all(obstacles <= 0.0001)
                and np.all(gaps[~prefix] >= 0.0)
            ),
        }
        if motor_ticks:
            violation = (
                (errors > transition.IK_TOLERANCE_M) | (obstacles > 0.0001) | (~prefix & (gaps < 0))
            )
            failed = np.flatnonzero(violation)
            report.update(
                scope=(
                    "walking reference interpolated at every 5 ms motor tick "
                    "including both endpoints"
                ),
                reference_pose_count=reference_rows,
                rejected_ticks=int(len(failed)),
                first_rejected_walk_time_s=float(times[failed[0]]) if len(failed) else None,
            )
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("walk", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--nominal", type=Path, default=BASELINE)
    parser.add_argument("--check-statics", action="store_true")
    parser.add_argument(
        "--motor-ticks",
        action="store_true",
        help="Check the commanded interpolation at every 5 ms motor tick",
    )
    args = parser.parse_args()
    result = check(args.walk, args.nominal, motor_ticks=args.motor_ticks)
    if args.check_statics:
        result["statics"] = check_statics(args.walk, args.nominal)
        result["verified"] = result["verified"] and result["statics"]["verified"]
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(result["verified"], result["samples"], result["min_closed_valve_clearance_m"])
    if not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
