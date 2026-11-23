"""Plan the last two rear footsteps farther back on their working tread."""

import hashlib

import numpy as np

from go2_wall import stairs, transition
from go2_wall.statics import torque_limits


def rear_landing_stance(stance, installed_feet, rear_offset_m):
    moved = {}
    for leg, (point, surface) in stance.items():
        xyz = np.asarray(point, dtype=float)
        if leg in ("RL", "RR") and np.allclose(xyz, installed_feet[leg], rtol=0, atol=1e-8):
            xyz = xyz - [rear_offset_m, 0.0, 0.0]
        moved[leg] = (xyz, surface)
    return moved


def rear_foot_reference(geometry, segments, ends, anchor, installed_feet, offset):
    """Plan both rear swings and the approach that follows on the same stair treads."""
    stop = ends[anchor][0]
    clocks = np.r_[0.0, np.cumsum([segment["seconds"] for segment in segments])]
    end_index = int(np.argmin(np.abs(clocks - stop)))
    if abs(clocks[end_index] - stop) > 1e-7:
        raise ValueError("The work landing must coincide with a walking segment boundary")
    start_index = next(
        i
        for i, segment in enumerate(segments[:end_index])
        if segment["kind"] == "swing"
        and segment["leg"] in ("RL", "RR")
        and np.allclose(
            segment["after"][segment["leg"]][0], installed_feet[segment["leg"]], atol=1e-8, rtol=0
        )
    )
    start = clocks[start_index]
    tail = []
    for old in segments[start_index:end_index]:
        moved = old.copy()
        for key in ("stance", "before", "after"):
            if key in old:
                moved[key] = rear_landing_stance(old[key], installed_feet, offset)
        tail.append(moved)
    stairs.ARM = geometry["arm"]
    stairs.use_stow()
    model, data = stairs.build_stairs(geometry["slope_deg"], geometry["rise_m"])
    reference = transition.sample_segments(
        model, data, tail, torque_limits(geometry["arm"]), geometry["arm_stow_rad"]
    )
    reference["time"] = start + reference["time"]
    if abs(reference["time"][-1] - stop) > 1e-7:
        raise ValueError("Replanned rear landings changed the work-station arrival time")
    ledger = {
        "rear_offset_m": offset,
        "reference_start_s": float(start),
        "reference_end_s": float(stop),
        "tail_qpos_sha256": hashlib.sha256(
            np.ascontiguousarray(reference["qpos"]).tobytes()
        ).hexdigest(),
    }
    changed_ends = [
        (t, rear_landing_stance(stance, installed_feet, offset), leg) for t, stance, leg in ends
    ]
    return reference, changed_ends, ledger


def approach_with_rear_foot_offset(geometry, segments, ends, anchor, walk, installed_feet, offset):
    reference, changed_ends, ledger = rear_foot_reference(
        geometry, segments, ends, anchor, installed_feet, offset
    )
    prefix = walk["time"] < ledger["reference_start_s"] - 1e-7
    combined = {
        key: np.concatenate((values[prefix], reference[key])) for key, values in walk.items()
    }
    return combined, changed_ends, ledger
