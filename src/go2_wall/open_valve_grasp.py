"""Convert a closing-valve candidate into the corresponding opening operation."""

from copy import deepcopy

import numpy as np

OPENING_BASE_OFFSET_M = np.array([0.08, 0.0, 0.06])


def opening_candidate(candidate: dict, *, travel_deg: float = 90.0) -> dict:
    """Start at the candidate's closed pose and rotate back to its open pose.

    MuJoCo's passive valve hinge always advances from zero toward ``travel_deg``.  The
    transformed candidate therefore moves the world-space initial angle to the original
    closed endpoint and reverses the hinge axis.  Stored closing-arc samples are removed
    because their angle meaning no longer matches the opening operation.  Only the
    interpolated closed-pose arm target is retained as an IK seed.
    """
    result = deepcopy(candidate)
    sense = result.get("sense")
    if sense not in (-1, 1):
        raise ValueError("Valve sense must be -1 or 1")
    if not np.isfinite(travel_deg) or travel_deg <= 0:
        raise ValueError("Valve travel must be finite and positive")

    start_deg = float(result["start_deg"])
    result["start_deg"] = (start_deg + sense * travel_deg) % 360.0
    result["sense"] = -sense
    result["operation"] = "open"
    result["gripper_tangent_sign"] = -1
    result["grip_lead_s"] = 0.4
    result["release_open_before_clear"] = True
    result["release_open_s"] = 0.5
    result["release_duration_s"] = 3.0
    result["world_open_angle_deg"] = start_deg % 360.0
    result["world_closed_angle_deg"] = result["start_deg"]
    result["travel_deg"] = float(travel_deg)
    base_pose = np.asarray(result["base_pose"], dtype=float)
    if base_pose.shape != (4,) or np.any(~np.isfinite(base_pose)):
        raise ValueError("Valve base pose must contain x, y, z, and pitch")
    base_pose[:3] += OPENING_BASE_OFFSET_M
    result["base_pose"] = base_pose.tolist()
    result["opening_base_offset_m"] = OPENING_BASE_OFFSET_M.tolist()

    angles = candidate.get("sample_angles_deg")
    targets = candidate.get("arm_joint_targets_rad")
    if angles is not None or targets is not None:
        angle_array = np.asarray(angles, dtype=float)
        target_array = np.asarray(targets, dtype=float)
        if (
            angle_array.ndim != 1
            or target_array.ndim != 2
            or target_array.shape[0] != angle_array.size
            or angle_array.size < 2
            or np.any(~np.isfinite(angle_array))
            or np.any(~np.isfinite(target_array))
            or np.any(np.diff(angle_array) <= 0)
            or travel_deg < angle_array[0]
            or travel_deg > angle_array[-1]
        ):
            raise ValueError("Closing arc cannot seed the requested closed pose")
        result["initial_arm_target_rad"] = [
            float(np.interp(travel_deg, angle_array, column)) for column in target_array.T
        ]
        result.pop("sample_angles_deg", None)
        result.pop("arm_joint_targets_rad", None)
        result.pop("peak_static_ratio", None)
    return result
