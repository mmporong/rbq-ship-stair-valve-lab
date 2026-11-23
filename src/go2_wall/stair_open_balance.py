"""Torque-limited leg position feedback while operating the valve on the stair."""

import numpy as np

from .statics import joint_names


class OpeningStanceControl:
    """Brace the four loaded legs at their measured angles when the fingers engage."""

    NAME = "opening_stance_rear_foot_v2"
    REAR_OFFSET_M = 0.040

    @staticmethod
    def work_pose(pose):
        result = list(pose)
        result[1] -= 0.016
        result[2] -= 0.005
        return result

    def __init__(self, model, positioning_start_s, grip_start_s):
        self.positioning_start_s = positioning_start_s
        self.grip_start_s = grip_start_s
        self.base_position_scale = 2.0
        self.base_rotation_scale = 2.0
        self.kp_nm_per_rad = 100.0
        self.kd_nm_s_per_rad = 3.0
        joints = [model.joint(name) for name in joint_names()[:12]]
        self.positions = [joint.qposadr[0] for joint in joints]
        self.velocities = [joint.dofadr[0] for joint in joints]
        self.anchor = None
        self.peak_correction_nm = 0.0

    def base_feedback_scales(self, time_s):
        if time_s < self.positioning_start_s:
            return 1.0, 1.0
        return self.base_position_scale, self.base_rotation_scale

    def command(self, model, data, track, row, weight, stance, nominal_forces, now, original):
        if now < self.grip_start_s or not np.all(stance):
            return original, 0
        position = data.qpos[self.positions]
        if self.anchor is None:
            self.anchor = position.copy()
        correction = (
            self.kp_nm_per_rad * (self.anchor - position)
            - self.kd_nm_s_per_rad * data.qvel[self.velocities]
        )
        command = original.copy()
        command[:12] += correction
        self.peak_correction_nm = max(self.peak_correction_nm, float(np.abs(correction).max()))
        return command, 0

    def describe(self):
        return {
            "name": self.NAME,
            "positioning_start_s": self.positioning_start_s,
            "grip_start_s": self.grip_start_s,
            "base_position_scale": self.base_position_scale,
            "base_rotation_scale": self.base_rotation_scale,
            "leg_kp_nm_per_rad": self.kp_nm_per_rad,
            "leg_kd_nm_s_per_rad": self.kd_nm_s_per_rad,
            "anchor_captured": self.anchor is not None,
            "peak_leg_correction_nm": self.peak_correction_nm,
        }
