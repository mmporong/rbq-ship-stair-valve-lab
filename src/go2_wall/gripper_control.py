"""Drive a named slide motor and audit the simulation boundaries used by a grasp replay."""

import mujoco
import numpy as np


def measured_arm_target(command, time_s, model, data, nominal, drive=None):
    """Evaluate feedback on the measured state and validate its six joint targets."""
    before = drive.snapshot(data) if drive else None
    target = np.asarray(command(time_s, model, data, nominal.copy()), dtype=float)
    if drive:
        drive.compare(data, before)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError("Arm feedback must return six finite joint targets in radians")
    return target


class FingerDrive:
    def __init__(self, model, data, name, command):
        actuator = model.actuator(name)
        if actuator.trntype[0] != mujoco.mjtTrn.mjTRN_JOINT or not actuator.ctrllimited[0]:
            raise ValueError(f"{name} must be a control-limited joint motor")
        joint = model.joint(int(actuator.trnid[0]))
        if joint.type[0] != mujoco.mjtJoint.mjJNT_SLIDE:
            raise ValueError(f"{name} must drive a slide joint")
        self.actuator = actuator.id
        self.position = joint.qposadr[0]
        self.velocity = joint.dofadr[0]
        self.low, self.high = actuator.ctrlrange
        self.command = command
        self.requested = 0.0
        self.start_time = data.time
        self.clipped = 0
        self.steps = 0
        self.max_external_generalized = 0.0
        self.max_external_body = 0.0
        self.state_writes = dict.fromkeys(("qpos", "qvel", "act", "mocap_pos", "mocap_quat"), 0)

    def snapshot(self, data):
        return tuple(getattr(data, name).copy() for name in self.state_writes)

    def compare(self, data, before):
        for name, original in zip(self.state_writes, before, strict=True):
            if not np.array_equal(getattr(data, name), original):
                self.state_writes[name] += 1
                raise ValueError(f"Task code wrote the live physics field {name} at {data.time}s")

    def apply(self, data, time_s, timestep, *, update_command=True):
        if not np.isclose(data.time, self.start_time + self.steps * timestep, rtol=0, atol=1e-7):
            raise ValueError(f"Physics clock discontinuity before step {self.steps}")
        if update_command:
            before = self.snapshot(data)
            self.requested = self.command(
                time_s, float(data.qpos[self.position]), float(data.qvel[self.velocity])
            )
            self.compare(data, before)
            if not np.isfinite(self.requested):
                raise ValueError("Nonfinite finger command")
        requested = self.requested
        self.clipped += int(requested < self.low or requested > self.high)
        data.ctrl[self.actuator] = np.clip(requested, self.low, self.high)
        self.max_external_generalized = max(
            self.max_external_generalized, float(np.max(np.abs(data.qfrc_applied), initial=0.0))
        )
        self.max_external_body = max(
            self.max_external_body, float(np.max(np.abs(data.xfrc_applied), initial=0.0))
        )
        self.steps += 1
        return requested

    def audit(self):
        return {
            "physics_steps": self.steps,
            "finger_clipped_steps": self.clipped,
            "qpos_write_count_after_start": self.state_writes["qpos"],
            "qvel_write_count_after_start": self.state_writes["qvel"],
            "act_write_count_after_start": self.state_writes["act"],
            "mocap_write_count_after_start": (
                self.state_writes["mocap_pos"] + self.state_writes["mocap_quat"]
            ),
            "state_reset_count": 0,
            "max_qfrc_applied_n": self.max_external_generalized,
            "max_xfrc_applied_n": self.max_external_body,
        }
