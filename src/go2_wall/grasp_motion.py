"""Arm and finger trajectory for contact-only operation of the stair valve."""

from collections.abc import Sequence

import mujoco
import numpy as np

from . import transition
from .robot import ARM_JOINTS, LEGS
from .statics import joint_names, solve_static, torque_limits

PAD_CENTRE_BEHIND_TIP_M = 0.035
PREGRASP_CLEARANCE_M = 0.08
RELEASE_CLEARANCE_M = 0.08
OPEN_FINGER_M = 0.04
CLOSED_FINGER_M = 0.0
FINGER_POSITION_GAIN = 5000.0
FINGER_VELOCITY_GAIN = 50.0
CLOSE_FORCE_FRACTION = 0.98
ARM_CORRECTION_LIMIT_RAD = 0.15
ARM_TARGET_STEP_LIMIT_RAD = 0.01
ARM_CORRECTION_DAMPING = 2e-3
OPERATION_ENTRY_HOLD_S = 0.25

DEFAULT_DURATIONS_S = {
    "approach": 3.0,
    "contact": 2.0,
    "settle": 2.0,
    "hold": 5.0,
    "release": 2.0,
    "retract": 6.0,
}


def handle_frame(saved: dict, angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return handle point, closing tangent, and radial direction in world coordinates."""
    angle = np.radians(saved["start_deg"] + saved["sense"] * angle_deg)
    radial = np.array([np.cos(angle), 0.0, np.sin(angle)])
    tangent = saved["sense"] * np.array([-radial[2], 0.0, radial[0]])
    point = np.asarray(saved["centre_m"], dtype=float) + saved["handle_length_m"] * radial
    return point, tangent, radial


def grasp_site_target(
    saved: dict, angle_deg: float, jaw_load_bias_m: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Place the pad centre on the handle with the jaw axis along the valve tangent.

    The gripper's long axis points across the stair width. This keeps the wrist beside the
    short lever instead of passing the wrist through the valve hub.
    """
    point, tangent, radial = handle_frame(saved, angle_deg)
    gripper_tangent = saved.get("gripper_tangent_sign", 1) * tangent
    gripper_axis = np.array([0.0, -1.0, 0.0])
    lateral = np.cross(gripper_axis, gripper_tangent)
    rotation = np.column_stack((gripper_tangent, lateral, gripper_axis))
    tip = point + PAD_CENTRE_BEHIND_TIP_M * gripper_axis + jaw_load_bias_m * tangent
    return tip, rotation


def clear_site_target(
    saved: dict,
    angle_deg: float,
    clearance_m: float = PREGRASP_CLEARANCE_M,
    jaw_load_bias_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an open-jaw pose clear of the valve plane across the stair width."""
    tip, rotation = grasp_site_target(saved, angle_deg, jaw_load_bias_m)
    return tip + np.array([0.0, clearance_m, 0.0]), rotation


def pregrasp_site_target(
    saved: dict, jaw_load_bias_m: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return the clear approach pose at the valve's initial angle."""
    return clear_site_target(saved, 0.0, jaw_load_bias_m=jaw_load_bias_m)


class GraspArmTargetCommand:
    """Read-only Cartesian feedback that returns corrected arm position targets."""

    def __init__(
        self,
        saved: dict,
        boundaries: dict[str, float],
        target_deg: float,
        motion_s: float,
        hold_target_deg: float | None = None,
        jaw_load_bias_m: float = 0.0,
    ) -> None:
        self.saved = saved
        self.boundaries = boundaries
        self.target_deg = target_deg
        self.hold_target_deg = target_deg if hold_target_deg is None else hold_target_deg
        self.motion_s = motion_s
        self.jaw_load_bias_m = jaw_load_bias_m
        self.previous: np.ndarray | None = None
        self.samples = 0
        self.correction_limited_steps = 0
        self.slew_limited_steps = 0
        self.max_position_error_m = 0.0
        self.max_rotation_error_rad = 0.0
        self.max_raw_correction_rad = 0.0
        self.last_raw_target_rad: np.ndarray | None = None
        self.last_applied_target_rad: np.ndarray | None = None
        self.raw_target_history_rad: list[np.ndarray] = []
        self.applied_target_history_rad: list[np.ndarray] = []

    def _target(self, time_s: float) -> tuple[np.ndarray, np.ndarray] | None:
        operation = self.boundaries["operation_name"]
        stage = stage_at(time_s, self.boundaries, operation)
        if stage == "approach":
            return None
        if stage == "contact":
            start = self.boundaries["approach"]
            align_s = self.boundaries["grip_start"] - start
            fraction = transition.smooth((time_s - start) / align_s)
            pregrasp_position, rotation = pregrasp_site_target(
                self.saved, self.jaw_load_bias_m
            )
            grasp_position, _ = grasp_site_target(self.saved, 0.0, self.jaw_load_bias_m)
            return (1.0 - fraction) * pregrasp_position + fraction * grasp_position, rotation
        if stage == operation:
            start = self.boundaries["operate_motion_start"]
            fraction = transition.smooth((time_s - start) / self.motion_s)
            return grasp_site_target(
                self.saved, self.target_deg * fraction, self.jaw_load_bias_m
            )
        if stage == "settle":
            start = self.boundaries["operate"]
            duration = self.boundaries["settle"] - start
            fraction = transition.smooth((time_s - start) / duration)
            angle_deg = (1.0 - fraction) * self.target_deg + fraction * self.hold_target_deg
            return grasp_site_target(self.saved, angle_deg, self.jaw_load_bias_m)
        if stage == "hold":
            return grasp_site_target(
                self.saved, self.hold_target_deg, self.jaw_load_bias_m
            )
        if stage == "release":
            start = self.boundaries["hold"]
            grasp_position, rotation = grasp_site_target(
                self.saved, self.hold_target_deg, self.jaw_load_bias_m
            )
            clear_position, _ = clear_site_target(
                self.saved,
                self.hold_target_deg,
                RELEASE_CLEARANCE_M,
                self.jaw_load_bias_m,
            )
            if self.boundaries.get("release_open_before_clear"):
                start = self.boundaries["release_clear_start"]
                duration = self.boundaries["release_clear_end"] - start
            else:
                duration = self.boundaries["release_clear_end"] - start
            fraction = transition.smooth((time_s - start) / duration)
            return (1.0 - fraction) * grasp_position + fraction * clear_position, rotation
        return None

    def __call__(
        self,
        time_s: float,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        nominal_arm_target_rad: np.ndarray,
    ) -> np.ndarray:
        nominal = np.asarray(nominal_arm_target_rad, dtype=float)
        desired = self._target(time_s)
        if desired is None:
            self.previous = nominal.copy()
            self.last_raw_target_rad = nominal.copy()
            self.last_applied_target_rad = nominal.copy()
            self.raw_target_history_rad.append(nominal.copy())
            self.applied_target_history_rad.append(nominal.copy())
            return nominal.copy()

        target_position, target_rotation = desired
        site = model.site("arm_tip").id
        actual_rotation = data.site_xmat[site].reshape(3, 3)
        position_error = target_position - data.site_xpos[site]
        rotation_error = 0.5 * sum(
            np.cross(actual_rotation[:, axis], target_rotation[:, axis]) for axis in range(3)
        )
        joints = [model.joint(name) for name in ARM_JOINTS]
        dofs = np.array([joint.dofadr[0] for joint in joints])
        jacobian_position = np.zeros((3, model.nv))
        jacobian_rotation = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, site)
        jacobian = np.vstack(
            (jacobian_position[:, dofs], jacobian_rotation[:, dofs])
        )
        error = np.r_[position_error, rotation_error]
        correction = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + ARM_CORRECTION_DAMPING * np.eye(6), error
        )
        raw_correction = correction.copy()
        raw = nominal + raw_correction
        correction = np.clip(
            correction, -ARM_CORRECTION_LIMIT_RAD, ARM_CORRECTION_LIMIT_RAD
        )
        self.correction_limited_steps += int(not np.array_equal(correction, raw_correction))
        limited = nominal + correction
        joint_low = np.array([joint.range[0] for joint in joints])
        joint_high = np.array([joint.range[1] for joint in joints])
        limited_to_range = np.clip(limited, joint_low, joint_high)
        self.correction_limited_steps += int(
            not np.array_equal(limited_to_range, limited)
        )
        previous = nominal if self.previous is None else self.previous
        applied = previous + np.clip(
            limited_to_range - previous,
            -ARM_TARGET_STEP_LIMIT_RAD,
            ARM_TARGET_STEP_LIMIT_RAD,
        )
        self.slew_limited_steps += int(not np.array_equal(applied, limited_to_range))
        self.samples += 1
        self.max_position_error_m = max(
            self.max_position_error_m, float(np.linalg.norm(position_error))
        )
        self.max_rotation_error_rad = max(
            self.max_rotation_error_rad, float(np.linalg.norm(rotation_error))
        )
        self.max_raw_correction_rad = max(
            self.max_raw_correction_rad, float(np.max(np.abs(raw_correction)))
        )
        self.previous = applied.copy()
        self.last_raw_target_rad = raw.copy()
        self.last_applied_target_rad = applied.copy()
        self.raw_target_history_rad.append(raw.copy())
        self.applied_target_history_rad.append(applied.copy())
        return applied

    def details(self) -> dict[str, float | int]:
        """Return raw-versus-applied correction diagnostics for trace metadata."""
        return {
            "samples": self.samples,
            "correction_limited_steps": self.correction_limited_steps,
            "slew_limited_steps": self.slew_limited_steps,
            "max_position_error_m": self.max_position_error_m,
            "max_rotation_error_rad": self.max_rotation_error_rad,
            "max_raw_correction_rad": self.max_raw_correction_rad,
            "correction_limit_rad": ARM_CORRECTION_LIMIT_RAD,
            "target_step_limit_rad": ARM_TARGET_STEP_LIMIT_RAD,
        }


def make_arm_target_command(
    saved: dict,
    boundaries: dict[str, float],
    target_deg: float,
    motion_s: float,
    hold_target_deg: float | None = None,
    jaw_load_bias_m: float = 0.0,
) -> GraspArmTargetCommand:
    """Create the arm-target callback accepted by ``transition.execute``."""
    return GraspArmTargetCommand(
        saved, boundaries, target_deg, motion_s, hold_target_deg, jaw_load_bias_m
    )


def solve_site_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_position: Sequence[float],
    target_rotation: np.ndarray,
    *,
    iterations: int = 160,
) -> tuple[np.ndarray, float, float]:
    """Solve all six RB1 joints for one position-and-orientation target."""
    joints = [model.joint(name) for name in ARM_JOINTS]
    qpos = np.array([joint.qposadr[0] for joint in joints])
    dofs = np.array([joint.dofadr[0] for joint in joints])
    site = model.site("arm_tip").id
    jacobian_position = np.zeros((3, model.nv))
    jacobian_rotation = np.zeros((3, model.nv))
    target_position = np.asarray(target_position, dtype=float)
    target_rotation = np.asarray(target_rotation, dtype=float)

    for _ in range(iterations):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        current_rotation = data.site_xmat[site].reshape(3, 3)
        position_error = target_position - data.site_xpos[site]
        rotation_error = 0.5 * sum(
            np.cross(current_rotation[:, axis], target_rotation[:, axis]) for axis in range(3)
        )
        if np.linalg.norm(position_error) < 3e-4 and np.linalg.norm(rotation_error) < 3e-3:
            break
        mujoco.mj_jacSite(
            model, data, jacobian_position, jacobian_rotation, site
        )
        jacobian = np.vstack(
            (jacobian_position[:, dofs], jacobian_rotation[:, dofs])
        )
        error = np.r_[position_error, rotation_error]
        update = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 2e-3 * np.eye(6), error
        )
        for joint, delta in zip(joints, update, strict=True):
            address = joint.qposadr[0]
            data.qpos[address] = np.clip(
                data.qpos[address] + np.clip(delta, -0.08, 0.08), *joint.range
            )

    mujoco.mj_kinematics(model, data)
    current_rotation = data.site_xmat[site].reshape(3, 3)
    position_error = float(np.linalg.norm(target_position - data.site_xpos[site]))
    rotation_error = float(
        np.linalg.norm(
            0.5
            * sum(
                np.cross(current_rotation[:, axis], target_rotation[:, axis])
                for axis in range(3)
            )
        )
    )
    return data.qpos[qpos].copy(), position_error, rotation_error


def solve_grasp_arc(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    saved: dict,
    angles_deg: Sequence[float],
    jaw_load_bias_m: float = 0.0,
) -> dict:
    """Solve a continuous contact pose sequence from the current arm seed."""
    targets = []
    position_errors = []
    rotation_errors = []
    for angle in angles_deg:
        position, rotation = grasp_site_target(saved, float(angle), jaw_load_bias_m)
        target, position_error, rotation_error = solve_site_pose(
            model, data, position, rotation
        )
        targets.append(target)
        position_errors.append(position_error)
        rotation_errors.append(rotation_error)
    return {
        "angles_deg": np.asarray(angles_deg, dtype=float),
        "arm_joint_targets_rad": np.asarray(targets),
        "position_error_m": np.asarray(position_errors),
        "rotation_error_rad": np.asarray(rotation_errors),
    }


def stage_at(time_s: float, boundaries: dict[str, float], operation_name: str) -> str:
    """Return the canonical stage label used by traces and the integrated mission."""
    if time_s < boundaries["approach"]:
        return "approach"
    if time_s < boundaries["contact"]:
        return "contact"
    if time_s < boundaries["operate"]:
        return operation_name
    if time_s < boundaries["settle"]:
        return "settle"
    if time_s < boundaries["hold"]:
        return "hold"
    if time_s < boundaries["release"]:
        return "release"
    return "retract"


def finger_command(
    time_s: float,
    finger_position_m: float,
    finger_velocity_mps: float,
    *,
    boundaries: dict[str, float],
    force_limit_n: float,
) -> float:
    """Open for approach/release and otherwise close the coupled RB1 fingers."""
    stage = stage_at(time_s, boundaries, "operate")
    aligning = stage == "contact" and time_s < boundaries["grip_start"]
    opening_release = stage == "release" and boundaries.get("release_open_before_clear")
    if opening_release and time_s < boundaries["release_force_end"]:
        progress = transition.smooth(
            (time_s - boundaries["hold"])
            / (boundaries["release_force_end"] - boundaries["hold"])
        )
        return -(1.0 - progress) * CLOSE_FORCE_FRACTION * force_limit_n
    if opening_release and time_s < boundaries["release_clear_end"]:
        return 0.0
    if opening_release:
        target = OPEN_FINGER_M
    elif stage == "release" and time_s >= boundaries["release_clear_end"]:
        release_s = boundaries["release"] - boundaries["release_clear_end"]
        progress = transition.smooth(
            (time_s - boundaries["release_clear_end"]) / release_s
        )
        target = OPEN_FINGER_M * progress
    elif stage in ("approach", "retract") or aligning:
        target = OPEN_FINGER_M
    else:
        return -CLOSE_FORCE_FRACTION * force_limit_n
    raw = (
        FINGER_POSITION_GAIN * (target - finger_position_m)
        - FINGER_VELOCITY_GAIN * finger_velocity_mps
    )
    return float(force_limit_n * np.tanh(raw / force_limit_n))


def open_finger_command(
    finger_position_m: float,
    finger_velocity_mps: float,
    *,
    force_limit_n: float,
) -> float:
    """Hold the coupled fingers open for the open-hand falsification trial."""
    raw = (
        FINGER_POSITION_GAIN * (OPEN_FINGER_M - finger_position_m)
        - FINGER_VELOCITY_GAIN * finger_velocity_mps
    )
    return float(force_limit_n * np.tanh(raw / force_limit_n))


def _boundaries(target_deg: float, motion_s: float, durations: dict[str, float]) -> dict:
    operation_name = "low_load" if target_deg <= 20.0 + 1e-9 else "operate"
    clock = durations["approach"]
    boundaries = {"approach": clock}
    clock += durations["contact"]
    boundaries["grip_start"] = clock - 0.25 * durations["contact"]
    boundaries["operate_motion_start"] = clock + OPERATION_ENTRY_HOLD_S
    boundaries["contact"] = clock
    clock = boundaries["operate_motion_start"] + motion_s
    boundaries["operate"] = clock
    clock += durations["settle"]
    boundaries["settle"] = clock
    clock += durations["hold"]
    boundaries["hold"] = clock
    clock += durations["release"]
    boundaries["release"] = clock
    boundaries["release_clear_end"] = clock - 0.5 * durations["release"]
    clock += durations["retract"]
    boundaries["retract"] = clock
    boundaries["duration"] = clock
    boundaries["operation_name"] = operation_name
    return boundaries


def _sample_times(duration_s: float) -> np.ndarray:
    """Sample the 20 ms grid and retain an exact non-grid endpoint."""
    count = int(np.floor(duration_s / transition.SAMPLE_S + 1e-9))
    times = np.arange(count + 1) * transition.SAMPLE_S
    if duration_s - times[-1] > 1e-9:
        times = np.r_[times, duration_s]
    return times


def turn_load_fraction(time_s: float, stage: str, boundaries: dict) -> float:
    """Fraction of rated tangent load requested by the robot's inverse statics."""
    if stage not in (boundaries["operation_name"], "settle", "hold"):
        return 0.0
    if "unload_start" not in boundaries:
        return 1.0
    start, end = boundaries["unload_start"], boundaries["unload_end"]
    progress = transition.smooth((time_s - start) / (end - start))
    return 1.0 - (1.0 - boundaries["hold_load_fraction"]) * progress


def make_grasp_track(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    study: dict,
    saved: dict,
    *,
    target_deg: float,
    motion_s: float,
    resistance_nm: float,
    durations_s: dict[str, float] | None = None,
    jaw_load_bias_m: float = 0.0,
    terminal_overtravel_deg: float = 0.0,
) -> tuple[dict, dict]:
    """Build a fixed-stance approach, contact operation, release, and retract track."""
    durations = {**DEFAULT_DURATIONS_S, **(durations_s or {})}
    if saved.get("release_open_before_clear"):
        durations["release"] = max(durations["release"], saved.get("release_duration_s", 3.5))
    boundaries = _boundaries(target_deg, motion_s, durations)
    if "opening_load_relief" in saved:
        relief = saved["opening_load_relief"]
        delay, ramp, fraction = relief["delay_s"], relief["ramp_s"], relief["hold_fraction"]
        if (
            saved.get("operation") != "open"
            or not np.isfinite([delay, ramp, fraction]).all()
            or delay < 0
            or ramp <= 0
            or not 0.0 <= fraction <= 1.0
        ):
            raise ValueError("Invalid schedule for the opening controller's tangent load")
        boundaries["unload_start"] = boundaries["operate"] + delay
        boundaries["unload_end"] = boundaries["unload_start"] + ramp
        boundaries["hold_load_fraction"] = fraction
    grip_lead_s = float(saved.get("grip_lead_s", 0.0))
    if not np.isfinite(grip_lead_s) or not 0.0 <= grip_lead_s < durations["contact"]:
        raise ValueError("Grip lead must be finite and shorter than the contact stage")
    boundaries["grip_start"] -= grip_lead_s
    if saved.get("release_open_before_clear"):
        boundaries["release_open_before_clear"] = True
        boundaries["release_force_end"] = boundaries["hold"] + saved.get(
            "release_open_s", 0.75
        )
        boundaries["release_clear_start"] = boundaries["release_force_end"]
        boundaries["release_clear_end"] = boundaries["release"] - 0.5
    operation_name = boundaries["operation_name"]
    command_target_deg = 23.0 if target_deg <= 20.0 + 1e-9 else 91.4
    hold_target_deg = command_target_deg
    if not np.isfinite(terminal_overtravel_deg) or terminal_overtravel_deg < 0:
        raise ValueError("Terminal overtravel must be finite and nonnegative")
    if terminal_overtravel_deg and target_deg <= 20.0 + 1e-9:
        raise ValueError("Terminal overtravel is defined for the rated valve turn")
    command_target_deg += terminal_overtravel_deg
    feet = {leg: np.asarray(saved["feet_xyz_m"][leg], dtype=float) for leg in LEGS}
    stance = {leg: (feet[leg], "floor") for leg in LEGS}
    contacts = transition.contacts_for(stance)
    limits = torque_limits(study["arm"])
    names = joint_names()
    addresses = np.array([model.joint(name).qposadr[0] for name in names])
    arm_addresses = np.array([model.joint(name).qposadr[0] for name in ARM_JOINTS])
    finger = model.joint("RB1_finger_upper_joint")
    lower_finger = model.joint("RB1_finger_lower_joint")

    stow_source = study.get("arm_stow_rad", transition.tuck_arm())
    seed = {name: stow_source[name] for name in ARM_JOINTS}
    open_pose = model.qpos0.copy()
    open_pose[finger.qposadr[0]] = OPEN_FINGER_M
    open_pose[lower_finger.qposadr[0]] = OPEN_FINGER_M
    if not transition.place(
        model, data, saved["base_pose"], feet, seed, warm=open_pose
    ):
        raise ValueError("Saved valve stance has no initial pose")
    data.qpos[finger.qposadr[0]] = OPEN_FINGER_M
    mujoco.mj_forward(model, data)
    base_qpos = data.qpos.copy()
    stow = np.array([stow_source[name] for name in ARM_JOINTS])
    initial_arm_target = saved.get("initial_arm_target_rad")
    if initial_arm_target is None:
        initial_arm_target = saved["arm_joint_targets_rad"][0]
    data.qpos[arm_addresses] = initial_arm_target
    pregrasp_position, pregrasp_rotation = pregrasp_site_target(saved, jaw_load_bias_m)
    pregrasp, pregrasp_position_error, pregrasp_rotation_error = solve_site_pose(
        model, data, pregrasp_position, pregrasp_rotation
    )
    angles = np.linspace(
        0.0,
        command_target_deg,
        max(9, int(np.ceil(command_target_deg / 2.5)) + 1),
    )
    arc = solve_grasp_arc(model, data, saved, angles, jaw_load_bias_m)
    hold_arm = np.array(
        [
            np.interp(hold_target_deg, arc["angles_deg"], column)
            for column in arc["arm_joint_targets_rad"].T
        ]
    )
    clear_position, clear_rotation = clear_site_target(
        saved, hold_target_deg, RELEASE_CLEARANCE_M, jaw_load_bias_m
    )
    retreat, retreat_position_error, retreat_rotation_error = solve_site_pose(
        model, data, clear_position, clear_rotation
    )
    if arc["position_error_m"].max() > 0.002 or arc["rotation_error_rad"].max() > 0.02:
        raise ValueError(
            "Contact arc exceeds IK tolerance: "
            f"{arc['position_error_m'].max():.6f} m, "
            f"{arc['rotation_error_rad'].max():.6f} rad"
        )
    if pregrasp_position_error > 0.002 or pregrasp_rotation_error > 0.02:
        raise ValueError(
            "Pregrasp exceeds IK tolerance: "
            f"{pregrasp_position_error:.6f} m, {pregrasp_rotation_error:.6f} rad"
        )
    if retreat_position_error > 0.002 or retreat_rotation_error > 0.02:
        raise ValueError(
            "Retreat exceeds IK tolerance: "
            f"{retreat_position_error:.6f} m, {retreat_rotation_error:.6f} rad"
        )

    times = _sample_times(boundaries["duration"])
    rows = {
        "time": [],
        "qpos": [],
        "target": [],
        "torque": [],
        "tip": [],
        "stance": [],
        "wall": [],
        "cap": [],
        "feet": [],
        "stage": [],
    }
    previous = None
    for time_s in times:
        stage = stage_at(time_s, boundaries, operation_name)
        if stage == "approach":
            progress = transition.smooth(time_s / durations["approach"])
            arm = (1.0 - progress) * stow + progress * pregrasp
            desired_angle = 0.0
        elif stage == "contact":
            start = boundaries["approach"]
            align_s = boundaries["grip_start"] - start
            progress = transition.smooth((time_s - start) / align_s)
            arm = (1.0 - progress) * pregrasp + progress * arc["arm_joint_targets_rad"][0]
            desired_angle = 0.0
        elif stage == operation_name:
            start = boundaries["operate_motion_start"]
            progress = transition.smooth((time_s - start) / motion_s)
            desired_angle = command_target_deg * progress
            arm = np.array(
                [
                    np.interp(desired_angle, arc["angles_deg"], column)
                    for column in arc["arm_joint_targets_rad"].T
                ]
            )
        elif stage in ("settle", "hold"):
            if stage == "settle":
                start = boundaries["operate"]
                progress = transition.smooth(
                    (time_s - start) / durations["settle"]
                )
                arm = (1.0 - progress) * arc["arm_joint_targets_rad"][-1] + progress * hold_arm
                desired_angle = (
                    (1.0 - progress) * command_target_deg + progress * hold_target_deg
                )
            else:
                arm = hold_arm
                desired_angle = hold_target_deg
        elif stage == "release":
            if boundaries.get("release_open_before_clear"):
                start = boundaries["release_clear_start"]
                retreat_s = boundaries["release_clear_end"] - start
            else:
                start = boundaries["hold"]
                retreat_s = boundaries["release_clear_end"] - start
            progress = transition.smooth((time_s - start) / retreat_s)
            arm = (1.0 - progress) * hold_arm + progress * retreat
            desired_angle = hold_target_deg
        else:
            start = boundaries["release"]
            progress = transition.smooth((time_s - start) / durations["retract"])
            arm = (1.0 - progress) * retreat + progress * stow
            desired_angle = hold_target_deg

        data.qpos[:] = base_qpos if previous is None else previous
        data.qpos[arm_addresses] = arm
        data.qpos[finger.qposadr[0]] = OPEN_FINGER_M
        _, tangent, _ = handle_frame(saved, desired_angle)
        load_progress = turn_load_fraction(time_s, stage, boundaries)
        reaction = -load_progress * resistance_nm / saved["handle_length_m"] * tangent
        solution = solve_static(
            model, data, contacts, limits, tip_force=reaction, smooth=True
        )
        if solution is None:
            raise ValueError(f"Contact valve pose is infeasible in {stage} at {time_s:.2f}s")
        previous = data.qpos.copy()
        rows["time"].append(time_s)
        rows["qpos"].append(previous)
        rows["target"].append(data.qpos[addresses].copy())
        rows["torque"].append(solution.torque.copy())
        rows["tip"].append(reaction)
        rows["stance"].append([True] * len(LEGS))
        rows["wall"].append([False] * len(LEGS))
        rows["cap"].append([np.inf] * len(LEGS))
        rows["feet"].append([feet[leg] for leg in LEGS])
        rows["stage"].append(stage)

    track = {
        key: np.asarray(value) if key != "stage" else np.asarray(value, dtype=str)
        for key, value in rows.items()
    }
    track["phase"] = track["stage"].copy()
    metadata = {
        "boundaries_s": boundaries,
        "track_duration_s": float(times[-1]),
        "target_deg": target_deg,
        "command_target_deg": command_target_deg,
        "hold_target_deg": hold_target_deg,
        "resistance_nm": resistance_nm,
        "arc_max_position_error_m": float(arc["position_error_m"].max()),
        "arc_max_rotation_error_rad": float(arc["rotation_error_rad"].max()),
        "pregrasp_position_error_m": pregrasp_position_error,
        "pregrasp_rotation_error_rad": pregrasp_rotation_error,
        "pregrasp_clearance_m": PREGRASP_CLEARANCE_M,
        "release_clearance_m": RELEASE_CLEARANCE_M,
        "retreat_position_error_m": retreat_position_error,
        "retreat_rotation_error_rad": retreat_rotation_error,
        "pad_centre_behind_tip_m": PAD_CENTRE_BEHIND_TIP_M,
        "jaw_load_bias_m": jaw_load_bias_m,
        "closing_axis": "handle_tangent",
        "stage_alias": "phase is an exact alias of stage",
    }
    return track, metadata
