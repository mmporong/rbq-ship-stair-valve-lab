"""RBQ-10 + RB1 operating a quarter-turn valve on a ship stair.

The site-site equality represents an ideal grip and transfers reaction forces between
the arm tip and handle in the original scene. The grasp scene retains the vendor's coupled
finger joints and transfers force through the two finger-pad contacts instead.
"""

from collections.abc import Callable, Sequence

import mujoco
import numpy as np

from .robot import build_robot
from .stairs import add_ship_stair

TREADS = 10
FLOOR_MU = 0.6
VALVE_HINGE = "valve_hinge"
VALVE_HANDLE = "valve_handle"
VALVE_PIN = "valve_pin"
VALVE_LEVER = "valve_lever"
GRIPPER_ACTUATOR = "RB1_finger_upper"
GRIPPER_JOINTS = ("RB1_finger_upper_joint", "RB1_finger_lower_joint")
GRIPPER_PADS = ("RB1_finger_upper_collision", "RB1_finger_lower_collision")
FRICTION_SOLREF = (0.0, -2000.0)
FRICTION_SOLIMP = (0.999, 0.999, 0.001, 0.5, 2.0)
LIMIT_SOLREF = (0.01, 1.0)
LIMIT_SOLIMP = (0.99, 0.999, 0.0015, 0.5, 2.0)
OBSERVATION_FIELDS = (
    "valve_angle_rad",
    "valve_angular_velocity_rad_s",
    "valve_friction_torque_nm",
    "pin_site_distance_m",
    "pin_hinge_torque_nm",
)

GRASP_OBSERVATION_FIELDS = (
    "valve_angle_deg",
    "valve_speed_rad_s",
    "hinge_friction_torque_nm",
    "signed_hinge_friction_torque_nm",
    "pad_normal_force_n",
    "pad_closing_torque_nm",
    "pad_closing_tangent_force_n",
    "pad_surface_gap_m",
    "handle_axis_offset_m",
    "handle_axial_position_m",
    "max_unexpected_lever_contact_n",
)


def _validate_valve_inputs(
    centre_m: Sequence[float], sense: int, handle_length_m: float, resistance_nm: float
) -> np.ndarray:
    centre = np.asarray(centre_m, dtype=float)
    if centre.shape != (3,):
        raise ValueError("centre_m must contain exactly three coordinates")
    if sense not in (-1, 1):
        raise ValueError("sense must be -1 or 1")
    if handle_length_m <= 0:
        raise ValueError("handle_length_m must be positive")
    if resistance_nm < 0:
        raise ValueError("resistance_nm must be non-negative")
    return centre


def _add_valve(
    spec: mujoco.MjSpec,
    centre: np.ndarray,
    start_deg: float,
    sense: int,
    handle_length_m: float,
    resistance_nm: float,
):
    angle = np.radians(start_deg)
    tip = handle_length_m * np.array([np.cos(angle), 0.0, np.sin(angle)])
    valve = spec.worldbody.add_body(name="valve", pos=centre.tolist())
    # The counterbalanced handle isolates the specified valve resistance from handle weight.
    valve.gravcomp = 1.0
    hinge = valve.add_joint(name=VALVE_HINGE, type=mujoco.mjtJoint.mjJNT_HINGE)
    hinge.axis = [0.0, float(-sense), 0.0]
    hinge.range = [0.0, np.pi / 2]
    hinge.limited = True
    hinge.frictionloss = resistance_nm
    hinge.solref_friction = list(FRICTION_SOLREF)
    hinge.solimp_friction = list(FRICTION_SOLIMP)
    hinge.solref_limit = list(LIMIT_SOLREF)
    hinge.solimp_limit = list(LIMIT_SOLIMP)
    hinge.armature = 0.001

    handle = valve.add_geom(
        name=VALVE_LEVER,
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0.0, 0.0, 0.0, *tip.tolist()],
        size=[0.01, 0.0, 0.0],
        rgba=[0.85, 0.2, 0.15, 1.0],
    )
    handle.mass = 0.1
    valve.add_site(name=VALVE_HANDLE, pos=tip.tolist(), size=[0.012, 0.012, 0.012])
    return handle


def build_scene(
    slope_deg: float,
    rise_m: float,
    centre_m: Sequence[float],
    start_deg: float,
    sense: int,
    handle_length_m: float,
    resistance_nm: float,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build the 10-tread stair, RBQ-10 + RB1, and a passive quarter-turn valve.

    Positive hinge position advances the handle from ``start_deg`` in ``sense``. The
    ``valve_pin`` connect equality represents an ideal ball-and-socket grip between
    ``arm_tip`` and ``valve_handle``. The replay starts with the two sites joined.
    """
    centre = _validate_valve_inputs(centre_m, sense, handle_length_m, resistance_nm)

    spec = build_robot(case="scaled", wall_x=None, floor_mu=FLOOR_MU, arm="rb1")
    add_ship_stair(spec.worldbody, slope_deg, rise_m, TREADS, FLOOR_MU)
    handle = _add_valve(spec, centre, start_deg, sense, handle_length_m, resistance_nm)
    handle.contype = 0
    handle.conaffinity = 0

    spec.add_equality(
        name=VALVE_PIN,
        type=mujoco.mjtEq.mjEQ_CONNECT,
        name1="arm_tip",
        name2=VALVE_HANDLE,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
    )
    model = spec.compile()
    return model, mujoco.MjData(model)


def build_grasp_scene(
    slope_deg: float,
    rise_m: float,
    centre_m: Sequence[float],
    start_deg: float,
    sense: int,
    handle_length_m: float,
    resistance_nm: float,
    *,
    floor_mu: float = FLOOR_MU,
    pad_mu: float = 0.8,
    finger_force_n: float = 100.0,
    slide_loss_n: float = 30.0,
    decorate: Callable[[mujoco.MjSpec], None] | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build a passive valve operated only through the coupled RB1 finger pads.

    The first 18 actuators retain the leg/arm order used by the controller. The named finger
    actuator is appended at index 18 and commands force in newtons along the upper slide joint.
    Negative control closes the coupled fingers. No equality connects the robot to the valve.
    """
    centre = _validate_valve_inputs(centre_m, sense, handle_length_m, resistance_nm)
    if floor_mu <= 0 or pad_mu <= 0 or finger_force_n <= 0 or slide_loss_n < 0:
        raise ValueError(
            "friction and finger force must be positive; slide loss cannot be negative"
        )

    spec = build_robot(case="scaled", wall_x=None, floor_mu=floor_mu, arm="rb1")
    add_ship_stair(spec.worldbody, slope_deg, rise_m, TREADS, floor_mu)
    for joint_name in GRIPPER_JOINTS:
        spec.joint(joint_name).frictionloss = slide_loss_n
    finger = spec.add_actuator(
        name=GRIPPER_ACTUATOR,
        target=GRIPPER_JOINTS[0],
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
    )
    finger.ctrlrange = [-finger_force_n, finger_force_n]
    finger.ctrllimited = True

    for geom_name in GRIPPER_PADS:
        pad = spec.geom(geom_name)
        pad.contype = 1
        pad.conaffinity = 1
        pad.condim = 4
        pad.friction = [pad_mu, 0.02, 0.005]
    handle = _add_valve(spec, centre, start_deg, sense, handle_length_m, resistance_nm)
    handle.contype = 1
    handle.conaffinity = 1
    handle.condim = 4
    handle.friction = [pad_mu, 0.02, 0.005]

    if decorate is not None:
        # Render-only additions (textures, materials, lights); must not add bodies or geoms.
        decorate(spec)
    model = spec.compile()
    return model, mujoco.MjData(model)


def _constraint_torque(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    constraint_type: mujoco.mjtConstraint,
    constraint_id: int,
    dof: int,
) -> float:
    """Return one DOF's generalized force from selected scalar constraint rows."""
    if data.nefc == 0:
        return 0.0
    selected = (data.efc_type[: data.nefc] == constraint_type) & (
        data.efc_id[: data.nefc] == constraint_id
    )
    if not np.any(selected):
        return 0.0
    constraint_force = np.zeros(data.nefc)
    constraint_force[selected] = data.efc_force[: data.nefc][selected]
    generalized_force = np.zeros(model.nv)
    mujoco.mj_mulJacTVec(model, data, generalized_force, constraint_force)
    return float(generalized_force[dof])


def _grasp_contacts(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[np.ndarray, float, float, dict[str, float]]:
    """Measure pad normal forces and signed closing torque from actual contacts."""
    lever = model.geom(VALVE_LEVER).id
    pads = [model.geom(name).id for name in GRIPPER_PADS]
    normals = np.zeros(len(pads))
    closing_torque = 0.0
    tangent_force = 0.0
    unexpected: dict[str, float] = {}
    hinge = model.joint(VALVE_HINGE)
    hinge_axis = data.xaxis[hinge.id]
    hinge_anchor = data.xanchor[hinge.id]
    tangent = np.cross(hinge_axis, data.site(VALVE_HANDLE).xpos - hinge_anchor)
    length = np.linalg.norm(tangent)
    if length > np.finfo(float).eps:
        tangent /= length
    else:
        tangent[:] = 0
    wrench = np.zeros(6)

    for index in range(data.ncon):
        contact = data.contact[index]
        pair = (contact.geom1, contact.geom2)
        if lever not in pair:
            continue
        mujoco.mj_contactForce(model, data, index, wrench)
        # MuJoCo reports the force on geom2 in the contact frame. Transform it to world.
        frame = contact.frame.reshape(3, 3)
        force_on_geom2 = frame.T @ wrench[:3]
        torque_on_geom2 = frame.T @ wrench[3:]
        force_on_lever = force_on_geom2 if contact.geom2 == lever else -force_on_geom2
        torque_on_lever = torque_on_geom2 if contact.geom2 == lever else -torque_on_geom2
        other = contact.geom1 if contact.geom2 == lever else contact.geom2
        magnitude = float(np.linalg.norm(force_on_lever))
        if other in pads:
            normals[pads.index(other)] += abs(float(wrench[0]))
            moment = np.cross(contact.pos - hinge_anchor, force_on_lever) + torque_on_lever
            closing_torque += float(np.dot(moment, hinge_axis))
            tangent_force += float(np.dot(force_on_lever, tangent))
        else:
            geom_name = model.geom(other).name
            body_name = model.body(model.geom_bodyid[other]).name
            name = geom_name or f"{body_name}:geom_{other}"
            unexpected[name] = max(unexpected.get(name, 0.0), magnitude)
    return normals, closing_torque, tangent_force, unexpected


def _pad_geometry(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, float, float]:
    """Return per-pad gaps, handle offset across the pads, and grip position along the lever."""
    lever = model.geom(VALVE_LEVER).id
    pads = [model.geom(name).id for name in GRIPPER_PADS]
    gaps = np.empty(len(pads))
    segment = np.empty(6)
    for index, pad in enumerate(pads):
        gaps[index] = mujoco.mj_geomDistance(model, data, pad, lever, 1.0, segment)

    centres = np.array([data.geom_xpos[pad] for pad in pads])
    pad_axis = centres[1] - centres[0]
    pad_axis /= np.linalg.norm(pad_axis)
    midpoint = centres.mean(axis=0)
    handle = data.site(VALVE_HANDLE).xpos
    handle_axis_offset = float(np.dot(handle - midpoint, pad_axis))

    hinge = model.joint(VALVE_HINGE)
    hinge_anchor = data.xanchor[hinge.id]
    radial = handle - hinge_anchor
    radial /= np.linalg.norm(radial)
    handle_axial_position = float(np.dot(midpoint - hinge_anchor, radial))
    return gaps, handle_axis_offset, handle_axial_position


def observe_grasp(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, object]:
    """Return SI contact measurements for the physical two-pad grasp.

    The historical field ``pad_closing_torque_nm`` is positive in the compiled hinge's
    positive joint direction. This is the commanded stroke, including the reversed opening
    installation. Positive ``hinge_friction_torque_nm`` measures resistance to this stroke.
    Pad arrays are ordered upper, lower.
    ``handle_axial_position_m`` can be differenced from its value at grasp confirmation to obtain
    axial slip along the lever.
    """
    hinge = model.joint(VALVE_HINGE)
    dof = int(hinge.dofadr[0])
    qpos = int(hinge.qposadr[0])
    normals, closing_torque, tangent_force, unexpected = _grasp_contacts(model, data)
    gaps, handle_axis_offset, handle_axial_position = _pad_geometry(model, data)
    friction = _constraint_torque(
        model,
        data,
        mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF,
        dof,
        dof,
    )
    return {
        "valve_angle_deg": float(np.degrees(data.qpos[qpos])),
        "valve_speed_rad_s": float(data.qvel[dof]),
        "hinge_friction_torque_nm": -friction,
        "signed_hinge_friction_torque_nm": friction,
        "pad_normal_force_n": normals,
        "pad_closing_torque_nm": closing_torque,
        "pad_closing_tangent_force_n": tangent_force,
        "pad_surface_gap_m": gaps,
        "handle_axis_offset_m": handle_axis_offset,
        "handle_axial_position_m": handle_axial_position,
        "max_unexpected_lever_contact_n": max(unexpected.values(), default=0.0),
        "unexpected_lever_contacts_n": unexpected,
    }


def observe(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return angle, speed, friction torque, pin gap, and pin hinge torque."""
    hinge = model.joint(VALVE_HINGE)
    dof = int(hinge.dofadr[0])
    qpos = int(hinge.qposadr[0])
    friction_torque = _constraint_torque(
        model,
        data,
        mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF,
        dof,
        dof,
    )
    pin_torque = _constraint_torque(
        model,
        data,
        mujoco.mjtConstraint.mjCNSTR_EQUALITY,
        model.equality(VALVE_PIN).id,
        dof,
    )
    separation = data.site(VALVE_HANDLE).xpos - data.site("arm_tip").xpos
    return np.array(
        [
            data.qpos[qpos],
            data.qvel[dof],
            friction_torque,
            np.linalg.norm(separation),
            pin_torque,
        ],
        dtype=float,
    )
