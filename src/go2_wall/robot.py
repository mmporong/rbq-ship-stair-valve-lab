"""Build Go2 with the D1 or Z1 arm, or Rainbow RBQ-10 with its RB1 arm, from pinned assets."""

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GO2_XML = ROOT / ".assets/menagerie/unitree_go2/go2.xml"
D1_URDF = ROOT / ".assets/d1/d1_550_description/urdf/d1_550_description.urdf"
D1_MESH = ROOT / ".assets/d1/d1_550_description/meshes"
Z1_XML = ROOT / ".assets/z1/unitree_z1/z1_gripper.xml"
Z1_TIP = (0.18, 0.0, -0.009)
"""Grasp point between the Z1 gripper pads in the link06 frame; the pads sit near x 0.186."""
RBQ_XML = ROOT / ".assets/rbq/resources/model/rbq10_with_rb1.xml"
"""Rainbow Robotics' own RBQ-10 + RB1 arm model (RBQ repo release v1.2609.0)."""
RBQ_LEGS = {"HR": "RR", "HL": "RL", "FR": "FR", "FL": "FL"}
RBQ_PARTS = {"R": "hip", "P": "thigh", "K": "calf"}
RB1_JOINTS = [
    "RB1_Base_Yaw_joint",
    "RB1_Shoulder_Pitch_joint",
    "RB1_Elbow_Pitch_joint",
    "RB1_Elbow_Yaw_joint",
    "RB1_Wrist_Pitch_joint",
    "RB1_Wrist_Yaw_joint",
]
RBQ_FOOT_Y = 0.19285
"""Foot centre y with the legs straight down: hip 0.09 + thigh 0.10285 in the RBQ model."""
RBQ_HIP_X = 0.31218
PARAMS = ROOT / "configs/d1_parameters.json"
ARM_JOINTS = [f"arm_joint{i}" for i in range(1, 7)]
LEGS = ["FL", "FR", "RL", "RR"]
LEG_PARTS = ["hip", "thigh", "calf"]
LEG_LIMITS = np.array([23.7, 23.7, 45.43] * 4)


def set_damping(joint: mujoco.MjsJoint, value: float) -> None:
    """Set joint damping on MuJoCo 4.0, where it is an array, and on 3.x, where it is a float.

    The training environment pins MuJoCo 3.5, and tools/integrator_study.py runs there too.
    """
    if np.ndim(joint.damping) == 0:
        joint.damping = value
    else:
        joint.damping[0] = value


def load_parameters() -> dict:
    return json.loads(PARAMS.read_text(encoding="utf-8"))


def resolve_mount(mount_x: float | None) -> float:
    """Return the arm position ahead of the stock mount; None means the chosen front bracket."""
    if mount_x is None:
        return load_parameters()["front_bracket"]["value"]["mount_forward_m"]
    return mount_x


def bracket_plate(mount_x: float, params: dict | None = None) -> dict:
    """Return the bracket plate box in the Go2 base frame, and its mass.

    >>> plate = bracket_plate(0.15)
    >>> [round(v, 4) for v in plate["size"]]
    [0.225, 0.08, 0.006]
    >>> round(plate["mass_kg"], 4)  # 2700 * 0.225 * 0.08 * 0.006
    0.2916
    """
    params = params or load_parameters()
    bracket = params["front_bracket"]["value"]
    rear = bracket["plate_rear_x_m"]
    front = mount_x + bracket["plate_ahead_of_arm_m"]
    thickness = bracket["plate_thickness_m"]
    base_z = params["mount_offset_on_go2_base_m"]["value"][2]
    size = [front - rear, bracket["plate_width_m"], thickness]
    return {
        "size": size,
        "center": [(front + rear) / 2, 0.0, base_z + thickness / 2],
        "mass_kg": params["aluminum_density_kg_m3"]["value"] * float(np.prod(size)),
    }


@dataclass
class Link:
    name: str
    parent: str | None
    pos: np.ndarray
    axis: np.ndarray | None
    joint: str | None
    limits: tuple[float, float] | None
    mass: float
    com: np.ndarray
    inertia: np.ndarray


def _vector(text: str | None) -> np.ndarray:
    return np.array([float(item) for item in (text or "0 0 0").split()])


def read_urdf(path: Path = D1_URDF) -> list[Link]:
    root = ET.parse(path).getroot()
    joints = {item.find("child").get("link"): item for item in root.findall("joint")}
    links = []
    for link in root.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        value = inertial.find("inertia")
        tensor = np.array(
            [float(value.get(key)) for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
        )
        joint = joints.get(name)
        limit = None if joint is None else joint.find("limit")
        links.append(
            Link(
                name=name,
                parent=None if joint is None else joint.find("parent").get("link"),
                pos=np.zeros(3) if joint is None else _vector(joint.find("origin").get("xyz")),
                axis=None if joint is None else _vector(joint.find("axis").get("xyz")),
                joint=None if joint is None else joint.get("name"),
                limits=None
                if limit is None
                else (float(limit.get("lower")), float(limit.get("upper"))),
                mass=float(inertial.find("mass").get("value")),
                com=_vector(inertial.find("origin").get("xyz")),
                inertia=tensor,
            )
        )
    return links


def _add_point_mass(link: Link, mass: float, point: np.ndarray) -> None:
    """Combine a point mass into a link and shift its inertia with the parallel-axis theorem."""
    total = link.mass + mass
    com = (link.mass * link.com + mass * point) / total
    tensor = np.zeros((3, 3))
    tensor[[0, 1, 2], [0, 1, 2]] = link.inertia[:3]
    tensor[0, 1] = tensor[1, 0] = link.inertia[3]
    tensor[0, 2] = tensor[2, 0] = link.inertia[4]
    tensor[1, 2] = tensor[2, 1] = link.inertia[5]

    def shift(offset: np.ndarray, weight: float) -> np.ndarray:
        return weight * (offset @ offset * np.eye(3) - np.outer(offset, offset))

    tensor = tensor + shift(link.com - com, link.mass) + shift(point - com, mass)
    link.mass, link.com = total, com
    link.inertia = tensor[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]


def distribute_mass(links: list[Link], case: str, params: dict | None = None) -> list[Link]:
    """Return links whose total mass matches the official value under one distribution case."""
    params = params or load_parameters()
    total = params["total_mass_kg"]["value"]
    urdf_mass = sum(link.mass for link in links)
    if case == "scaled":
        ratio = total / urdf_mass
        for link in links:
            link.mass *= ratio
            link.inertia = link.inertia * ratio
    elif case == "servo_lumped":
        share = (total - urdf_mass) / params["servo_count"]["value"]
        for link in links:
            if link.joint and link.joint.startswith("Joint") and link.joint[-1].isdigit():
                _add_point_mass(link, share, np.zeros(3))
        wrist = next(link for link in links if link.name == "Empty_Link6")
        _add_point_mass(wrist, share, np.array([0.0718, 0.0, 0.0]))
    else:
        raise ValueError(case)
    return links


def _add_arm(
    spec: mujoco.MjSpec, mount: mujoco.MjsBody, links: list[Link], params: dict, collide: bool
) -> None:
    bodies = {}
    fused = {"Empty_Link_L", "Empty_Link_R"}
    damping = params["arm_joint_damping_nms_rad"]["value"]
    armature = params["arm_joint_armature_kgm2"]["value"]
    for link in links:
        mesh = f"arm_{link.name}"
        spec.add_mesh(name=mesh, file=str(D1_MESH / f"{link.name}.STL"))
        parent = mount if link.parent is None else bodies[link.parent]
        body = parent.add_body(name=f"arm_{link.name}", pos=link.pos.tolist())
        body.mass = link.mass
        body.ipos = link.com.tolist()
        body.fullinertia = link.inertia.tolist()
        body.explicitinertial = True
        bodies[link.name] = body
        if link.joint and link.name not in fused:
            index = int(link.joint[-1])
            joint = body.add_joint()
            joint.name = f"arm_joint{index}"
            joint.type = mujoco.mjtJoint.mjJNT_HINGE
            joint.axis = link.axis.tolist()
            joint.range = list(link.limits)
            joint.limited = True
            set_damping(joint, damping)
            joint.armature = armature
        visual = body.add_geom()
        visual.type = mujoco.mjtGeom.mjGEOM_MESH
        visual.meshname = mesh
        visual.contype = 0
        visual.conaffinity = 0
        visual.group = 2
        if collide:
            hull = body.add_geom()
            hull.type = mujoco.mjtGeom.mjGEOM_MESH
            hull.meshname = mesh
            hull.group = 3
            hull.mass = 0
            hull.density = 0
    for index, limit in enumerate(params["arm_torque_lower_bound_nm"]["value"], start=1):
        motor = spec.add_actuator(
            name=f"arm_motor{index}", target=f"arm_joint{index}", trntype=mujoco.mjtTrn.mjTRN_JOINT
        )
        motor.ctrlrange = [-limit, limit]
        motor.ctrllimited = True
    wrist = bodies["Empty_Link6"]
    wrist.add_site(name="arm_tip", pos=[params["gripper_tip_from_joint6_m"]["value"], 0, 0])
    wrist.add_site(name="arm_payload", pos=[params["payload_point_from_joint6_m"]["value"], 0, 0])


def _add_z1(spec: mujoco.MjSpec, mount: mujoco.MjsBody, params: dict) -> None:
    """Attach the Menagerie Z1 with gripper, with torque motors at the model force ranges."""
    child = mujoco.MjSpec.from_file(str(Z1_XML))
    for mesh in child.meshes:
        mesh.file = str(Z1_XML.parent / "assets" / Path(mesh.file).name)
    for actuator in list(child.actuators):
        child.delete(actuator)
    for key in list(child.keys):
        child.delete(key)
    child.option.integrator = spec.option.integrator
    frame = mount.add_frame()
    frame.attach_body(child.body("link00"), "arm_", "")
    limits = params["z1_torque_limit_nm"]["value"]
    for index, limit in enumerate(limits, start=1):
        motor = spec.add_actuator(
            name=f"arm_motor{index}", target=f"arm_joint{index}", trntype=mujoco.mjtTrn.mjTRN_JOINT
        )
        motor.ctrlrange = [-limit, limit]
        motor.ctrllimited = True
    grip = spec.add_actuator(
        name="arm_gripper", target="arm_jointGripper", trntype=mujoco.mjtTrn.mjTRN_JOINT
    )
    grip.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    grip.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    grip.gainprm[0] = 50.0
    grip.biasprm[:3] = [0.0, -50.0, -2.0]
    grip.ctrlrange = [-1.5, 0.0]
    grip.ctrllimited = True
    grip.forcerange = [-30.0, 30.0]
    grip.forcelimited = True
    wrist = spec.body("arm_link06")
    wrist.add_site(name="arm_tip", pos=list(Z1_TIP))
    wrist.add_site(name="arm_payload", pos=list(Z1_TIP))


def platform_of(arm: str) -> str:
    """The body an arm sits on: Go2 for D1 and Z1, RBQ-10 for RB1.

    >>> platform_of("rb1"), platform_of("z1")
    ('rbq', 'go2')
    """
    return "rbq" if arm == "rb1" else "go2"


def _rbq_spec(params: dict) -> mujoco.MjSpec:
    """RBQ-10 + RB1 renamed to the Go2 conventions used by the planners.

    Base body "base", leg joints "<leg>_<hip|thigh|calf>_joint", arm joints "arm_joint1..6",
    and torque motors in statics.joint_names order. The vendor scene extras (target cube,
    floor contact pairs, sensors, cameras) are removed; the gripper fingers stay passive.
    """
    spec = mujoco.MjSpec.from_file(str(RBQ_XML))
    for item in [*spec.actuators, *spec.sensors, *spec.pairs, *spec.keys, *spec.cameras]:
        spec.delete(item)
    spec.delete(spec.body("target_cube_green_body"))
    spec.body("base_link").name = "base"
    for joint in spec.joints:
        if joint.name.startswith("joint") and "_" in joint.name:
            code = joint.name.split("_")[1]
            joint.name = f"{RBQ_LEGS[code[:2]]}_{RBQ_PARTS[code[2]]}_joint"
    for index, name in enumerate(RB1_JOINTS, start=1):
        spec.joint(name).name = f"arm_joint{index}"
    tip = list(spec.site("ee_site").pos)
    wrist = spec.body("RB1_Wrist_Yaw")
    wrist.add_site(name="arm_tip", pos=tip)
    wrist.add_site(name="arm_payload", pos=tip)
    for name in ("RB1_Base", "RB1_Base_Yaw", "RB1_Shoulder_Pitch"):
        spec.add_exclude(bodyname1="base", bodyname2=name)
    legs = params["rbq_leg_torque_limit_nm"]["value"]
    names = [f"{leg}_{part}_joint" for leg in LEGS for part in LEG_PARTS]
    limits = list(legs) * 4 + list(params["rb1_torque_limit_nm"]["value"])
    for name, limit in zip(names + ARM_JOINTS, limits):
        motor = spec.add_actuator(
            name=name.replace("_joint", "_motor"), target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT
        )
        motor.ctrlrange = [-limit, limit]
        motor.ctrllimited = True
    return spec


def build_robot(
    case: str = "scaled",
    wall_x: float | None = None,
    floor_mu: float = 0.6,
    wall_mu: float = 0.6,
    mount_x: float | None = None,
    arm: str = "d1",
) -> mujoco.MjSpec:
    """Return a Go2+D1 spec, optionally with a wall. Floor and wall friction override the feet.

    `mount_x` is the arm position ahead of the stock mount. None uses the chosen front bracket.
    Any forward position adds the bracket plate, its mass, and its thickness under the arm.
    """
    params = load_parameters()
    if arm == "rb1":
        return _add_ground(_rbq_spec(params), wall_x, floor_mu, wall_mu)
    mount_x = resolve_mount(mount_x)
    spec = mujoco.MjSpec.from_file(str(GO2_XML))
    spec.meshdir = str(GO2_XML.parent / "assets")
    offset = np.array(params["mount_offset_on_go2_base_m"]["value"]) + [mount_x, 0.0, 0.0]
    if mount_x > 0:
        plate = bracket_plate(mount_x, params)
        offset[2] += plate["size"][2]
        bracket = spec.body("base").add_body(name="bracket", pos=plate["center"])
        geom = bracket.add_geom(
            name="bracket_plate",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(np.array(plate["size"]) / 2).tolist(),
            rgba=[0.75, 0.75, 0.78, 1],
        )
        geom.density = params["aluminum_density_kg_m3"]["value"]
        spec.add_exclude(bodyname1="base", bodyname2="bracket")
    mount = spec.body("base").add_body(name="arm_mount", pos=offset.tolist())
    if arm == "z1":
        _add_z1(spec, mount, params)
        first = ("arm_link00", "arm_link01", "arm_link02")
    else:
        links = distribute_mass(read_urdf(), case, params)
        _add_arm(spec, mount, links, params, collide=True)
        first = ("arm_base_link", "arm_Empty_Link1")
    for name in first:
        spec.add_exclude(bodyname1="base", bodyname2=name)
        if mount_x > 0:
            spec.add_exclude(bodyname1="bracket", bodyname2=name)
    return _add_ground(spec, wall_x, floor_mu, wall_mu)


def _add_ground(
    spec: mujoco.MjSpec, wall_x: float | None, floor_mu: float, wall_mu: float
) -> mujoco.MjSpec:
    world = spec.worldbody
    floor = world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
    floor.friction = [floor_mu, 0.02, 0.01]
    floor.priority = 2
    if wall_x is not None:
        wall = world.add_geom(
            name="wall",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.05, 3, 2],
            pos=[wall_x + 0.05, 0, 2],
        )
        wall.friction = [wall_mu, 0.02, 0.01]
        wall.priority = 2
    return spec


def world_geoms(model: mujoco.MjModel) -> list[int]:
    """Geoms fixed to the world: floor, wall, stair treads. Feet may touch only these."""
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == 0]


def compile_robot(**kwargs) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = build_robot(**kwargs).compile()
    return model, mujoco.MjData(model)


def build_arm_only(case: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Return the D1 arm alone on a fixed base, for rated-load calculations."""
    params = load_parameters()
    spec = mujoco.MjSpec()
    links = distribute_mass(read_urdf(), case, params)
    mount = spec.worldbody.add_body(name="arm_mount")
    _add_arm(spec, mount, links, params, collide=False)
    model = spec.compile()
    return model, mujoco.MjData(model)
