"""Spatial stress field along each RBQ-10/RB1 link from the two end wrenches of that link.

Each measured joint reaction is the wrench the parent applies to the child subtree at the joint
anchor (``StairLoads``: force and moment about the anchor, world axes). For one link the
internal wrench carried across a cut at point ``x`` on the segment from proximal anchor ``p``
to distal end ``c`` is

* proximal estimate: ``F_p`` and ``M_p + (p - x) x F_p``
* distal estimate: ``F_c`` and ``M_c + (c - x) x F_c``

where the distal wrench is the next joint's reaction or, for a calf, the negative ground
force at the foot contact (zero moment). Both are exact for a massless link; the field blends
them linearly so link weight and inertia are spread along the segment. The section is the same
conditional thin-wall rectangle as ``stair_body_stress`` (outer envelope from the pinned model,
assumed wall). Stress is decomposed against the link axis: axial ``N/A``, bending
``(M_perp x r_hat) . e / Z_min`` on the side of the surface point, torsion ``T/(2 A_m t)`` and
shear ``2 |F_perp| / A``, combined as von Mises. This is a visual estimate, not FEA.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from stair_body_stress import JOINT_FAMILIES, SECTION_FAMILIES, _section_properties

from go2_wall.stair_loads import FOOT_NAMES, JOINT_NAMES

LEG_KO = {"FL": "앞왼", "FR": "앞오", "RL": "뒤왼", "RR": "뒤오"}
PART_KO = {"hip": "고관절 블록", "thigh": "허벅지", "calf": "종아리"}
ENDS_KO = {
    "hip": ("몸통 쪽", "허벅지 쪽"),
    "thigh": ("고관절 쪽", "무릎 쪽"),
    "calf": ("무릎 쪽", "발 쪽"),
    "arm": ("어깨 쪽", "손 쪽"),
}
FOOT_RADIUS_M = 0.03
SHORT_LINK_M = 0.06
PROFILE_POINTS = np.linspace(0.0, 1.0, 21)


def link_label(index: int) -> str:
    """Korean name of a measured link.

    >>> link_label(JOINT_NAMES.index("RL_calf_joint"))
    '뒤왼 종아리'
    >>> link_label(JOINT_NAMES.index("arm_joint2"))
    '팔 2번 링크'
    """
    name = JOINT_NAMES[index]
    if name.startswith("arm_joint"):
        return f"팔 {name.removeprefix('arm_joint')}번 링크"
    leg, part, _ = name.split("_")
    return f"{LEG_KO[leg]} {PART_KO[part]}"


def location_label(index: int, t: float) -> str:
    """Where along the link the peak sits, from proximal (0) to distal (1).

    >>> location_label(JOINT_NAMES.index("RL_calf_joint"), 0.05)
    '무릎 쪽'
    >>> location_label(JOINT_NAMES.index("RL_calf_joint"), 0.5)
    '중간'
    """
    name = JOINT_NAMES[index]
    part = "arm" if name.startswith("arm_joint") else name.split("_")[1]
    near, far = ENDS_KO[part]
    return near if t < 1 / 3 else far if t > 2 / 3 else "중간"


@dataclass(frozen=True)
class LinkTopology:
    """Static model addresses for the 18 measured links."""

    joint_ids: tuple[int, ...]
    body_ids: tuple[int, ...]
    distal: tuple[tuple[str, int], ...]  # ("joint", link) | ("foot", leg) | ("tip", -1)
    geoms: tuple[tuple[int, ...], ...]
    foot_geoms: tuple[int, ...]
    tip_bodies: tuple[int, ...]
    sections: tuple[tuple[float, float, float, float], ...]  # area, Z_min, A_m, wall


def link_topology(model: mujoco.MjModel) -> LinkTopology:
    joint_ids = tuple(model.joint(name).id for name in JOINT_NAMES)
    body_ids = tuple(int(model.jnt_bodyid[j]) for j in joint_ids)
    foot_geoms = tuple(model.geom(name).id for name in FOOT_NAMES)
    tip_bodies = tuple(model.body(name).id for name in ("RB1_finger_upper", "RB1_finger_lower"))
    distal, geoms = [], []
    for index, body in enumerate(body_ids):
        children = [k for k in range(model.nbody) if model.body_parentid[k] == body]
        child_links = [body_ids.index(k) for k in children if k in body_ids]
        owned = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
        name = JOINT_NAMES[index]
        if child_links:
            distal.append(("joint", child_links[0]))
        elif name.endswith("_calf_joint"):
            leg = FOOT_NAMES.index(name.split("_")[0])
            distal.append(("foot", leg))
            foot_body = int(model.geom_bodyid[foot_geoms[leg]])
            owned += [g for g in range(model.ngeom) if model.geom_bodyid[g] == foot_body]
        else:
            distal.append(("tip", -1))
            owned += [g for g in range(model.ngeom) if model.geom_bodyid[g] in tip_bodies]
        geoms.append(tuple(owned))
    sections = []
    for family in JOINT_FAMILIES:
        spec = SECTION_FAMILIES[family]
        area, modulus, median = _section_properties(spec)
        sections.append((area, modulus, median, spec["wall_m"]))
    return LinkTopology(
        joint_ids, body_ids, tuple(distal), tuple(geoms), foot_geoms, tip_bodies, tuple(sections)
    )


@dataclass(frozen=True)
class LinkState:
    """End points and end wrenches of every link at one recorded instant."""

    p: np.ndarray  # (18, 3)
    c: np.ndarray  # (18, 3)
    f_p: np.ndarray
    m_p: np.ndarray
    f_c: np.ndarray
    m_c: np.ndarray


def link_state(
    topology: LinkTopology,
    data: mujoco.MjData,
    force: np.ndarray,
    moment: np.ndarray,
    foot_force: np.ndarray,
) -> LinkState:
    """Collect end wrenches; ``foot_force`` is (4, 3), ground force on each foot."""
    p = np.array([data.xanchor[j] for j in topology.joint_ids])
    c = np.empty_like(p)
    f_c = np.empty_like(p)
    m_c = np.zeros_like(p)
    for index, (kind, ref) in enumerate(topology.distal):
        if kind == "joint":
            c[index] = data.xanchor[topology.joint_ids[ref]]
            f_c[index] = force[ref]
            m_c[index] = moment[ref]
        elif kind == "foot":
            ground = foot_force[ref]
            centre = data.geom_xpos[topology.foot_geoms[ref]]
            magnitude = np.linalg.norm(ground)
            direction = ground / magnitude if magnitude > 1e-9 else np.array([0.0, 0.0, 1.0])
            c[index] = centre - FOOT_RADIUS_M * direction
            f_c[index] = -ground
        else:
            c[index] = np.mean([data.xpos[b] for b in topology.tip_bodies], axis=0)
            f_c[index] = force[index]
            m_c[index] = moment[index] + np.cross(p[index] - c[index], force[index])
        if np.linalg.norm(c[index] - p[index]) < SHORT_LINK_M:
            # Coincident anchors (RB1 base yaw and shoulder pitch): run the segment toward the
            # link's own centre of mass and carry only the proximal wrench along it.
            toward = data.xipos[topology.body_ids[index]] - p[index]
            if np.linalg.norm(toward) < 1e-6:
                toward = data.xaxis[topology.joint_ids[index]]
            c[index] = p[index] + SHORT_LINK_M * toward / np.linalg.norm(toward)
            f_c[index] = force[index]
            m_c[index] = moment[index] + np.cross(p[index] - c[index], force[index])
    return LinkState(p, c, np.asarray(force, float), np.asarray(moment, float), f_c, m_c)


def _section_wrench(state: LinkState, index: int, t: np.ndarray):
    p, c = state.p[index], state.c[index]
    x = p + t[:, None] * (c - p)
    proximal = state.m_p[index] + np.cross(p - x, state.f_p[index])
    distal = state.m_c[index] + np.cross(c - x, state.f_c[index])
    weight = t[:, None]
    force = (1 - weight) * state.f_p[index] + weight * state.f_c[index]
    return x, force, (1 - weight) * proximal + weight * distal


def _components(state: LinkState, topology: LinkTopology, index: int, t: np.ndarray):
    axis = state.c[index] - state.p[index]
    length = np.linalg.norm(axis)
    if length < 1e-6:
        raise ValueError(f"degenerate link segment: {JOINT_NAMES[index]}")
    e = axis / length
    x, force, moment = _section_wrench(state, index, t)
    axial = force @ e
    shear = np.linalg.norm(force - axial[:, None] * e, axis=1)
    torsion = moment @ e
    bending = moment - torsion[:, None] * e
    area, modulus, median, wall = topology.sections[index]
    tau = np.abs(torsion) / (2 * median * wall) + 2 * shear / area
    return e, x, axial / area, bending / modulus, tau


def link_profile(state: LinkState, topology: LinkTopology, index: int) -> np.ndarray:
    """Outer-fibre von Mises stress (MPa) at ``PROFILE_POINTS`` along one link."""
    _, _, sigma_n, sigma_b, tau = _components(state, topology, index, PROFILE_POINTS)
    sigma = np.abs(sigma_n) + np.linalg.norm(sigma_b, axis=1)
    return np.sqrt(sigma**2 + 3 * tau**2) / 1e6


def link_peaks(state: LinkState, topology: LinkTopology) -> tuple[np.ndarray, np.ndarray]:
    """Peak stress (MPa) and its position ``t`` for all 18 links."""
    profiles = np.array([link_profile(state, topology, k) for k in range(len(JOINT_NAMES))])
    where = np.argmax(profiles, axis=1)
    return profiles[np.arange(len(where)), where], PROFILE_POINTS[where]


def end_mismatch(state: LinkState, index: int) -> float:
    """Distance (N m) between proximal and distal moment estimates at the link midpoint."""
    p, c = state.p[index], state.c[index]
    x = (p + c) / 2
    proximal = state.m_p[index] + np.cross(p - x, state.f_p[index])
    distal = state.m_c[index] + np.cross(c - x, state.f_c[index])
    return float(np.linalg.norm(proximal - distal))


def surface_stress(
    state: LinkState, topology: LinkTopology, index: int, points: np.ndarray
) -> np.ndarray:
    """Von Mises stress (MPa) at world surface points of one link."""
    p, c = state.p[index], state.c[index]
    axis = c - p
    t = np.clip((points - p) @ axis / (axis @ axis), 0.0, 1.0)
    e, x, sigma_n, sigma_b, tau = _components(state, topology, index, t)
    offset = points - x
    offset -= (offset @ e)[:, None] * e
    norm = np.linalg.norm(offset, axis=1)
    worst = np.cross(e, sigma_b)  # direction where (M x r) . e is largest
    worst_norm = np.linalg.norm(worst, axis=1)
    fallback = np.divide(
        worst, worst_norm[:, None], where=worst_norm[:, None] > 0, out=np.zeros_like(worst)
    )
    direction = np.where(norm[:, None] > 1e-6, offset / np.maximum(norm, 1e-12)[:, None], fallback)
    bending = np.einsum("ij,ij->i", np.cross(sigma_b, direction), np.broadcast_to(e, x.shape))
    sigma = sigma_n + bending
    return np.sqrt(sigma**2 + 3 * tau**2) / 1e6


def world_points(depth: np.ndarray, mask: np.ndarray, camera, shape) -> np.ndarray:
    """Reproject masked depth pixels to world coordinates (same model as the chassis)."""
    rows, columns = np.nonzero(mask)
    distance = depth[mask].astype(float)
    forward = np.asarray(camera.forward, dtype=float)
    up = np.asarray(camera.up, dtype=float)
    right = np.cross(forward, up)
    half_height = shape[0] / 2
    horizontal = ((columns + 0.5) - shape[1] / 2) / half_height
    vertical = (half_height - (rows + 0.5)) / half_height
    projection = float(camera.frustum_top / camera.frustum_near)
    rays = forward + projection * (horizontal[:, None] * right + vertical[:, None] * up)
    return np.asarray(camera.pos) + distance[:, None] * rays


def paint_links(
    frame: np.ndarray,
    segmentation: np.ndarray,
    depth: np.ndarray,
    camera,
    state: LinkState,
    topology: LinkTopology,
    colour,
    shade_reference: float,
) -> None:
    """Colour every visible pixel of the 18 links by its own position on the link."""
    geom_ids = segmentation[..., 0]
    is_geom = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    for index, geoms in enumerate(topology.geoms):
        mask = is_geom & np.isin(geom_ids, geoms)
        if not mask.any():
            continue
        points = world_points(depth, mask, camera, frame.shape)
        stress = surface_stress(state, topology, index, points)
        shade = np.clip(frame[mask].mean(axis=1, keepdims=True) / shade_reference, 0.42, 1.12)
        frame[mask] = np.rint(np.clip(colour(stress) * 255 * shade, 0, 255)).astype(np.uint8)


# Trunk: the same thin-wall estimate on the base body, loaded only through its five children.
TRUNK_WALL_M = 0.003
TRUNK_MOUNT_BODY = "RB1_Base"
HIP_JOINTS = tuple(f"{leg}_hip_joint" for leg in ("FL", "FR", "RL", "RR"))
TRUNK_LOAD_LABELS = ("FL hip", "FR hip", "RL hip", "RR hip", "RB1 mount")


def trunk_box_geom(model: mujoco.MjModel) -> int:
    """The pinned trunk's collision box (the only box geom on ``base``)."""
    base = model.body("base").id
    boxes = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == base and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
    ]
    if len(boxes) != 1:
        raise ValueError("expected exactly one trunk box geom on the base body")
    return boxes[0]


@dataclass(frozen=True)
class TrunkState:
    """Trunk axis, section and the external wrenches it receives at one instant."""

    centre: np.ndarray  # box centre (world)
    axis: np.ndarray  # box x axis (world)
    half_length: float
    points: np.ndarray  # (5, 3) load points: four hip anchors, RB1 mount origin
    forces: np.ndarray  # (5, 3) force ON the trunk
    moments: np.ndarray  # (5, 3) moment ON the trunk about each load point
    residual_force: np.ndarray  # trunk gravity and inertia, spread along its length
    residual_moment: np.ndarray  # about ``centre``
    section: tuple[float, float, float, float]


def trunk_section(model: mujoco.MjModel) -> tuple[float, float, float, float]:
    half = model.geom_size[trunk_box_geom(model)]
    spec = {"width_m": 2 * half[1], "height_m": 2 * half[2], "wall_m": TRUNK_WALL_M}
    area, modulus, median = _section_properties(spec)
    return area, modulus, median, TRUNK_WALL_M


def trunk_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    force: np.ndarray,
    moment: np.ndarray,
    mount_wrench: np.ndarray,
) -> TrunkState:
    """Loads on the trunk are minus the reactions it applies to the hips and to RB1."""
    hips = [JOINT_NAMES.index(name) for name in HIP_JOINTS]
    points = [data.xanchor[model.joint(JOINT_NAMES[k]).id] for k in hips]
    points.append(data.xpos[model.body(TRUNK_MOUNT_BODY).id])
    points = np.asarray(points)
    mount_wrench = np.asarray(mount_wrench, dtype=float)
    forces = np.vstack([-np.asarray(force, dtype=float)[hips], -mount_wrench[None, :3]])
    moments = np.vstack([-np.asarray(moment, dtype=float)[hips], -mount_wrench[None, 3:]])
    box = trunk_box_geom(model)
    centre = data.geom_xpos[box].copy()
    axis = data.geom_xmat[box].reshape(3, 3)[:, 0].copy()
    residual_force = -forces.sum(axis=0)
    residual_moment = -(moments + np.cross(points - centre, forces)).sum(axis=0)
    return TrunkState(
        centre,
        axis,
        float(model.geom_size[box][0]),
        points,
        forces,
        moments,
        residual_force,
        residual_moment,
        trunk_section(model),
    )


def _trunk_components(trunk: TrunkState, s: np.ndarray):
    """Internal wrench carried to the front part across cuts at axial positions ``s``."""
    e, half = trunk.axis, trunk.half_length
    x = trunk.centre + s[:, None] * e
    load_s = (trunk.points - trunk.centre) @ e
    front = (load_s[None, :] > s[:, None]).astype(float)
    force = -(front @ trunk.forces)
    lever = trunk.points[None, :, :] - x[:, None, :]
    moment = -np.einsum(
        "nk,nkj->nj", front, trunk.moments[None] + np.cross(lever, trunk.forces[None])
    )
    share = np.clip((half - s) / (2 * half), 0.0, 1.0)
    centroid = trunk.centre + ((s + half) / 2)[:, None] * e
    spread_force = share[:, None] * trunk.residual_force
    force -= spread_force
    moment -= share[:, None] * trunk.residual_moment + np.cross(centroid - x, spread_force)
    axial = force @ e
    shear = np.linalg.norm(force - axial[:, None] * e, axis=1)
    torsion = moment @ e
    bending = moment - torsion[:, None] * e
    area, modulus, median, wall = trunk.section
    tau = np.abs(torsion) / (2 * median * wall) + 2 * shear / area
    return x, axial / area, bending / modulus, tau


def trunk_profile(trunk: TrunkState, samples: int = 41) -> tuple[np.ndarray, np.ndarray]:
    """Outer-fibre von Mises stress (MPa) along the trunk and the axial positions used."""
    s = np.linspace(-trunk.half_length, trunk.half_length, samples)
    _, sigma_n, sigma_b, tau = _trunk_components(trunk, s)
    sigma = np.abs(sigma_n) + np.linalg.norm(sigma_b, axis=1)
    return np.sqrt(sigma**2 + 3 * tau**2) / 1e6, s


def trunk_surface_stress(trunk: TrunkState, points: np.ndarray) -> np.ndarray:
    e = trunk.axis
    s = np.clip((points - trunk.centre) @ e, -trunk.half_length, trunk.half_length)
    x, sigma_n, sigma_b, tau = _trunk_components(trunk, s)
    offset = points - x
    offset -= (offset @ e)[:, None] * e
    norm = np.linalg.norm(offset, axis=1)
    worst = np.cross(e, sigma_b)
    worst_norm = np.linalg.norm(worst, axis=1)
    fallback = np.divide(
        worst, worst_norm[:, None], where=worst_norm[:, None] > 0, out=np.zeros_like(worst)
    )
    direction = np.where(norm[:, None] > 1e-6, offset / np.maximum(norm, 1e-12)[:, None], fallback)
    bending = np.cross(sigma_b, direction) @ e
    return np.sqrt((sigma_n + bending) ** 2 + 3 * tau**2) / 1e6


def paint_trunk(
    frame, segmentation, depth, camera, trunk: TrunkState, geoms, colour, shade_reference
) -> dict:
    """Colour visible trunk pixels; returns the visible stress range for the receipt."""
    mask = (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(
        segmentation[..., 0], geoms
    )
    if not mask.any():
        return {"mesh_pixel_count": 0}
    stress = trunk_surface_stress(trunk, world_points(depth, mask, camera, frame.shape))
    shade = np.clip(frame[mask].mean(axis=1, keepdims=True) / shade_reference, 0.42, 1.12)
    frame[mask] = np.rint(np.clip(colour(stress) * 255 * shade, 0, 255)).astype(np.uint8)
    return {
        "mesh_pixel_count": int(mask.sum()),
        "visible_min_mpa": float(stress.min()),
        "visible_max_mpa": float(stress.max()),
        "visible_mean_mpa": float(stress.mean()),
    }
