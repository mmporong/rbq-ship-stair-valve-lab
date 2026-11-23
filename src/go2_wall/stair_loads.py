"""Per-step load measurements for the RBQ-10 + RB1 stair-valve mission."""

import mujoco
import numpy as np

from .robot import LEGS
from .statics import joint_names

JOINT_NAMES = tuple(joint_names())
FOOT_NAMES = tuple(LEGS)
SURFACE_NAMES = ("floor", *(f"tread{k}" for k in range(1, 11)), "platform")
RB1_MOUNT_BODY = "RB1_Base"
MOUNT_WRENCH_FIELDS = (
    "force_x_n",
    "force_y_n",
    "force_z_n",
    "moment_x_nm",
    "moment_y_nm",
    "moment_z_nm",
)


def _moment_about(
    data: mujoco.MjData, body: int, point: np.ndarray, root: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the parent reaction on a body in world axes, with moment about ``point``."""
    torque, force = data.cfrc_int[body][:3], data.cfrc_int[body][3:]
    centre = data.subtree_com[root]
    return force.copy(), torque + np.cross(centre - point, force)


class StairLoads:
    """Cache model addresses and sample mission loads after each physical step.

    Joint arrays follow ``JOINT_NAMES``. Foot/surface reactions have shape ``(4, 12, 3)``
    ordered by ``FOOT_NAMES``, ``SURFACE_NAMES``, and world XYZ. The mount wrench is force XYZ
    followed by moment XYZ about the RB1 base origin, all expressed in world axes.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.root = int(model.body_rootid[model.body("base").id])
        self.joints = [model.joint(name) for name in JOINT_NAMES]
        self.joint_bodies = np.array([model.jnt_bodyid[joint.id] for joint in self.joints])
        self.mount_body = model.body(RB1_MOUNT_BODY).id
        self.feet = [model.geom(name).id for name in FOOT_NAMES]
        self.surfaces = [model.geom(name).id for name in SURFACE_NAMES]
        self.foot_index = {geom: index for index, geom in enumerate(self.feet)}
        self.surface_index = {geom: index for index, geom in enumerate(self.surfaces)}
        self.wrench = np.zeros(6)

    def _foot_surface_reactions(self, data: mujoco.MjData) -> np.ndarray:
        reactions = np.zeros((len(self.feet), len(self.surfaces), 3))
        for index in range(data.ncon):
            contact = data.contact[index]
            pair = (contact.geom1, contact.geom2)
            if pair[0] in self.surface_index and pair[1] in self.foot_index:
                surface, foot, sign = pair[0], pair[1], 1.0
            elif pair[1] in self.surface_index and pair[0] in self.foot_index:
                surface, foot, sign = pair[1], pair[0], -1.0
            else:
                continue
            mujoco.mj_contactForce(self.model, data, index, self.wrench)
            force_on_geom2 = contact.frame.reshape(3, 3).T @ self.wrench[:3]
            reactions[self.foot_index[foot], self.surface_index[surface]] += sign * force_on_geom2
        return reactions

    def sample(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        """Measure the current constrained dynamics state without advancing simulation."""
        mujoco.mj_rnePostConstraint(self.model, data)
        forces, moments = [], []
        for joint, body in zip(self.joints, self.joint_bodies):
            force, moment = _moment_about(data, int(body), data.xanchor[joint.id], self.root)
            forces.append(force)
            moments.append(moment)
        force = np.asarray(forces)
        moment = np.asarray(moments)
        axes = np.array([data.xaxis[joint.id] for joint in self.joints])
        bending = moment - np.sum(moment * axes, axis=1)[:, None] * axes

        mount_force, mount_moment = _moment_about(
            data, self.mount_body, data.xpos[self.mount_body], self.root
        )
        return {
            "joint_force_world_n": force,
            "joint_moment_world_nm": moment,
            "joint_off_axis_bending_nm": np.linalg.norm(bending, axis=1),
            "joint_reaction_n": np.linalg.norm(force, axis=1),
            "rb1_mount_wrench_world": np.r_[mount_force, mount_moment],
            "foot_surface_force_world_n": self._foot_surface_reactions(data),
        }
