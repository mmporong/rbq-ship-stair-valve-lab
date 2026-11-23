"""Torque controllers compared on the same articulated RBQ stair and passive valve."""

import hashlib
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

from go2_wall.grasp_motion import OPEN_FINGER_M
from go2_wall.robot import LEGS, ROOT


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_reference(source, walk, destination):
    """Sample a single nominal expert demonstration on its original motor clock."""
    import json

    scene = json.loads((source / "scene.json").read_bytes())
    result = json.loads((source / "result.json").read_bytes())
    raw = source / "mission.npz"
    if result["raw_sha256"] != digest(raw) or result["scene_sha256"] != digest(
        source / "scene.json"
    ):
        raise ValueError("Demonstration fingerprints do not match the published trial")
    if digest(walk) != scene["walking_reference"]["sha256"]:
        raise ValueError("Initial floor pose belongs to a different walking reference")
    from render_stair_mission import build_model

    model, _ = build_model(scene)
    stride = round(scene["actuation_period_s"] / model.opt.timestep)
    with np.load(walk, allow_pickle=False) as track:
        pose = track["qpos"][0]
    start = model.qpos0.copy()
    start[: len(pose)] = pose
    for joint in ("RB1_finger_upper_joint", "RB1_finger_lower_joint"):
        start[model.joint(joint).qposadr[0]] = OPEN_FINGER_M
    with np.load(raw, allow_pickle=False) as teacher:
        time_s = teacher["time_s"]
        if stride != 5 or not np.allclose(time_s, (np.arange(len(time_s)) + 1) * 0.001):
            raise ValueError("Expected 1 ms states and a 5 ms control period")
        previous = np.arange(stride - 1, len(time_s) - 1, stride)
        qpos = np.concatenate((start[None], teacher["qpos"][previous]))
        qvel = np.concatenate((np.zeros((1, model.nv)), teacher["qvel"][previous]))
        command = teacher["ctrl"][::stride]
        feet = teacher["feet_xyz_m"][::stride]
        stance = teacher["tread_load_n"][::stride].max(axis=2) > 8.0
        with destination.open("xb") as f:
            np.savez_compressed(
                f,
                qpos=qpos,
                qvel=qvel,
                command=command,
                feet=feet,
                stance=stance,
                raw_sha256=result["raw_sha256"],
                scene_sha256=result["scene_sha256"],
                walk_sha256=scene["walking_reference"]["sha256"],
                stride=stride,
                dt=0.001,
            )


class ReferenceControl:
    """Feedback around an expert's nominal torques, with estimated ground reactions."""

    def __init__(self, model, reference, mode, y, yaw):
        self.model = model
        self.ref = reference
        self.mode = mode
        self.y = y
        self.yaw = math.radians(yaw)
        joints = model.actuator_trnid[:, 0]
        self.dofs = model.jnt_dofadr[joints]
        self.feet = [model.geom(name).id for name in LEGS]
        self.base = model.body("base").id
        self.mass = model.body_mass.sum()
        self.kp = np.repeat([170.0, 110.0, 1000.0], [12, 6, 1])
        self.kd = np.repeat([4.5, 3.5, 8.0], [12, 6, 1])
        self.difference = np.zeros(model.nv)
        self.jac = np.zeros((3, model.nv))
        self.wrench_map = np.zeros((6, 12))
        self.wrench_map[:3] = np.tile(np.eye(3), 4)
        self.foot_velocity = np.diff(reference["feet"], axis=0) / 0.005
        self.weight = np.diag(np.full(6, 0.015))

    def torque(self, k, data, supported):
        r = self.ref
        u = r["command"][k]
        if self.mode == "teacher_open_loop":
            return u.copy()
        position, velocity = r["qpos"][k], r["qvel"][k]
        mujoco.mj_differentiatePos(self.model, self.difference, 1.0, data.qpos, position)
        correction = self.kp * self.difference[self.dofs]
        correction += self.kd * (velocity[self.dofs] - data.qvel[self.dofs])
        if self.mode != "contact_gated_reference":
            return u + correction

        # Unload the initial frame offset gradually as actual feet take new steps.
        alpha = max(0.0, 1 - k * 0.005 / 18.0)
        angle = self.yaw * alpha
        c, s = math.cos(angle), math.sin(angle)
        yaw_rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        origin = r["qpos"][0, :3]
        shifted = origin + np.array([0.0, self.y * alpha, 0.0])
        centre = shifted + yaw_rotation @ (position[:3] - origin)
        desired_feet = (r["feet"][k] - origin) @ yaw_rotation.T + shifted
        planned = r["stance"][k]
        loaded = planned & supported
        orientation = np.empty(4)
        mujoco.mju_mulQuat(
            orientation,
            np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]),
            position[3:7],
        )
        orientation_error = np.empty(3)
        mujoco.mju_subQuat(orientation_error, orientation, data.qpos[3:7])
        force = self.mass * (
            [32.0, 45.0, 61.0] * (centre - data.qpos[:3])
            + [9.0, 12.0, 12.0] * (velocity[:3] - data.qvel[:3])
        )
        moment = data.xmat[self.base].reshape(3, 3) @ (
            [440.0, 550.0, 210.0] * orientation_error
            + [32.0, 40.0, 19.0] * (velocity[3:6] - data.qvel[3:6])
        )
        reactions = np.zeros((4, 3))
        if loaded.any():
            for i, foot in enumerate(self.feet):
                x, y, z = data.geom_xpos[foot] - data.xpos[self.base]
                self.wrench_map[3:, 3 * i : 3 * i + 3] = [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]
            mapping = self.wrench_map[:, np.repeat(loaded, 3)]
            solution = np.linalg.solve(mapping @ mapping.T + self.weight, np.r_[force, moment])
            reactions[loaded] = (mapping.T @ solution).reshape(-1, 3)
        foot_velocity = self.foot_velocity[max(0, k - 1)]
        for i, foot in enumerate(self.feet):
            mujoco.mj_jacGeom(self.model, data, self.jac, None, foot)
            foot_force = -reactions[i]
            if not planned[i]:
                foot_force = 2200.0 * (desired_feet[i] - data.geom_xpos[foot])
                foot_force += 32.0 * (foot_velocity[i] - self.jac @ data.qvel)
            correction[:12] += self.jac[:, self.dofs[:12]].T @ foot_force
        return u + correction


class ArticulatedActor:
    """Run the frozen actor on measurements, reproducing its named position-servo law."""

    def __init__(self, model, bundle, mode):
        if str(ROOT / "train") not in sys.path:
            sys.path.insert(0, str(ROOT / "train"))
        from probe_stair_entry import load_policy

        self.policy, _ = load_policy(Path(bundle))
        self.mirror = mujoco.MjData(self.policy.model)
        self.mode = mode
        self.policy.phase_offset_steps = 25 if mode == "phase_25" else 0
        self.policy.target_adjustment = None
        names = [n.removeprefix("robot/") for n in self.policy.meta["action_joints"]]
        self.pose_ids = [model.joint(name).qposadr[0] for name in names]
        self.dof_ids = [model.joint(name).dofadr[0] for name in names]
        destination = list(model.actuator_trnid[:, 0])
        self.motors = [destination.index(model.joint(name).id) for name in names]
        ids = self.policy.actuator_ids
        self.stiffness = self.policy.model.actuator_gainprm[ids, 0]
        self.damping = -self.policy.model.actuator_biasprm[ids, 2]
        self.previous = np.zeros(12, dtype=np.float32)

    def apply(self, k, data, command):
        from eval_target_stairs import route_command

        p = self.policy
        if k % 4 == 0:
            # This buffer calculates policy inputs only. The physics state is never assigned here.
            self.mirror.qpos[:7] = data.qpos[:7]
            self.mirror.qvel[:6] = data.qvel[:6]
            self.mirror.qpos[p.pos_ids] = data.qpos[self.pose_ids]
            self.mirror.qvel[p.vel_ids] = data.qvel[self.dof_ids]
            mujoco.mj_forward(p.model, self.mirror)
            speed = 0.6
            if self.mode == "edge_slow_rule":
                speed = np.interp(data.qpos[0], [-0.65, -0.35], [0.6, 0.22])
            velocity, _ = route_command(p, self.mirror, "ascend", speed)
            obs = p.observe(self.mirror, velocity, self.previous, k // 4)
            self.previous = p.applied_action(p.actor(obs))
            self.target = p.position_targets(self.previous)
        command[self.motors] = (
            self.stiffness * (self.target - data.qpos[self.pose_ids])
            - self.damping * data.qvel[self.dof_ids]
        )
        return command
