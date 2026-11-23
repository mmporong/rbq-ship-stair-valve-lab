"""mjlab task: Rainbow Robotics RBQ-10 with the RB1 arm stowed, walking up and down stairs.

The robot comes from Rainbow's own MuJoCo model (RBQ repo, resources/model/rbq10_with_rb1.xml,
fetched by tools/fetch_assets.py). For training:

- the RB1 arm and gripper joints are removed and their bodies fixed in the stair stow used by
  the static analysis (src/go2_wall/stairs.py STOW_RB1), so the arm keeps its real mass,
  inertia and collision cylinders;
- legs get the Unitree-style names the velocity task expects (FL_hip_joint, ...), feet get
  "<leg>_foot_collision" geoms and "<leg>" sites, and the IMU sensors are added at the
  vendor's "imu" site;
- visual meshes are dropped; collision uses the vendor's primitives only.

Leg PD gains, default joint angles and armature follow Rainbow's published rbq10_trot policy
(resources/policy/rbq10_trot/info.json) and the vendor model. The terrain mixes closed and
open stairs. The rise grows from 0 at level 0 to 0.20 m at the hardest level; open stairs
narrow from a 0.35 m to a 0.12 m run (0 to 59 deg), closed stairs keep a 0.25 m run (0 to
39 deg).

Version 1 (rise from 0.10 m, robots spawned on levels 0 to 5, mjlab's terrain_levels_vel)
learned to stand still on the spawn platform: every robot fell to level 0 within 50
iterations and none was promoted again (evidence/t1-train/). The terrain curriculum is now
`stair_levels`, tied to this terrain's geometry instead of the commanded speed.

A second task, RBQ10RB1-Flat, uses the same robot on flat ground only.

Run from the unitree_rl_mjlab checkout:
  python <this file> train [stairs|flat] --env.scene.num-envs 2048 --agent.logger tensorboard
"""

import dataclasses
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))  # the unitree_rl_mjlab checkout provides `src`

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mjlab.actuator import BuiltinPositionActuatorCfg  # noqa: E402
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg  # noqa: E402
from mjlab.managers.curriculum_manager import CurriculumTermCfg  # noqa: E402
from mjlab.managers.reward_manager import RewardTermCfg  # noqa: E402
from mjlab.sensor import (  # noqa: E402
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
)
from mjlab.tasks.registry import register_mjlab_task  # noqa: E402
from mjlab.terrains import primitive_terrains as prim  # noqa: E402
from mjlab.terrains.terrain_generator import TerrainGeometry  # noqa: E402
from mjlab.utils.spec_config import CollisionCfg  # noqa: E402
from src.tasks.velocity.config.go2.env_cfgs import (  # noqa: E402
    unitree_go2_flat_env_cfg,
    unitree_go2_rough_env_cfg,
)
from src.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_runner_cfg  # noqa: E402
from src.tasks.velocity.mdp import self_collision_cost  # noqa: E402
from src.tasks.velocity.rl import VelocityOnPolicyRunner  # noqa: E402

RBQ_XML = Path(__file__).resolve().parents[1] / ".assets/rbq/resources/model/rbq10_with_rb1.xml"
LEGS = {"HR": "RR", "HL": "RL", "FR": "FR", "FL": "FL"}
PARTS = {"R": "hip", "P": "thigh", "K": "calf"}
STOW = {"RB1_Shoulder_Pitch_joint": -1.6, "RB1_Wrist_Pitch_joint": -2.967}
"""Same stow as STOW_RB1 in src/go2_wall/stairs.py (arm_joint2, arm_joint5)."""

# rbq10_trot/info.json: stiffness R, P 123.39138, K 127.77215; damping R 2.5, P 2.3, K 2.3.
# Effort limits and armature from the vendor model (100, 100, 150 N m).
ACTUATORS = (
    BuiltinPositionActuatorCfg(
        target_names_expr=(".*_hip_joint",),
        stiffness=123.39138,
        damping=2.5,
        effort_limit=100.0,
        armature=0.014058265,
    ),
    BuiltinPositionActuatorCfg(
        target_names_expr=(".*_thigh_joint",),
        stiffness=123.39138,
        damping=2.3,
        effort_limit=100.0,
        armature=0.014058265,
    ),
    BuiltinPositionActuatorCfg(
        target_names_expr=(".*_calf_joint",),
        stiffness=127.77215,
        damping=2.3,
        effort_limit=150.0,
        armature=0.0214816,
    ),
)
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.56),
    joint_pos={
        ".*_hip_joint": 0.0,
        "F._thigh_joint": 0.7941246495,
        "F._calf_joint": -1.431169698,
        "R._thigh_joint": 0.8115779385,
        "R._calf_joint": -1.39626312,
    },
    joint_vel={".*": 0.0},
)
FOOT = "^[FR][LR]_foot_collision$"
COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision\\d*",),
    condim={FOOT: 3, ".*_collision\\d*": 1},
    priority={FOOT: 1},
    friction={FOOT: (0.6,)},
    solimp={FOOT: (0.9, 0.95, 0.023)},
    contype=1,
    conaffinity=0,
)
STEP_HEIGHT_RANGE = (0.0, 0.20)
STEP_WIDTH_RANGE = (0.12, 0.35)
CLOSED_STEP_WIDTH = 0.25
"""Closed stairs keep one run: 0.20 m rise on it is 39 deg."""
PLATFORM_M = 2.0
BORDER_M = 0.5
PROMOTE_M = 3.0
"""Stairs span 1.0 to 3.5 m from the spawn point (8 m tile, 2 m platform, 0.5 m border);
3 m means at least 2 of the 2.5 m of stairs were crossed."""
DEMOTE_M = 1.0
"""Still on the spawn platform at the end of the episode."""
MAX_INIT_LEVEL = 2
LEG = r"^[FR][LR]_(thigh|calf)_collision\d*$"
"""Hip capsule, thigh and both shin boxes of every leg (not the foot spheres)."""
LEG_CONTACT_WEIGHT = -1.0
"""Per substep with a leg contact over 10 N; x dt 0.02 and 4 substeps per step, 50 steps of
constant contact cost as much as one termination (-200 x 0.02). legged_gym uses -1 too."""


FLOOR_THICKNESS = 0.5


@dataclasses.dataclass(kw_only=True)
class OpenStairsOnFloorCfg(prim.BoxOpenStairsTerrainCfg):
    """mjlab open stairs with a solid floor under the treads.

    mjlab builds open stairs from 3 cm boards with nothing below them, the spawn platform
    included (a ray cast down from under any board finds no surface, train/
    void_check.py). A real ship ladder has open risers but stands on a deck. The floor here
    sits under the lowest board: just below the bottom platform when the stairs go up from
    the spawn point, and at ground level (z = 0, the border top) when they go down.
    """

    def function(self, difficulty, spec, rng):
        out = super().function(difficulty, spec, rng)
        step_height = self.step_height_range[0] + difficulty * (
            self.step_height_range[1] - self.step_height_range[0]
        )
        if self.inverted:
            top = out.origin[2] - self.step_thickness
        else:
            top = 0.0
        inner = (self.size[0] - 2 * self.border_width, self.size[1] - 2 * self.border_width)
        floor = spec.body("terrain").add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(inner[0] / 2.0, inner[1] / 2.0, FLOOR_THICKNESS / 2.0),
            pos=(self.size[0] / 2.0, self.size[1] / 2.0, top - FLOOR_THICKNESS / 2.0),
        )
        del step_height
        out.geometries.append(TerrainGeometry(geom=floor, color=(0.35, 0.35, 0.4, 1.0)))
        return out


STALL_WEIGHT = -1.0
"""Per second stalled while told to move. On top of the lost velocity tracking (about 0.6/s
between standing and walking at 0.6 m/s), so standing at a stair edge costs more than
the -4 of one termination after about 2.5 s."""


def stall(env, command_name: str = "twist", min_command: float = 0.2, max_speed: float = 0.1):
    """1 while the planar command exceeds min_command but the base moves slower than max_speed.

    Version 7. The 5,000-iteration policy stopped at the platform edge in front of every
    descent of 11 cm or more (100% survival, 0.6 to 1.1 m in 20 s), the stand-still optimum
    that Rudin et al. 2022 and Hoeller et al. 2024 break with a similar stall penalty.
    """
    command = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    speed = torch.norm(env.scene["robot"].data.root_link_lin_vel_b[:, :2], dim=1)
    return ((command > min_command) & (speed < max_speed)).float()


def stair_levels(env, env_ids: torch.Tensor, command_name: str = "twist") -> torch.Tensor:
    """Promote robots that crossed most of the stairs, demote moving ones that never left.

    Robots told to stand still (command under 0.1 m/s at the end) are not demoted.
    """
    terrain = env.scene.terrain
    robot = env.scene["robot"]
    offset = robot.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    distance = torch.norm(offset, dim=1)
    command = torch.norm(env.command_manager.get_command(command_name)[env_ids, :2], dim=1)
    move_up = distance > PROMOTE_M
    move_down = (distance < DEMOTE_M) & (command > 0.1) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def _freeze(spec: mujoco.MjSpec, joint: mujoco.MjsJoint, angle: float) -> None:
    """Remove a hinge or slide joint, fixing its body at `angle` (hinges at the body origin)."""
    body = spec.body(joint.parent.name)
    if joint.type == mujoco.mjtJoint.mjJNT_HINGE and angle:
        turn = np.zeros(4)
        mujoco.mju_axisAngle2Quat(turn, np.asarray(joint.axis, float), angle)
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, np.asarray(body.quat, float), turn)
        body.quat = quat.tolist()
    spec.delete(joint)


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(RBQ_XML))
    for item in [*spec.actuators, *spec.sensors, *spec.pairs, *spec.keys, *spec.cameras]:
        spec.delete(item)
    for item in list(spec.equalities):
        spec.delete(item)
    spec.delete(spec.body("target_cube_green_body"))
    for geom in list(spec.geoms):
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(geom)
    for mesh in list(spec.meshes):
        spec.delete(mesh)
    for joint in list(spec.joints):
        name = joint.name
        if name.startswith(("RB1_", "LIMS")):
            _freeze(spec, joint, STOW.get(name, 0.0))
        elif name.startswith("joint") and "_" in name:
            code = name.split("_")[1]
            joint.name = f"{LEGS[code[:2]]}_{PARTS[code[2]]}_joint"
    for leg in LEGS.values():
        geom = spec.geom(leg)
        geom.name = f"{leg}_foot_collision"
        spec.body(f"{leg}_foot").add_site(name=leg, pos=list(geom.pos), size=[0.03, 0, 0])
    count = {}
    for geom in spec.geoms:
        if not geom.name:
            owner = geom.parent.name
            count[owner] = count.get(owner, 0) + 1
            geom.name = f"{owner}_collision{count[owner]}"
    spec.add_sensor(
        name="imu_ang_vel", type=mujoco.mjtSensor.mjSENS_GYRO,
        objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu",
    )  # fmt: skip
    spec.add_sensor(
        name="imu_lin_vel", type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
        objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu",
    )  # fmt: skip
    spec.add_sensor(
        name="imu_lin_acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
        objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu",
    )  # fmt: skip
    spec.add_sensor(
        name="root_angmom", type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
        objtype=mujoco.mjtObj.mjOBJ_BODY, objname="base_link",
    )  # fmt: skip
    return spec


def robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(COLLISION,),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=ACTUATORS, soft_joint_pos_limit_factor=0.9
        ),
    )


def _use_rbq(cfg):
    """Swap the robot and set the command ranges and foot clearance shared by both tasks."""
    cfg.scene.entities = {"robot": robot_cfg()}
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (-0.5, 0.8)
    twist.ranges.lin_vel_y = (-0.3, 0.3)
    twist.ranges.ang_vel_z = (-0.8, 0.8)
    cfg.curriculum.pop("command_vel", None)
    cfg.rewards["foot_clearance"].params["target_height"] = 0.15
    cfg.viewer.distance = 2.5
    return cfg


def flat_env_cfg(play: bool = False):
    """Flat ground only: checks that the robot learns to walk at all."""
    return _use_rbq(unitree_go2_flat_env_cfg(play=play))


def _leg_contact_is_penalty(cfg) -> None:
    """Leg contacts with the terrain are penalised; only base and arm contacts end episodes.

    Version 3 did this for the lower shin box only: at 1,100 iterations, 41 of 60 robots on
    level-4 open stairs going up ended by the front lower shin touching a tread. Version 5
    extends it to the whole leg, as legged_gym does (terminate on base, penalise thigh and
    calf): at 2,800 iterations on level 6, 13 of 18 robots on open stairs going up ended by
    the upper shin box (calf_collision1) and none by the base (train/diag_contact.py).
    """
    sensors = list(cfg.scene.sensors)
    for i, sensor in enumerate(sensors):
        if sensor.name == "nonfoot_ground_touch":
            primary = dataclasses.replace(sensor.primary, exclude=(*sensor.primary.exclude, LEG))
            sensors[i] = dataclasses.replace(sensor, primary=primary)
    sensors.append(
        ContactSensorCfg(
            name="leg_ground_touch",
            primary=ContactMatch(mode="geom", entity="robot", pattern=LEG),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=4,
        )
    )
    cfg.scene.sensors = tuple(sensors)
    cfg.rewards["leg_contact"] = RewardTermCfg(
        func=self_collision_cost,
        weight=LEG_CONTACT_WEIGHT,
        params={"sensor_name": "leg_ground_touch", "force_threshold": 10.0},
    )


FOOT_SITES = ("FR", "FL", "RR", "RL")


def _foot_ray(site: str) -> RayCastSensorCfg:
    """One ray straight down from a foot site, to find the surface under that foot."""
    return RayCastSensorCfg(
        name=f"ground_under_{site}",
        frame=ObjRef(type="site", name=site, entity="robot"),
        ray_alignment="world",
        pattern=GridPatternCfg(size=(0.0, 0.0)),
        max_distance=5.0,
    )


def _foot_height_above_ground(env, asset_cfg) -> torch.Tensor:
    """[B, 4] foot site height above the surface right under it, in asset_cfg site order.

    A foot with no surface within 5 m below (off the terrain edge) counts as 0.05 m, the
    standing height on flat ground.
    """
    asset = env.scene[asset_cfg.name]
    names = [asset.site_names[i] for i in asset_cfg.site_ids]
    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    heights = []
    for k, name in enumerate(names):
        ray = env.scene[f"ground_under_{name}"].data
        hit = ray.distances[:, 0] >= 0
        heights.append(torch.where(hit, foot_z[:, k] - ray.hit_pos_w[:, 0, 2], 0.05))
    return torch.stack(heights, dim=1)


def foot_height_terrain(env, asset_cfg) -> torch.Tensor:
    return _foot_height_above_ground(env, asset_cfg)


def feet_clearance_terrain(
    env, target_height: float, command_name: str, command_threshold: float, asset_cfg
) -> torch.Tensor:
    """mjlab feet_clearance with the foot height taken above the ground under the foot.

    The original uses the world z, which on a stair tile whose spawn platform sits at
    z = -0.9 m or +1.0 m (level 4) charges every moving foot for being ~1 m off target.
    """
    asset = env.scene[asset_cfg.name]
    delta = torch.abs(_foot_height_above_ground(env, asset_cfg) - target_height)
    speed = torch.norm(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
    cost = torch.sum(delta * speed, dim=1)
    command = env.command_manager.get_command(command_name)
    moving = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) > command_threshold
    return cost * moving.float()


def _foot_height_from_ground(cfg) -> None:
    """Version 4: foot clearance (reward) and foot height (critic) relative to the ground.

    At 1,800 iterations, robots pinned to level 4 paid -0.61 to -0.74 per second in
    foot_clearance while stepping in place, against -0.19 for robots walking on flat
    ground (train/reward_terms.py).
    """
    cfg.scene.sensors = (*cfg.scene.sensors, *(_foot_ray(site) for site in FOOT_SITES))
    clearance = cfg.rewards["foot_clearance"]
    clearance.func = feet_clearance_terrain
    critic = cfg.observations["critic"].terms
    if "foot_height" in critic:
        critic["foot_height"].func = foot_height_terrain


def stairs_env_cfg(play: bool = False):
    cfg = _use_rbq(unitree_go2_rough_env_cfg(play=play))
    generator = cfg.scene.terrain.terrain_generator
    common = dict(
        step_height_range=STEP_HEIGHT_RANGE, platform_width=PLATFORM_M, border_width=BORDER_M
    )
    closed = dict(step_width=CLOSED_STEP_WIDTH, **common)
    open_ = dict(step_width_range=STEP_WIDTH_RANGE, step_thickness=0.03, **common)
    # Robots spawn on the central platform and walk outward. mjlab's pyramid (and open stairs
    # with inverted=False) puts that platform at the top, so leaving it means going down;
    # the inverted variants put it at the bottom. Before 2026-11-10 the names were swapped.
    generator.sub_terrains = {
        "flat": prim.BoxFlatTerrainCfg(proportion=0.1),
        "stairs_up": prim.BoxInvertedPyramidStairsTerrainCfg(proportion=0.2, **closed),
        "stairs_down": prim.BoxPyramidStairsTerrainCfg(proportion=0.2, **closed),
        "open_stairs_up": OpenStairsOnFloorCfg(proportion=0.25, inverted=True, **open_),
        "open_stairs_down": OpenStairsOnFloorCfg(proportion=0.25, inverted=False, **open_),
    }
    _leg_contact_is_penalty(cfg)
    _foot_height_from_ground(cfg)
    cfg.rewards["stall"] = RewardTermCfg(
        func=stall, weight=STALL_WEIGHT, params={"command_name": "twist"}
    )
    if not play:
        cfg.scene.terrain.max_init_terrain_level = MAX_INIT_LEVEL
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=stair_levels, params={"command_name": "twist"}
        )
    return cfg


def runner_cfg(experiment: str, iterations: int):
    cfg = unitree_go2_ppo_runner_cfg()
    cfg.experiment_name = experiment
    cfg.max_iterations = iterations
    return cfg


TASK_ID = "RBQ10RB1-Stairs"
FLAT_TASK_ID = "RBQ10RB1-Flat"
TASKS = {
    "stairs": (TASK_ID, stairs_env_cfg, "rbq10_rb1_stairs", 4000),
    "flat": (FLAT_TASK_ID, flat_env_cfg, "rbq10_rb1_flat", 1000),
}
for task_id, env_fn, experiment, iterations in TASKS.values():
    register_mjlab_task(
        task_id=task_id,
        env_cfg=env_fn(),
        play_env_cfg=env_fn(play=True),
        rl_cfg=runner_cfg(experiment, iterations),
        runner_cls=VelocityOnPolicyRunner,
    )


if __name__ == "__main__":
    mode, rest = sys.argv[1], sys.argv[2:]
    task = TASKS[rest.pop(0)][0] if rest and rest[0] in TASKS else TASK_ID
    script = Path.cwd() / "scripts" / f"{mode}.py"
    sys.argv = [str(script), task, *rest]
    runpy.run_path(str(script), run_name="__main__")
