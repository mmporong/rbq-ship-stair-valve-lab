"""mjlab task: Go2 with a stowed D1 arm walking up and down ship-stair-like open stairs.

The quasi-static map (docs/stairs-feasibility.md) found no one-foot-at-a-time cycle on
50-70 degree ship stairs, so climbing is learned as a dynamic gait. The arm is not moved
while climbing; it is a rigid payload in the stair stow (mass, centre of mass, and inertia
computed from the full Go2+D1 model in src/go2_wall, see STOW_* below) with three capsules
for collision. Everything else follows the Unitree-Go2-Rough task of unitree_rl_mjlab.

Run from the unitree_rl_mjlab checkout:
  python <path to this file> train --env.scene.num-envs 4096
"""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))  # the unitree_rl_mjlab checkout provides `src`

import mujoco  # noqa: E402
from mjlab.tasks.registry import register_mjlab_task  # noqa: E402
from mjlab.terrains import primitive_terrains as prim  # noqa: E402
from src.assets.robots.unitree_go2 import go2_constants as go2  # noqa: E402
from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_rough_env_cfg  # noqa: E402
from src.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_runner_cfg  # noqa: E402
from src.tasks.velocity.rl import VelocityOnPolicyRunner  # noqa: E402

# Composite of the D1 arm in the stair stow and the 150 mm bracket plate, in the base frame.
STOW_MASS_KG = 2.6616
STOW_COM_M = (0.0778, -0.0003, 0.1748)
STOW_INERTIA = (0.013478, 0.040652, 0.030221, 6e-06, 0.003734, 4.2e-05)
STOW_CAPSULES = (
    ((0.15, 0.0, 0.07), (0.15, 0.0, 0.19)),
    ((0.15, 0.0, 0.19), (-0.12, 0.0, 0.19)),
    ((-0.10, 0.0, 0.23), (0.28, 0.0, 0.21)),
)
CAPSULE_RADIUS_M = 0.035

# Open stairs: difficulty 0 is 0.10 m rise on 0.35 m run (16 deg); difficulty 1 is 0.20 m on
# 0.12 m (59 deg), inside the OSHA ship-stair range of 50-70 deg.
STEP_HEIGHT_RANGE = (0.10, 0.20)
STEP_WIDTH_RANGE = (0.12, 0.35)


def spec_with_arm() -> mujoco.MjSpec:
    spec = go2.get_spec()
    body = spec.body("base_link").add_body(name="arm_stow")
    body.mass = STOW_MASS_KG
    body.ipos = list(STOW_COM_M)
    body.fullinertia = list(STOW_INERTIA)
    body.explicitinertial = True
    for index, (start, end) in enumerate(STOW_CAPSULES, start=1):
        body.add_geom(
            name=f"arm_stow{index}_collision",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[*start, *end],
            size=[CAPSULE_RADIUS_M, 0, 0],
            mass=0,
            density=0,
            rgba=[0.3, 0.3, 0.32, 1],
        )
    return spec


def robot_cfg():
    cfg = go2.get_go2_robot_cfg()
    cfg.spec_fn = spec_with_arm
    return cfg


def stairs_env_cfg(play: bool = False):
    cfg = unitree_go2_rough_env_cfg(play=play)
    cfg.scene.entities = {"robot": robot_cfg()}
    generator = cfg.scene.terrain.terrain_generator
    common = dict(
        step_height_range=STEP_HEIGHT_RANGE,
        step_width_range=STEP_WIDTH_RANGE,
        platform_width=2.0,
        border_width=0.5,
        step_thickness=0.03,
    )
    generator.sub_terrains = {
        "flat": prim.BoxFlatTerrainCfg(proportion=0.1),
        "open_stairs_up": prim.BoxOpenStairsTerrainCfg(proportion=0.45, inverted=False, **common),
        "open_stairs_down": prim.BoxOpenStairsTerrainCfg(proportion=0.45, inverted=True, **common),
    }
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (-0.5, 0.8)
    twist.ranges.lin_vel_y = (-0.3, 0.3)
    twist.ranges.ang_vel_z = (-0.8, 0.8)
    cfg.curriculum.pop("command_vel", None)
    return cfg


def runner_cfg():
    cfg = unitree_go2_ppo_runner_cfg()
    cfg.experiment_name = "go2_d1_stairs"
    cfg.max_iterations = 4000
    return cfg


TASK_ID = "Go2D1-ShipStairs"
register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=stairs_env_cfg(),
    play_env_cfg=stairs_env_cfg(play=True),
    rl_cfg=runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)


if __name__ == "__main__":
    mode = sys.argv[1]
    script = Path.cwd() / "scripts" / f"{mode}.py"
    sys.argv = [str(script), TASK_ID, *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")
