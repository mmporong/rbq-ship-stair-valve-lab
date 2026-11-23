"""Export RBQ stair scene, frozen actor and reference observations from mjlab on CPU.

Run from the pinned unitree_rl_mjlab checkout. Keep the bundle (model, weights and
fixtures) in a local ignored directory. The portable evaluator needs only MuJoCo and NumPy.
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch
from eval_stairs import (
    ManagerBasedRlEnv,
    RslRlVecEnvWrapper,
    column_names,
    evaluation_config,
    load_rl_cfg,
    load_runner_cls,
    load_task,
)

FIXTURE_STEPS = (0, 1, 8)
DISABLED_EVENTS = ("push_robot", "foot_friction", "encoder_bias", "base_com")


def actor_arrays(policy: torch.nn.Module) -> dict[str, np.ndarray]:
    layers = list(policy.mlp)
    if tuple(policy.obs_groups) != ("actor",) or not layers or len(layers) % 2 != 1:
        raise ValueError("Expected a feedforward actor over the actor observation group")
    arrays = {
        "obs_mean": policy.obs_normalizer.mean.detach().numpy(),
        "obs_std": policy.obs_normalizer.std.detach().numpy(),
        "obs_eps": np.asarray(policy.obs_normalizer.eps, dtype=np.float32),
    }
    for i, layer in enumerate(layers):
        if i % 2:
            if not isinstance(layer, torch.nn.ELU) or layer.alpha != 1.0:
                raise ValueError(f"Unsupported actor activation: {layer}")
        else:
            if not isinstance(layer, torch.nn.Linear):
                raise ValueError(f"Unsupported actor layer: {layer}")
            arrays[f"weight_{i // 2}"] = layer.weight.detach().numpy()
            arrays[f"bias_{i // 2}"] = layer.bias.detach().numpy()
    return arrays


def scene_xml(env: ManagerBasedRlEnv) -> tuple[str, dict[str, float]]:
    """Serialize the compiled task settings applied by mjlab to the scene."""
    spec = env.scene.spec.copy()
    source = env.sim.mj_model.opt
    for field in (
        "timestep",
        "integrator",
        "jacobian",
        "cone",
        "solver",
        "impratio",
        "iterations",
        "tolerance",
        "ls_iterations",
        "ls_tolerance",
        "ccd_iterations",
        "disableflags",
        "enableflags",
    ):
        setattr(spec.option, field, getattr(source, field))
    spec.option.gravity[:] = source.gravity
    spec.compile()

    # MjSpec attaches an empty terrain default that 3.5 writes as a nested <default/>.
    root = ET.fromstring(spec.to_xml())
    for parent in root.iter("default"):
        for child in parent.findall("default"):
            if not child.attrib and len(child) == 0:
                parent.remove(child)
    xml = ET.tostring(root, encoding="unicode")
    restored = mujoco.MjModel.from_xml_string(xml)
    original = env.sim.mj_model
    errors = {}
    for field in (
        "qpos0",
        "jnt_axis",
        "jnt_range",
        "dof_armature",
        "dof_damping",
        "dof_frictionloss",
        "body_mass",
        "body_inertia",
        "body_ipos",
        "geom_size",
        "geom_pos",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
    ):
        expect, actual = getattr(original, field), getattr(restored, field)
        np.testing.assert_allclose(actual, expect, atol=5e-5, rtol=5e-6, err_msg=field)
        errors[field] = float(np.max(np.abs(actual - expect), initial=0))
    return xml, errors


def copy_array(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def capture_fixtures(env, vec, policy, levels: list[int]) -> tuple[dict, dict]:
    """Capture identical starts and independent policy/controller reference data."""
    terrain = env.scene.terrain
    sim = env.sim.data
    starts = []
    fixtures = []
    for level in levels:
        terrain.terrain_levels[:] = level
        terrain.env_origins[:] = terrain.terrain_origins[level, terrain.terrain_types]
        obs, _ = vec.reset()
        alive = torch.ones(env.num_envs, dtype=torch.bool)
        starts.append(
            {
                "qpos": copy_array(sim.qpos),
                "qvel": copy_array(sim.qvel),
                "origin": copy_array(terrain.env_origins),
                "command": copy_array(env.command_manager.get_command("twist")),
            }
        )
        for step in range(FIXTURE_STEPS[-1] + 1):
            action = policy(obs)
            if step in FIXTURE_STEPS:
                fixture = {
                    "step": np.full(env.num_envs, step),
                    "qpos": copy_array(sim.qpos),
                    "qvel": copy_array(sim.qvel),
                    "command": copy_array(env.command_manager.get_command("twist")),
                    "previous_action": copy_array(env.action_manager.action),
                    "obs": copy_array(obs["actor"]),
                    "action": copy_array(action),
                }
            obs, _, dones, _ = vec.step(action)
            alive &= ~dones.bool()
            if step in FIXTURE_STEPS:
                fixture["ctrl"] = copy_array(sim.ctrl)
                mask = alive.numpy()
                fixtures.append({key: value[mask] for key, value in fixture.items()})
        print(f"Captured reference level {level}", flush=True)
    return (
        {key: np.stack([row[key] for row in starts]) for key in starts[0]},
        {key: np.concatenate([row[key] for row in fixtures]) for key in fixtures[0]},
    )


def bundle_metadata(env, cfg, agent_cfg, args, task, names: list[str]) -> dict:
    model = env.sim.mj_model
    robot = env.scene["robot"]
    scan = env.scene["terrain_scan"].cfg
    actor_terms = cfg.observations["actor"].terms
    action = env.action_manager.get_term("joint_pos")
    offset = copy_array(action.offset)
    if not isinstance(action.scale, float) or not np.all(offset == offset[0]):
        raise ValueError(
            "Expected one action scale and the same joint target offset in every world"
        )
    twist = cfg.commands["twist"]
    contact_params = cfg.terminations["illegal_contact"].params
    contact_name = contact_params["sensor_name"]
    contact_sensors = [
        name
        for i in range(model.nsensor)
        if (name := model.sensor(i).name).startswith(contact_name + "_") and name.endswith("_force")
    ]
    if not contact_sensors or env.scene[contact_name].cfg.history_length != cfg.decimation:
        raise ValueError("Expected illegal contact force history over one control step")
    with args.checkpoint.open("rb") as checkpoint:
        checksum = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    return {
        "task": task.TASK_ID,
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": checksum,
        "source_mujoco": mujoco.__version__,
        "seed": args.seed,
        "levels": args.levels,
        "terrain_columns": names,
        "world_columns": copy_array(env.scene.terrain.terrain_types).tolist(),
        "samples_per_column": args.samples_per_column,
        "actor_terms": list(env.observation_manager.active_terms["actor"]),
        "actor_term_dims": env.observation_manager.group_obs_term_dim["actor"],
        "joint_names": [f"robot/{name}" for name in robot.joint_names],
        "default_joint_pos": copy_array(robot.data.default_joint_pos[0]).tolist(),
        "default_joint_vel": copy_array(robot.data.default_joint_vel[0]).tolist(),
        "action_joints": [f"robot/{name}" for name in action.target_names],
        "action_offset": offset[0].tolist(),
        "action_scale": action.scale,
        "clip_actions": agent_cfg.clip_actions,
        "model_joint_order": [model.joint(i).name for i in range(model.njnt)],
        "actuator_names": [model.actuator(i).name for i in range(model.nu)],
        "scan_body": f"robot/{scan.frame.name}",
        "scan_geom_groups": scan.include_geom_groups,
        "scan_excludes_body": scan.exclude_parent_body,
        "scan_max_distance_m": scan.max_distance,
        "scan_scale": actor_terms["height_scan"].scale,
        "period_s": actor_terms["phase"].params["period"],
        "heading_target_rad": twist.ranges.heading[0],
        "heading_gain": twist.heading_control_stiffness,
        "yaw_rate_limits": twist.ranges.ang_vel_z,
        "physics_timestep_s": model.opt.timestep,
        "integrator": mujoco.mjtIntegrator(model.opt.integrator).name,
        "control_timestep_s": env.step_dt,
        "steps": int(env.max_episode_length) - 2,
        "fall_angle_rad": cfg.terminations["fell_over"].params["limit_angle"],
        "illegal_contact_sensors": contact_sensors,
        "illegal_contact_force_n": contact_params["force_threshold"],
        "promote_m": task.PROMOTE_M,
        "demote_m": task.DEMOTE_M,
        "observation_noise": False,
        "disabled_events": list(DISABLED_EVENTS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--samples-per-column", type=int, default=3)
    parser.add_argument("--levels", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()
    if args.samples_per_column < 1 or not set(args.levels) <= set(range(10)):
        parser.error("Use at least one sample per column and levels 0 through 9")
    if args.out.exists():
        parser.error(f"Bundle directory already exists: {args.out}")

    torch.set_num_threads(1)
    task = load_task()
    cfg = evaluation_config(task.TASK_ID, 10 * args.samples_per_column, "straight")
    generator = cfg.scene.terrain.terrain_generator
    cfg.seed = generator.seed = args.seed
    for name in DISABLED_EVENTS:
        cfg.events.pop(name)
    cfg.observations["actor"].enable_corruption = False
    ray = next(sensor for sensor in cfg.scene.sensors if sensor.name == "terrain_scan")
    if ray.ray_alignment != "yaw" or cfg.commands["twist"].rel_heading_envs != 1.0:
        raise ValueError("Expected yaw-aligned height scan and heading control in all worlds")
    agent_cfg = load_rl_cfg(task.TASK_ID)

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = load_runner_cls(task.TASK_ID)(vec, asdict(agent_cfg), device="cpu")
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location="cpu")
        policy = runner.get_inference_policy(device="cpu")
        arrays = actor_arrays(policy)
        offsets, directions = ray.pattern.generate_rays(env.sim.mj_model, "cpu")
        if not np.all(copy_array(directions) == [0, 0, -1]):
            raise ValueError("Expected vertical height-scan rays")
        arrays["ray_offsets"] = copy_array(offsets)
        meta = bundle_metadata(env, cfg, agent_cfg, args, task, column_names(generator))
        xml, meta["xml_roundtrip_max_abs_error"] = scene_xml(env)
        with torch.no_grad():
            starts, fixtures = capture_fixtures(env, vec, policy, args.levels)
    finally:
        env.close()

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "scene.xml").write_text(xml, encoding="utf-8")
    (args.out / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savez_compressed(args.out / "actor.npz", **arrays)
    np.savez_compressed(args.out / "starts.npz", **starts)
    np.savez_compressed(args.out / "fixtures.npz", **fixtures)


if __name__ == "__main__":
    main()
