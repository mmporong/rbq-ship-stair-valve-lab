"""Measure a trained RBQ10RB1-Stairs policy per terrain type and level.

All robots are pinned to one terrain level (the curriculum term is removed), run for one
episode, and scored at the last step before the time-out:

- distance: horizontal distance from the spawn origin;
- promote / demote: the task's `stair_levels` rule (distance over PROMOTE_M; distance under
  DEMOTE_M while commanded to move). Results saved before that rule existed (version 1)
  used mjlab's terrain_levels_vel: over 4 m, and under |command| x episode x 0.5;
- survived: no termination during the episode;
- dz: base height above the spawn origin at the end (about 0.53 m when standing on
  the spawn level; more = climbed, less = descended);
- speed, tracking_error: episode means of the body-frame horizontal base speed and of its
  distance from the commanded body-frame velocity.

`--command straight` replaces the random twist command by 0.6 m/s forward on a fixed
world heading, to test stair climbing apart from the command sampling.

Run from the unitree_rl_mjlab checkout:
  python <path to this file> <checkpoint.pt> --command train --out result.json
"""

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

TASK_FILE = Path(__file__).with_name("rbq_rb1_stairs.py")
LEVELS = (0, 2, 4, 6, 9)


def load_task():
    spec = importlib.util.spec_from_file_location("rbq_rb1_stairs", TASK_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def column_names(generator) -> list[str]:
    """Sub-terrain name per column, same allocation as mjlab's curriculum generator."""
    names = list(generator.sub_terrains)
    weights = torch.tensor([c.proportion for c in generator.sub_terrains.values()])
    edges = torch.cumsum(weights / weights.sum(), 0)
    return [
        names[int(torch.nonzero(i / generator.num_cols + 0.001 < edges)[0])]
        for i in range(generator.num_cols)
    ]


def _share(flags: torch.Tensor) -> torch.Tensor:
    return flags.float().mean()


def _stat(values: torch.Tensor, fn, digits: int) -> float | None:
    """`fn(values)` rounded, or None when no robot survived."""
    return round(float(fn(values)), digits) if values.numel() else None


def evaluation_config(task_id: str, num_envs: int, command: str):
    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = num_envs
    generator = cfg.scene.terrain.terrain_generator
    if generator is not None:
        generator.num_rows = generator.num_cols = 10
    cfg.curriculum.pop("terrain_levels", None)
    if command == "straight":
        twist = cfg.commands["twist"]
        twist.ranges.lin_vel_x = (0.6, 0.6)
        twist.ranges.lin_vel_y = (0.0, 0.0)
        twist.ranges.heading = (0.0, 0.0)
        twist.rel_standing_envs = 0.0
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--task", choices=("stairs", "flat"), default="stairs")
    parser.add_argument("--command", choices=("train", "straight"), default="train")
    parser.add_argument("--num-envs", type=int, default=1000)
    parser.add_argument("--levels", type=int, nargs="+", default=list(LEVELS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    task = load_task()
    task_id = task.TASKS[args.task][0]
    cfg = evaluation_config(task_id, args.num_envs, args.command)
    generator = cfg.scene.terrain.terrain_generator
    agent_cfg = load_rl_cfg(task_id)

    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id)
    runner = runner_cls(vec, asdict(agent_cfg), device="cuda:0")
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")

    terrain = env.scene.terrain
    robot = env.scene["robot"]
    if generator is None:  # plane: one "level", every robot on flat ground
        names, levels = ["plane"], (None,)
        types = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
    else:
        names, levels = column_names(generator), args.levels
        types = terrain.terrain_types
    steps = int(env.max_episode_length) - 2
    result = {
        "checkpoint": args.checkpoint,
        "task": args.task,
        "command": args.command,
        "num_envs": args.num_envs,
        "steps": steps,
        "promote_m": task.PROMOTE_M,
        "demote_m": task.DEMOTE_M,
        "levels": {},
    }
    for level in levels:
        if level is not None:
            terrain.terrain_levels[:] = level
            terrain.env_origins[:] = terrain.terrain_origins[level, types]
        obs, _ = vec.reset()
        alive = torch.ones(args.num_envs, dtype=torch.bool, device=types.device)
        speed = torch.zeros(args.num_envs, device=types.device)
        error = torch.zeros(args.num_envs, device=types.device)
        with torch.no_grad():  # inference_mode tensors break the next reset's in-place writes
            for _ in range(steps):
                target = env.command_manager.get_command("twist")[:, :2].clone()
                obs, _, dones, _ = vec.step(policy(obs))
                alive &= ~dones.bool()
                velocity = robot.data.root_link_lin_vel_b[:, :2]
                speed += torch.norm(velocity, dim=1) / steps
                error += torch.norm(velocity - target, dim=1) / steps
        origin = env.scene.env_origins
        offset = robot.data.root_link_pos_w - origin
        distance = torch.norm(offset[:, :2], dim=1)
        command = torch.norm(env.command_manager.get_command("twist")[:, :2], dim=1)
        demote = (distance < task.DEMOTE_M) & (command > 0.1)
        per_type = {}
        for name in dict.fromkeys(names):
            cols = [i for i, n in enumerate(names) if n == name]
            mask = torch.isin(types, torch.tensor(cols, device=types.device))
            ok = mask & alive
            per_type[name] = {
                "envs": int(mask.sum()),
                "survived": round(float(ok.sum() / mask.sum()), 3),
                "distance_median_m": _stat(distance[ok], torch.median, 2),
                "promote_share": _stat(distance[ok] > task.PROMOTE_M, _share, 3),
                "demote_share": _stat(demote[ok], _share, 3),
                "dz_median_m": _stat(offset[ok, 2], torch.median, 3),
                "command_median_mps": _stat(command[ok], torch.median, 2),
                "speed_median_mps": _stat(speed[ok], torch.median, 2),
                "tracking_error_median_mps": _stat(error[ok], torch.median, 2),
            }
        result["levels"][str(level if level is not None else "plane")] = per_type
        print(level, json.dumps(per_type), flush=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
