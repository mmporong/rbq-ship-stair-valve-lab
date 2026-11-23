"""How often feet are under a tread (in a riser gap) per stair type, and whether robots fall.

Adds one upward ray per foot site; a hit within UNDER_M above the foot means the foot is
under a board. Robots are pinned to one level and run one episode under a 0.6 m/s forward
command; the foot-under share is counted until each robot's first termination.

Run from the unitree_rl_mjlab checkout (CUDA_VISIBLE_DEVICES= for CPU):
  python <this file> <checkpoint.pt> <level> <out.json> [--device cpu] [--num-envs 60]
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
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

UNDER_M = 0.30
TYPES = ("stairs_up", "stairs_down", "open_stairs_up", "open_stairs_down")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("level", type=int)
    parser.add_argument("out")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=60)
    args = parser.parse_args()

    ev = load(Path(__file__).with_name("eval_stairs.py"), "eval_stairs")
    task = ev.load_task()
    task_id = task.TASKS["stairs"][0]
    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = args.num_envs
    generator = cfg.scene.terrain.terrain_generator
    generator.num_rows = generator.num_cols = 10
    cfg.curriculum.pop("terrain_levels", None)
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (0.6, 0.6)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.heading = (0.0, 0.0)
    twist.rel_standing_envs = 0.0
    cfg.scene.sensors = (
        *cfg.scene.sensors,
        *(
            RayCastSensorCfg(
                name=f"above_{site}",
                frame=ObjRef(type="site", name=site, entity="robot"),
                ray_alignment="world",
                pattern=GridPatternCfg(size=(0.0, 0.0), direction=(0.0, 0.0, 1.0)),
                max_distance=UNDER_M,
            )
            for site in task.FOOT_SITES
        ),
    )

    agent_cfg = load_rl_cfg(task_id)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(task_id)(vec, asdict(agent_cfg), device=args.device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    terrain = env.scene.terrain
    names = ev.column_names(generator)
    types = terrain.terrain_types
    terrain.terrain_levels[:] = args.level
    terrain.env_origins[:] = terrain.terrain_origins[args.level, types]

    n = args.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    under_steps = torch.zeros(n, device=env.device)
    live_steps = torch.zeros(n, device=env.device)
    ever_under = torch.zeros(n, dtype=torch.bool, device=env.device)
    obs, _ = vec.reset()
    with torch.no_grad():
        for _ in range(int(env.max_episode_length) - 2):
            hits = torch.stack(
                [env.scene[f"above_{s}"].data.distances[:, 0] >= 0 for s in task.FOOT_SITES], 1
            )
            any_under = hits.any(dim=1)
            under_steps += (any_under & alive).float()
            live_steps += alive.float()
            ever_under |= any_under & alive
            obs, _, dones, _ = vec.step(policy(obs))
            alive &= ~dones.bool()

    result = {"checkpoint": args.checkpoint, "level": args.level, "under_m": UNDER_M, "types": {}}
    for t in TYPES:
        cols = torch.tensor([i for i, c in enumerate(names) if c == t], device=types.device)
        m = torch.isin(types, cols)
        fell, kept = m & ~alive, m & alive

        def share(mask):
            steps = live_steps[mask].sum()
            return round(float(under_steps[mask].sum() / steps), 3) if steps > 0 else None

        result["types"][t] = {
            "robots": int(m.sum()),
            "ended": int(fell.sum()),
            "foot_under_step_share_ended": share(fell),
            "foot_under_step_share_survived": share(kept),
            "ever_under_ended": int((ever_under & fell).sum()),
            "ever_under_survived": int((ever_under & kept).sum()),
        }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["types"]))
    env.close()


if __name__ == "__main__":
    main()
