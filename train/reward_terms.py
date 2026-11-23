"""Per-terrain mean per-step reward terms and foot height above z=0, at fixed levels."""

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

ev_spec = importlib.util.spec_from_file_location("ev", Path(__file__).with_name("eval_stairs.py"))
ev = importlib.util.module_from_spec(ev_spec)
ev_spec.loader.exec_module(ev)
task = ev.load_task()
tid = task.TASKS["stairs"][0]
ckpt, levels, out = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
cfg = load_env_cfg(tid)
cfg.scene.num_envs = 200
g = cfg.scene.terrain.terrain_generator
g.num_rows = g.num_cols = 10
cfg.curriculum.pop("terrain_levels", None)
tw = cfg.commands["twist"]
tw.ranges.lin_vel_x = (0.6, 0.6)
tw.ranges.lin_vel_y = (0.0, 0.0)
tw.ranges.heading = (0.0, 0.0)
tw.rel_standing_envs = 0.0
agent = load_rl_cfg(tid)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
vec = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = load_runner_cls(tid)(vec, asdict(agent), device="cuda:0")
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
rm = env.reward_manager
names = list(rm.active_terms)
terrain = env.scene.terrain
types = terrain.terrain_types
cols = ev.column_names(g)
robot = env.scene["robot"]
site_ids = rm._term_cfgs[names.index("foot_clearance")].params["asset_cfg"].site_ids
res = {}
for L in levels:
    terrain.terrain_levels[:] = L
    terrain.env_origins[:] = terrain.terrain_origins[L, types]
    obs, _ = vec.reset()
    foot_z0 = robot.data.site_pos_w[:, site_ids, 2].mean(1).clone()
    ac = rm._term_cfgs[names.index("foot_clearance")].params["asset_cfg"]
    rel0 = (
        task._foot_height_above_ground(env, ac).mean(1).clone()
        if hasattr(task, "_foot_height_above_ground")
        else foot_z0
    )
    acc = torch.zeros(200, len(names), device="cuda:0")
    n = 300
    with torch.no_grad():
        for _ in range(n):
            obs, _, _, _ = vec.step(policy(obs))
            acc += rm._step_reward / n
    res[L] = {}
    for t in dict.fromkeys(cols):
        m = torch.isin(
            types, torch.tensor([i for i, c in enumerate(cols) if c == t], device=types.device)
        )
        row = {
            k: round(float(acc[m, j].mean()), 4)
            for j, k in enumerate(names)
            if k
            in (
                "track_linear_velocity",
                "foot_clearance",
                "stand_still",
                "pose",
                "foot_gait",
                "shin_contact",
                "is_terminated",
            )
        }
        row["foot_z_at_spawn"] = round(float(foot_z0[m].mean()), 3)
        row["foot_above_ground_at_spawn"] = round(float(rel0[m].mean()), 3)
        res[L][t] = row
    print(L, json.dumps(res[L]), flush=True)
Path(out).write_text(
    json.dumps(
        {
            "checkpoint": ckpt,
            "note": "_step_reward per step mean over 6 s, straight 0.6 m/s",
            "levels": res,
        },
        indent=2,
    )
)
env.close()
