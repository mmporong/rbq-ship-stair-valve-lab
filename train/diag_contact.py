"""Which non-foot geoms end episodes at one terrain level, plus snapshots of one robot."""

import importlib.util
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer.viewer_config import ViewerConfig

ckpt, level, out_dir = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
DEV = __import__("os").environ.get("DEVICE", "cuda:0")
TRAIN = Path(__file__).parent
spec = importlib.util.spec_from_file_location("ev", TRAIN / "eval_stairs.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
task = ev.load_task()
task_id = task.TASKS["stairs"][0]

cfg = load_env_cfg(task_id)
cfg.scene.num_envs = int(__import__("os").environ.get("NENV", "200"))
gen = cfg.scene.terrain.terrain_generator
gen.num_rows = gen.num_cols = 10
cfg.curriculum.pop("terrain_levels", None)
tw = cfg.commands["twist"]
tw.ranges.lin_vel_x = (0.6, 0.6)
tw.ranges.lin_vel_y = (0.0, 0.0)
tw.ranges.heading = (0.0, 0.0)
tw.rel_standing_envs = 0.0
cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
cfg.viewer.entity_name = "robot"
cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 2.4, -12.0, 90.0
cfg.viewer.width, cfg.viewer.height = 640, 400
cfg.viewer.max_extra_envs = 0

term = cfg.terminations["illegal_contact"]
orig = term.func
hits = {}  # env -> list of geom names at its first illegal contact


def wrapped(env, **kw):
    done = orig(env, **kw)
    fh = env.scene[kw["sensor_name"]].data.force_history
    over = (torch.norm(fh, dim=-1) > kw.get("force_threshold", 10.0)).any(-1)  # [B, N]
    for i in torch.nonzero(done).flatten().tolist():
        if i not in hits:
            hits[i] = [NAMES[j] for j in torch.nonzero(over[i]).flatten().tolist()]
    return done


term.func = wrapped

agent = load_rl_cfg(task_id)
env = ManagerBasedRlEnv(cfg=cfg, device=DEV, render_mode="rgb_array")
sensor = env.scene["nonfoot_ground_touch"]
NAMES = list(dict.fromkeys(s.primary_name for s in sensor._slots))
vec = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = load_runner_cls(task_id)(vec, asdict(agent), device=DEV)
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=DEV)
policy = runner.get_inference_policy(device=DEV)

terrain = env.scene.terrain
names = ev.column_names(gen)
types = terrain.terrain_types
terrain.terrain_levels[:] = level
terrain.env_origins[:] = terrain.terrain_origins[level, types]
cam_type = sys.argv[4] if len(sys.argv) > 4 else "stairs_up"
cam_env = int(torch.nonzero(types == names.index(cam_type))[0])
R = env._offline_renderer
R._cfg.env_idx = cam_env
m = R._model
print("geom groups", sorted(set(m.geom_group.tolist())))
m.geom_group[m.geom_group > 2] = 2  # show collision primitives (render model only)


def shot():
    R.update(env.sim.data)
    return R.render().copy()


obs, _ = vec.reset()
steps = int(env.max_episode_length) - 2
ring = []
cam_done = None
with torch.no_grad():
    for t in range(steps):
        if cam_done is None and t % 10 == 0:
            ring = (ring + [(t, shot())])[-8:]
        obs, _, dones, _ = vec.step(policy(obs))
        if cam_done is None and bool(dones[cam_env]):
            cam_done = t
            ring = (ring + [(t, shot())])[-8:]  # right after reset: robot already respawned
per_type = {}
for name in dict.fromkeys(names):
    cols = [i for i, n in enumerate(names) if n == name]
    envs = (
        torch.nonzero(torch.isin(types, torch.tensor(cols, device=types.device))).flatten().tolist()
    )
    c = Counter(g for e in envs if e in hits for g in hits[e])
    per_type[name] = {
        "envs": len(envs),
        "illegal_first": sum(e in hits for e in envs),
        "geoms": c.most_common(8),
    }
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / f"contact_L{level}.json").write_text(
    json.dumps(
        {
            "checkpoint": ckpt,
            "level": level,
            "command": "straight 0.6",
            "geoms": NAMES,
            "per_type": per_type,
            "camera": {"type": cam_type, "env": cam_env, "done_step": cam_done},
        },
        indent=2,
        ensure_ascii=False,
    )
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

frames = ring[:-1] if cam_done is not None else ring
n = len(frames)
cols = 4
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 4.1 * rows), dpi=90)
for ax in np.atleast_1d(axes).flat:
    ax.axis("off")
for ax, (t, img) in zip(np.atleast_1d(axes).flat, frames):
    ax.imshow(img)
    ax.set_title(f"t = {t * env.step_dt:.1f} s", fontsize=12)
fig.suptitle(
    f"level {level} {cam_type} (env {cam_env}), {Path(ckpt).name}, 0.6 m/s forward"
    + (f" - episode ended at {cam_done * env.step_dt:.1f} s" if cam_done is not None else ""),
    fontsize=13,
)
fig.tight_layout()
fig.savefig(out_dir / f"frames_L{level}_{cam_type}.png", facecolor="white")
print(json.dumps(per_type, ensure_ascii=False), cam_done)
env.close()
