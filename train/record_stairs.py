"""Record one robot on one stair type as an mp4, to keep failure modes as video.

Robots are pinned to one terrain level under a 0.6 m/s forward command. A few robots of
the chosen stair type are rendered every other step; with --pick ended the clip follows
the first of them whose episode terminates (and stops there), otherwise the first one.

--task-file loads another version of the task, so a failure seen with an earlier version
can be recorded with that version's checkpoint. Extract it next to this file so that its
model path still resolves, e.g.
  git show ef40e49:train/rbq_rb1_stairs.py > train/_task_v5.py

Run from the unitree_rl_mjlab checkout (MUJOCO_GL=egl; CUDA_VISIBLE_DEVICES= for CPU):
  python <this file> <checkpoint.pt> --level 6 --type open_stairs_up --out clip.mp4
"""

import argparse
import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.viewer.viewer_config import ViewerConfig  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")


def column_names(generator) -> list[str]:
    """Sub-terrain name per column, same allocation as mjlab's curriculum generator."""
    names = list(generator.sub_terrains)
    weights = torch.tensor([c.proportion for c in generator.sub_terrains.values()])
    edges = torch.cumsum(weights / weights.sum(), 0)
    return [
        names[int(torch.nonzero(i / generator.num_cols + 0.001 < edges)[0])]
        for i in range(generator.num_cols)
    ]


def caption(frame: np.ndarray, lines: list[str], font) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    y = 8
    for line in lines:
        draw.text(
            (10, y), line, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0)
        )
        y += font.size + 6
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--task-file", default=str(Path(__file__).with_name("rbq_rb1_stairs.py")))
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--type", required=True, help="sub-terrain key in that task version")
    parser.add_argument("--pick", choices=("ended", "first"), default="ended")
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--seconds", type=float, default=14.0)
    parser.add_argument("--title", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("task_version", args.task_file)
    task = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task)
    task_id = task.TASKS["stairs"][0] if hasattr(task, "TASKS") else task.TASK_ID

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
    viewer = cfg.viewer
    viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
    viewer.entity_name = "robot"
    viewer.distance, viewer.elevation, viewer.azimuth = 2.6, -18.0, 90.0
    viewer.width, viewer.height, viewer.max_extra_envs = 768, 432, 0

    agent_cfg = load_rl_cfg(task_id)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")
    vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(task_id)(vec, asdict(agent_cfg), device=args.device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    renderer = env._offline_renderer
    renderer._model.geom_group[renderer._model.geom_group > 2] = 2

    terrain = env.scene.terrain
    names = column_names(generator)
    types = terrain.terrain_types
    terrain.terrain_levels[:] = args.level
    terrain.env_origins[:] = terrain.terrain_origins[args.level, types]
    cols = torch.tensor([i for i, n in enumerate(names) if n == args.type], device=types.device)
    candidates = torch.nonzero(torch.isin(types, cols)).flatten()[: args.candidates].tolist()

    dt = env.step_dt
    steps = min(int(args.seconds / dt), int(env.max_episode_length) - 2)
    frames: dict[int, list[np.ndarray]] = {e: [] for e in candidates}
    end_step: dict[int, int] = {}
    obs, _ = vec.reset()
    with torch.no_grad():
        for step in range(steps):
            if step % 2 == 0:
                for e in candidates:
                    if e not in end_step:
                        renderer._cfg.env_idx = e
                        renderer.update(env.sim.data)
                        frames[e].append(renderer.render().copy())
            obs, _, dones, _ = vec.step(policy(obs))
            for e in candidates:
                if e not in end_step and bool(dones[e]):
                    end_step[e] = step
            if args.pick == "ended" and end_step and step > min(end_step.values()) + 10:
                break

    ended = [e for e in candidates if e in end_step]
    chosen = ended[0] if (args.pick == "ended" and ended) else candidates[0]
    clip = frames[chosen]
    clip += [clip[-1]] * 25  # hold the last frame for 1 s
    font = ImageFont.truetype(str(FONT), 22) if FONT.exists() else ImageFont.load_default()
    note = f"{end_step[chosen] * dt:.1f}초에 에피소드 종료" if chosen in end_step else "종료 없음"
    lines = [
        args.title,
        f"{Path(args.checkpoint).parent.name[-3:].strip('_')} "
        f"{Path(args.checkpoint).stem} · {args.level}단계 {args.type} · {note}",
    ]
    out = [caption(f, [ln for ln in lines if ln], font) for f in clip]
    imageio.mimwrite(
        args.out, out, fps=round(1 / (2 * dt)), codec="libx264", quality=7, pixelformat="yuv420p"
    )
    print(f"env {chosen} frames {len(clip)} ended {sorted(end_step.items())} -> {args.out}")
    env.close()


if __name__ == "__main__":
    main()
