"""Render a checkpoint on each stair type at one terrain level, as one contact sheet.

Rows are the four stair sub-terrains (real direction: up = robot climbs when leaving the
spawn platform), columns are moments of one episode under a 0.6 m/s forward command. A
robot whose episode ended keeps its last frame, marked with the end time and cause. A few
robots per column are scored as well (survival, base height change, distance), so the
sheet is a quick visual check between full `eval_stairs.py` runs.

Few robots are simulated on purpose: the run shares the GPU with training, and a
1000-robot evaluation next to a 2048-robot training run filled the 12 GB card and stalled
both (2026-11-10).

Run from the unitree_rl_mjlab checkout (MUJOCO_GL=egl):
  python <this file> <checkpoint.pt> --level 2 --out sheet.png
"""

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.viewer.viewer_config import ViewerConfig  # noqa: E402

EVAL_FILE = Path(__file__).with_name("eval_stairs.py")
TYPES = ("stairs_up", "stairs_down", "open_stairs_up", "open_stairs_down")
LABELS = {
    "stairs_up": "닫힌 계단 오르막",
    "stairs_down": "닫힌 계단 내리막",
    "open_stairs_up": "열린 계단 오르막",
    "open_stairs_down": "열린 계단 내리막",
}
SHOTS_S = (0.0, 2.0, 4.0, 7.0, 12.0)
KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--device", default="cuda:0", help="cpu keeps the GPU free for training")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ev = load(EVAL_FILE, "eval_stairs")
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
    viewer = cfg.viewer
    viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
    viewer.entity_name = "robot"
    viewer.distance, viewer.elevation, viewer.azimuth = 2.6, -18.0, 90.0
    viewer.width, viewer.height, viewer.max_extra_envs = 480, 320, 0

    # Record which termination ends each robot's first episode.
    ended_by: dict[int, str] = {}
    for name, term in cfg.terminations.items():
        if name == "time_out":
            continue

        def wrapped(env, _f=term.func, _n=name, **kw):
            done = _f(env, **kw)
            for i in torch.nonzero(done).flatten().tolist():
                ended_by.setdefault(i, _n)
            return done

        term.func = wrapped

    agent_cfg = load_rl_cfg(task_id)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")
    vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(task_id)(vec, asdict(agent_cfg), device=args.device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    renderer = env._offline_renderer
    model = renderer._model
    model.geom_group[model.geom_group > 2] = 2  # training model has collision geoms only

    terrain = env.scene.terrain
    robot = env.scene["robot"]
    names = ev.column_names(generator)
    types = terrain.terrain_types
    terrain.terrain_levels[:] = args.level
    terrain.env_origins[:] = terrain.terrain_origins[args.level, types]
    members = {
        t: torch.nonzero(
            torch.isin(
                types, torch.tensor([i for i, n in enumerate(names) if n == t], device=types.device)
            )
        ).flatten()
        for t in TYPES
    }
    cams = {t: int(members[t][0]) for t in TYPES}

    def shot(env_id: int):
        renderer._cfg.env_idx = env_id
        renderer.update(env.sim.data)
        return renderer.render().copy()

    obs, _ = vec.reset()
    dt = env.step_dt
    steps = int(env.max_episode_length) - 2
    shot_steps = {round(s / dt): s for s in SHOTS_S}
    alive = torch.ones(args.num_envs, dtype=torch.bool, device=types.device)
    end_step = torch.full((args.num_envs,), -1, device=types.device)
    last = {t: None for t in TYPES}
    frames = {t: {} for t in TYPES}
    origin_z = robot.data.root_link_pos_w[:, 2].clone()
    final_pos = robot.data.root_link_pos_w.clone()
    with torch.no_grad():
        for step in range(steps):
            for t, e in cams.items():
                if alive[e] and (step % 5 == 0 or step in shot_steps):
                    last[t] = (step, shot(e))
                if step in shot_steps:
                    frames[t][shot_steps[step]] = last[t]
            obs, _, dones, _ = vec.step(policy(obs))
            newly = dones.bool() & alive
            end_step[newly] = step
            alive &= ~dones.bool()
            final_pos[alive] = robot.data.root_link_pos_w[alive]
    final_pos[alive] = robot.data.root_link_pos_w[alive]

    origin_xy = env.scene.env_origins[:, :2]
    stats = {}
    for t in TYPES:
        ids = members[t]
        ok = alive[ids]
        dz = final_pos[ids, 2] - origin_z[ids]
        dist = torch.norm(final_pos[ids, :2] - origin_xy[ids], dim=1)
        causes = {}
        for i in ids.tolist():
            if not bool(alive[i]):
                c = ended_by.get(i, "?")
                causes[c] = causes.get(c, 0) + 1
        stats[t] = {
            "robots": len(ids),
            "survived": round(float(ok.float().mean()), 2),
            "dz_median_m": round(float(dz.median()), 3),
            "distance_median_m": round(float(dist.median()), 2),
            "ended_by": causes,
        }

    if KOREAN_FONT.exists():
        font_manager.fontManager.addfont(str(KOREAN_FONT))
        plt.rcParams["font.family"] = "Malgun Gothic"
    rows, cols = len(TYPES), len(SHOTS_S)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.55 * rows + 0.6), dpi=90)
    for r, t in enumerate(TYPES):
        e = cams[t]
        for c, s in enumerate(SHOTS_S):
            ax = axes[r, c]
            ax.axis("off")
            got = frames[t].get(s)
            if got is None:
                continue
            step, img = got
            ax.imshow(img)
            ended = int(end_step[e]) >= 0 and step < round(s / dt)
            note = f"  끝남 {int(end_step[e]) * dt:.1f}s ({ended_by.get(e, '?')})" if ended else ""
            ax.set_title(f"{s:.0f}s{note}", fontsize=9, color="#c5221f" if ended else "black")
        st = stats[t]
        axes[r, 0].text(
            -0.04,
            0.5,
            f"{LABELS[t]}\n생존 {st['survived']:.0%}\n높이 {st['dz_median_m']:+.2f}m\n"
            f"거리 {st['distance_median_m']:.1f}m\n({st['robots']}대)",
            transform=axes[r, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )
    level_note = f"단 {task.STEP_HEIGHT_RANGE[1] * (args.level + 0.5) / 10 * 100:.0f}cm 안팎"
    fig.suptitle(
        f"{Path(args.checkpoint).parent.name}/{Path(args.checkpoint).name} · 지형 {args.level}단계"
        f"({level_note}) · 0.6m/s 직진 명령 · 높이는 출발 대비 몸통 높이 변화",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.09, 0, 1, 0.97))
    out = Path(args.out)
    fig.savefig(out, facecolor="white")
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "level": args.level,
                "camera_envs": cams,
                "stats": stats,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(json.dumps(stats, ensure_ascii=False))
    env.close()


if __name__ == "__main__":
    main()
