"""Is there any floor under the open-stair treads? Cast rays down from just below each tread."""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import mujoco
import numpy as np

spec_ = importlib.util.spec_from_file_location("t", Path(__file__).with_name("rbq_rb1_stairs.py"))
t = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(t)
cfg = t.stairs_env_cfg(play=False)
subs = cfg.scene.terrain.terrain_generator.sub_terrains
size = cfg.scene.terrain.terrain_generator.size
for level in (2, 6, 9):
    d = (level + 0.5) / 10
    for name in ("open_stairs_up", "open_stairs_down", "stairs_up"):
        sub = subs[name]
        sub.size = size
        s = mujoco.MjSpec()
        s.worldbody.add_body(name="terrain")
        out = sub.function(d, s, np.random.default_rng(0))
        m = s.compile()
        dd = mujoco.MjData(m)
        mujoco.mj_forward(m, dd)
        o = np.asarray(out.origin, float)
        rows = []
        for r in np.arange(0.5, 3.6, 0.25):
            gid = np.array([-1], np.int32)
            top = 5.0 - mujoco.mj_ray(
                m, dd, np.array([o[0] + r, o[1], 5.0]), np.array([0, 0, -1.0]), None, 1, -1, gid
            )
            # start 1 cm under the surface found from above, look further down
            below = mujoco.mj_ray(
                m,
                dd,
                np.array([o[0] + r, o[1], top - 0.04]),
                np.array([0, 0, -1.0]),
                None,
                1,
                -1,
                gid,
            )
            rows.append(f"{r:.2f}:{top - o[2]:+.2f}/{'void' if below < 0 else f'{below:.2f}'}")
        print(
            f"L{level} {name:17s} (x from origin: tread height rel. origin / gap below tread)",
            " ".join(rows),
        )
