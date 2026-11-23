"""Search an arm stow pose for stairs: collision-free, gravity torque within a share of the
arm limits at 0 and 60 deg body pitch, and the smallest arm centre-of-mass height plus half its
fore-aft offset from the base origin.

Usage: python tools/find_stow.py <d1|z1|rb1>
"""

import itertools
import sys

import mujoco
import numpy as np

from go2_wall.robot import ARM_JOINTS, build_robot
from go2_wall.statics import torque_limits

SHARE = 0.6


def main() -> None:
    arm = sys.argv[1]
    model = build_robot(arm=arm).compile()
    data = mujoco.MjData(model)
    bodies = [b for b in range(model.nbody) if model.body(b).name.startswith(("arm_", "RB1_"))]
    geoms = {
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] in bodies and (model.geom_contype[g] or model.geom_conaffinity[g])
    }
    dofs = [model.joint(j).dofadr[0] for j in ARM_JOINTS]
    limits = torque_limits({"d1": "urdf"}.get(arm, arm))[12:]
    names = ("arm_joint2", "arm_joint3", "arm_joint5")
    ranges = [model.joint(n).range for n in names]
    mass = sum(model.body_mass[b] for b in bodies)
    rows = []
    grid = itertools.product(*(np.linspace(*r, n) for r, n in zip(ranges, (25, 25, 7))))
    for values in grid:
        worst, clear, height, ahead = 0.0, True, 0.0, 0.0
        for pitch in (0.0, np.radians(60)):
            data.qpos[:] = model.qpos0
            data.qpos[:3] = [0, 0, 2.0]
            data.qpos[3:7] = [np.cos(pitch / 2), 0, np.sin(-pitch / 2), 0]
            for name, value in zip(names, values):
                data.qpos[model.joint(name).qposadr[0]] = value
            mujoco.mj_forward(model, data)
            for k in range(data.ncon):
                c = data.contact[k]
                if (c.geom1 in geoms or c.geom2 in geoms) and c.dist < 0.005:
                    clear = False
            if not clear:
                break
            worst = max(worst, float(np.max(np.abs(data.qfrc_bias[dofs]) / limits)))
            if pitch == 0.0:
                centre = sum(model.body_mass[b] * data.xipos[b] for b in bodies) / mass
                height, ahead = centre[2] - 2.0, centre[0]
        if clear and worst <= SHARE:
            rows.append((height + 0.5 * abs(ahead), height, ahead, worst, values))
    rows.sort(key=lambda r: r[0])
    for score, height, ahead, worst, values in rows[:5]:
        print(
            f"score {score:.3f} height {height:.3f} ahead {ahead:.3f} ratio {worst:.3f}",
            dict(zip(names, np.round(values, 2).tolist())),
        )


if __name__ == "__main__":
    main()
