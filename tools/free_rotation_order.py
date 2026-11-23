"""Order of accuracy of the integrators on the mission robot tumbling freely.

The mission comparison (tools/integrator_study.py) cannot show the order of RK4 itself: the
controller runs outside the integrator and holds its input for each step. Here the same robot
(Go2 + D1, lever scene) spins with no gravity, no constraints (so no contacts or limits), and
no actuation, so the motion is a smooth ODE: free-joint rotation plus the coupled arm and leg
joints with their passive damping. MuJoCo 4.0 changelog: RK4 is now fourth-order accurate for
free and ball joint orientation (3.x was second order).

  python tools/free_rotation_order.py evidence/<run-id>/mujoco-4.0/free_rotation.json

Run it with the WSL training environment (MuJoCo 3.5) into another file for the 3.x numbers.
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2_wall import lever as lv  # noqa: E402

DURATION_S = 2.0
BASE_ANGVEL = (1.5, -2.0, 1.0)  # rad/s, free-joint frame
JOINT_VEL = 0.5  # rad/s, alternating sign over the hinge joints
TIMESTEPS = (0.004, 0.002, 0.001, 0.0005, 0.00025)
REFERENCE_DT = 0.00025 / 16


def tumble(integrator: str, dt: float) -> tuple[np.ndarray, mujoco.MjModel]:
    model, data = lv.build_scene()
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    model.opt.gravity[:] = 0.0
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    model.opt.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{integrator.upper()}")
    model.opt.timestep = dt
    mujoco.mj_resetData(model, data)
    data.qvel[3:6] = BASE_ANGVEL
    data.qvel[6:] = JOINT_VEL * (-1.0) ** np.arange(model.nv - 6)
    steps = round(DURATION_S / dt)
    assert np.isclose(steps * dt, DURATION_S), dt
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return data.qpos.copy(), model


def error(model: mujoco.MjModel, qpos: np.ndarray, ref: np.ndarray) -> dict:
    """Configuration error as a tangent vector: base position, base rotation, other joints."""
    v = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, v, 1.0, ref, qpos)
    return {
        "base_position_m": float(np.linalg.norm(v[:3])),
        "base_rotation_rad": float(np.linalg.norm(v[3:6])),
        "joints_max_rad": float(np.abs(v[6:]).max()),
    }


def order(errors: list[dict]) -> list[dict]:
    """log2 of the error ratio between neighbouring timesteps (each half of the previous).

    >>> order([{"a": 16.0}, {"a": 1.0}, {"a": 0.0}])
    [{'a': 4.0}, {'a': None}]
    """
    out = []
    for coarse, fine in zip(errors, errors[1:]):
        out.append(
            {
                k: round(float(np.log2(coarse[k] / fine[k])), 2)
                if coarse[k] > 0 and fine[k] > 0
                else None
                for k in coarse
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    ref, model = tumble("rk4", REFERENCE_DT)
    table = {
        "mujoco": mujoco.__version__,
        "duration_s": DURATION_S,
        "base_angvel_rad_s": list(BASE_ANGVEL),
        "joint_vel_rad_s": JOINT_VEL,
        "reference": f"rk4 at {REFERENCE_DT * 1e3:g} ms",
        "timesteps_ms": [dt * 1e3 for dt in TIMESTEPS],
        "integrators": {},
    }
    for integrator in ("euler", "rk4"):
        errors = [error(model, tumble(integrator, dt)[0], ref) for dt in TIMESTEPS]
        table["integrators"][integrator] = {"errors": errors, "observed_order": order(errors)}
        print(integrator, json.dumps(table["integrators"][integrator]["observed_order"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
