"""Replay bounded entrance changes on the exact ship stair with frozen weights."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache, cached_property, partial
from itertools import product
from pathlib import Path

import mujoco
import numpy as np
from eval_stairs_cpu import CpuStairs
from eval_target_stairs import FOOT_NAMES, START_X_M, mission_model, route_command, run_route

SLOPE_DEG = 50
RISE_M = 0.19
TREADS = 10
SPEED_BLEND_X_M = (-0.65, -0.35)
EXTENSION_BLEND_X_M = (-0.72, -0.37, 0.17, 0.48)


@dataclass(frozen=True)
class Probe:
    checkpoint: int
    speed_mps: float
    phase_offset_steps: int = 0
    edge_speed_mps: float | None = None
    extension_legs: str | None = None
    nominal_foot_extension_m: float = 0.0


class EntryPolicy(CpuStairs):
    phase_offset_steps = 0
    target_adjustment: np.ndarray | None = None

    def observe(self, data, command, previous, step):
        return super().observe(data, command, previous, step + self.phase_offset_steps)

    @cached_property
    def extension_vectors(self) -> dict[str, np.ndarray]:
        """Position target changes per metre of nominal downward foot offset."""
        pose = mujoco.MjData(self.model)
        pose.qpos[:7] = [START_X_M, 0.0, 0.56, 1.0, 0.0, 0.0, 0.0]
        pose.qpos[self.pos_ids] = self.defaults
        mujoco.mj_forward(self.model, pose)
        names = self.meta["action_joints"]
        extensions = {mode: np.zeros(len(names), dtype=np.float32) for mode in ("all", "front")}
        jacobian = np.zeros((3, self.model.nv))
        for leg in FOOT_NAMES:
            leg_names = [f"robot/{leg}_{joint}_joint" for joint in ("thigh", "calf")]
            action_ids = [names.index(name) for name in leg_names]
            velocity_ids = [int(self.model.joint(name).dofadr[0]) for name in leg_names]
            mujoco.mj_jacSite(self.model, pose, jacobian, None, self.model.site(f"robot/{leg}").id)
            extension = np.linalg.solve(jacobian[np.ix_([0, 2], velocity_ids)], [0.0, -1.0])
            extensions["all"][action_ids] = extension
            if leg.startswith("F"):
                extensions["front"][action_ids] = extension
        return extensions

    @cached_property
    def joint_bounds(self) -> np.ndarray:
        return np.array([self.model.joint(name).range for name in self.meta["action_joints"]])

    def applied_action(self, action: np.ndarray) -> np.ndarray:
        applied = super().applied_action(action)
        if self.target_adjustment is None:
            return applied
        changed = self.target_adjustment != 0
        targets = self.position_targets(applied)[changed] + self.target_adjustment[changed]
        lo, hi = self.joint_bounds[changed].T
        applied[changed] = (np.clip(targets, lo, hi) - self.offset[changed]) / self.meta[
            "action_scale"
        ]
        return super().applied_action(applied)


@cache
def load_policy(bundle: Path) -> tuple[EntryPolicy, float]:
    policy = EntryPolicy(bundle, model=mission_model(bundle, SLOPE_DEG, RISE_M, TREADS))
    with np.load(bundle / "starts.npz", allow_pickle=False) as starts:
        height = float(starts["qpos"][0, 0, 2] - starts["origin"][0, 0, 2])
    return policy, height


def replay(case: Probe, bundle_root: Path) -> dict:
    cpu, height = load_policy(bundle_root / f"t1-v8-cpu-{case.checkpoint}")
    cpu.phase_offset_steps = case.phase_offset_steps
    extension = (
        cpu.extension_vectors[case.extension_legs] if case.extension_legs is not None else None
    )

    def entry_command(cpu, data, phase, speed):
        cpu.target_adjustment = None
        if phase == "ascend":
            x = data.xpos[cpu.base, 0]
            if case.edge_speed_mps is not None:
                speed = float(np.interp(x, SPEED_BLEND_X_M, (speed, case.edge_speed_mps)))
            if extension is not None:
                weight = float(np.interp(x, EXTENSION_BLEND_X_M, (0, 1, 1, 0)))
                if weight > 0:
                    cpu.target_adjustment = weight * case.nominal_foot_extension_m * extension
        return route_command(cpu, data, phase, speed)

    trial = run_route(cpu, case.speed_mps, TREADS, height, command_for=entry_command)
    return {
        "checkpoint": case.checkpoint,
        "checkpoint_sha256": cpu.meta["checkpoint_sha256"],
        "phase_offset_steps": case.phase_offset_steps,
        "edge_speed_mps": case.edge_speed_mps,
        "extension_legs": case.extension_legs,
        "nominal_foot_extension_m": case.nominal_foot_extension_m,
        **trial,
    }


def cases_for(mode: str) -> list[Probe]:
    speeds = (0.35, 0.50, 0.60)
    if mode == "extension":
        return [
            Probe(checkpoint, speed, extension_legs=legs, nominal_foot_extension_m=amount)
            for checkpoint, speed, legs, amount in product(
                (9100, 9200, 9496), speeds, ("all", "front"), (0.035, 0.07)
            )
        ]
    checkpoints = (9000, 9100, 9200, 9496)
    cases = [Probe(*args) for args in product(checkpoints, speeds, range(0, 30, 5))]
    cases.extend(
        Probe(*args) for args in product(checkpoints, (0.50,), (0, 7, 15, 22), (0.22, 0.75))
    )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundle_root", type=Path, help="Directory containing the t1-v8-cpu-* bundles"
    )
    parser.add_argument("out", type=Path)
    parser.add_argument("--mode", choices=("timing", "extension"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists() or args.workers < 1:
        parser.error("Use a new output path and at least one worker")
    cases = cases_for(args.mode)
    function = partial(replay, bundle_root=args.bundle_root.resolve())
    trials = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, trial in enumerate(pool.map(function, cases), start=1):
            trials.append(trial)
            if index % 16 == 0 or index == len(cases):
                print(f"Replayed {index}/{len(cases)} cases", flush=True)
    result = {
        "mode": args.mode,
        "mujoco": mujoco.__version__,
        "slope_deg": SLOPE_DEG,
        "rise_m": RISE_M,
        "treads": TREADS,
        "approach_speed_blend_x_m": SPEED_BLEND_X_M,
        "extension_blend_x_m": EXTENSION_BLEND_X_M,
        "trials": trials,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
