"""Measure the start of a descent with the frozen staircase CPU evaluator."""

import argparse
import json
import math
from pathlib import Path
from statistics import median

import mujoco
import numpy as np
from eval_stairs_cpu import CpuStairs, evaluation_substeps, supporting_contacts

FEET = ("FL", "FR", "RL", "RR")
DESCENTS = ("stairs_down", "open_stairs_down")
TRACE_INTERVAL = 10  # Control steps between retained samples for the example traces.
LATE_WINDOW_S = 2.0


def platform_edge(cpu: CpuStairs, origin: np.ndarray) -> float:
    """Find the +x edge of the top board containing the spawn origin."""
    covers_spawn = np.all(
        (cpu.box_min[:, :2] <= origin[:2]) & (cpu.box_max[:, :2] >= origin[:2]), axis=1
    )
    aligned_z = np.isclose(cpu.box_max[:, 2], origin[2], atol=2e-4, rtol=0)
    if np.count_nonzero(covers_spawn & aligned_z) != 1:
        raise ValueError(f"Expected one spawn platform at {origin.tolist()}")
    return float(cpu.box_max[covers_spawn & aligned_z, 0].item())


def trace_descent(
    cpu: CpuStairs, start: dict, steps: int, substeps: int
) -> tuple[dict, list[dict]]:
    model = cpu.model
    origin = start["origin"]
    edge = platform_edge(cpu, origin)
    foot_sites = [model.site(f"robot/{leg}").id for leg in FEET]
    foot_geoms = {model.geom(f"robot/{leg}_foot_collision").id: leg for leg in FEET}
    radius = min(float(model.geom_size[geom, 0]) for geom in foot_geoms)
    terrain_id = model.body("terrain").id
    centers, sizes = model.geom_pos, model.geom_size
    # Each forward face of this pyramid has a rectangular tread centered on the spawn y.
    tread_geoms = set(
        np.flatnonzero(
            (model.geom_bodyid == terrain_id)
            & np.isclose(centers[:, 1], origin[1], atol=2e-4, rtol=0)
            & np.isclose(centers[:, 0] - sizes[:, 0], origin[0] + sizes[:, 1], atol=2e-4, rtol=0)
            & (centers[:, 0] - sizes[:, 0] >= edge - 2e-4)
            & (centers[:, 2] + sizes[:, 2] < origin[2] - radius)
        )
    )
    if not tread_geoms:
        raise ValueError(f"Expected forward descent treads below {origin.tolist()}")
    frames = []
    first_lower_step = None
    contacted_treads = set()
    final_x = float(start["qpos"][0])
    forward_velocities = []
    pitches = []
    maximum_front_x = -math.inf
    maximum_base_x = -math.inf
    minimum_base_z = math.inf

    def measure(data: mujoco.MjData) -> None:
        nonlocal first_lower_step, final_x
        final_x = float(data.qpos[0])
        for foot, surface, point in supporting_contacts(model, data, foot_geoms):
            if surface not in tread_geoms or point[2] > origin[2] - radius:
                continue
            contacted_treads.add(surface)
            if first_lower_step is None:
                pitch = math.degrees(math.asin(np.clip(data.xmat[cpu.base, 6], -1.0, 1.0)))
                first_lower_step = {
                    "time_s": data.time,
                    "foot": foot_geoms[foot],
                    "contact_xyz_m": (point - origin).tolist(),
                    "body_pitch_up_deg": pitch,
                }

    def observe(step: int, data: mujoco.MjData, command: np.ndarray) -> None:
        nonlocal maximum_front_x, maximum_base_x, minimum_base_z
        rotation = data.xmat[cpu.base].reshape(3, 3)
        pitch = math.degrees(math.asin(np.clip(rotation[2, 0], -1.0, 1.0)))
        pose = data.xpos[cpu.base] - origin
        feet = data.site_xpos[foot_sites] - origin
        forward_velocities.append(float(data.qvel[0]))
        pitches.append(pitch)
        maximum_front_x = max(maximum_front_x, float(feet[:2, 0].max()))
        maximum_base_x = max(maximum_base_x, float(pose[0]))
        minimum_base_z = min(minimum_base_z, float(pose[2]))
        time_s = step * cpu.control_dt
        if step % TRACE_INTERVAL == 0:
            frames.append(
                {
                    "time_s": time_s,
                    "command": command.tolist(),
                    "world_forward_velocity_mps": float(data.qvel[0]),
                    "body_pitch_up_deg": pitch,
                    "body_xyz_m": pose.tolist(),
                    "feet_xyz_m": dict(zip(FEET, feet.tolist())),
                }
            )

    result = cpu.rollout(start, steps, substeps, observer=observe, physics_observer=measure)
    late_samples = round(LATE_WINDOW_S / cpu.control_dt)
    result.update(
        {
            "platform_edge_x_m": edge - float(origin[0]),
            "initial_body_x_m": float(start["qpos"][0] - origin[0]),
            "maximum_forward_body_x_m": maximum_base_x,
            "maximum_forward_forefoot_x_m": maximum_front_x,
            "minimum_body_z_above_start_terrain_m": minimum_base_z,
            "forward_displacement_m": final_x - float(start["qpos"][0]),
            "body_z_change_m": result["dz_m"] + float(origin[2] - start["qpos"][2]),
            "loaded_forward_surfaces": len(contacted_treads),
            "late_forward_velocity_mps": median(forward_velocities[-late_samples:]),
            "late_body_pitch_up_deg": median(pitches[-late_samples:]),
            "first_lower_foot_contact": first_lower_step,
        }
    )
    return result, frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--levels", type=int, nargs="+", default=(5, 6))
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    cpu = CpuStairs(args.bundle)
    if not set(args.levels) <= set(cpu.meta["levels"]):
        parser.error("Requested levels must be present in the bundle")
    substeps = evaluation_substeps(cpu, cpu.meta["physics_timestep_s"], False)
    result = {
        "mujoco": mujoco.__version__,
        "checkpoint": cpu.meta["checkpoint"],
        "checkpoint_sha256": cpu.meta["checkpoint_sha256"],
        "seed": cpu.meta["seed"],
        "physics_timestep_s": cpu.model.opt.timestep,
        "control_timestep_s": cpu.control_dt,
        "late_window_s": LATE_WINDOW_S,
        "trials": [],
        "example_traces": {},
    }
    with np.load(args.bundle / "starts.npz", allow_pickle=False) as starts:
        arrays = {key: starts[key] for key in starts.files}
    for level in args.levels:
        level_index = cpu.meta["levels"].index(level)
        for world, column in enumerate(cpu.meta["world_columns"]):
            terrain = cpu.meta["terrain_columns"][column]
            if terrain not in DESCENTS:
                continue
            start = {key: value[level_index, world] for key, value in arrays.items()}
            metrics, frames = trace_descent(cpu, start, cpu.meta["steps"], substeps)
            result["trials"].append({"level": level, "world": world, "terrain": terrain, **metrics})
            trace_name = f"{level}-{terrain}"
            if trace_name not in result["example_traces"]:
                result["example_traces"][trace_name] = {"world": world, "frames": frames}
        print(f"Measured level {level}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
