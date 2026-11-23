"""Run B1 with an XY stance spring added only to the physical motor command."""

# ruff: noqa: E402, I001 -- fixed-snapshot paths must precede local imports.

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import textwrap
from pathlib import Path

import numpy as np


SPRING_START_S = 5.0
SPRING_END_S = 21.25


def sha256_lf(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode()).hexdigest()


def text_sha256_lf(value: str) -> str:
    return hashlib.sha256(value.replace("\r\n", "\n").encode()).hexdigest()


def canonical_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--spring-gain", type=float, default=2500.0)
    parser.add_argument("--timestep", type=float, default=0.001)
    parser.add_argument("--motion-seconds", type=float, default=9.0)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    if args.spring_gain <= 0 or not np.isfinite(args.spring_gain):
        parser.error("Spring gain must be finite and positive")

    sys.path.insert(0, str(source_root / "src"))
    sys.path.insert(0, str(source_root / "tools"))
    import mujoco
    import replay_contact_valve
    from go2_wall import grasp_motion, transition
    from go2_wall.mission_verdict import evaluate_valve

    grasp_motion.JAW_LOAD_BIAS_M = 0.0005
    anchors_xy: dict[int, np.ndarray] = {}
    maximum_error_m = 0.0
    maximum_force_n = 0.0
    maximum_joint_torque_nm = 0.0

    def stance_xy_motor_correction(model, data, geom, dofs, time_s):
        nonlocal maximum_error_m, maximum_force_n, maximum_joint_torque_nm
        if time_s < SPRING_START_S - 1e-9 or time_s > SPRING_END_S + 1e-9:
            return np.zeros(len(dofs))
        centre_xy = data.geom_xpos[geom][:2].copy()
        anchor_xy = anchors_xy.setdefault(int(geom), centre_xy)
        error_xy = centre_xy - anchor_xy
        force_xy = args.spring_gain * error_xy
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jacGeom(model, data, jacobian, None, geom)
        correction = -(jacobian[:2, dofs].T @ force_xy)
        maximum_error_m = max(maximum_error_m, float(np.linalg.norm(error_xy)))
        maximum_force_n = max(maximum_force_n, float(np.linalg.norm(force_xy)))
        maximum_joint_torque_nm = max(
            maximum_joint_torque_nm, float(np.max(np.abs(correction)))
        )
        return correction

    original_execute_source = textwrap.dedent(inspect.getsource(transition.execute))
    needle = (
        "                    command[cols] -= foot_jacobian[:, velocity[cols]].T @ "
        "(SLIDE_DAMPING * slide)\n"
    )
    replacement = needle + (
        "                    command[cols] += _stance_xy_motor_correction(\n"
        "                        model, data, feet_ids[k], velocity[cols], t\n"
        "                    )\n"
    )
    if original_execute_source.count(needle) != 1:
        raise RuntimeError("Fixed execute() motor insertion point changed")
    transformed_execute_source = original_execute_source.replace(needle, replacement, 1)
    transition._stance_xy_motor_correction = stance_xy_motor_correction
    exec(compile(transformed_execute_source, "<torque-stance-spring>", "exec"), transition.__dict__)

    result = replay_contact_valve.run(
        replay_contact_valve.read(
            source_root / "evidence/t3-valve-20261120/fixed-body-candidates.json"
        ),
        replay_contact_valve.read(
            source_root
            / "evidence/t1-stair-ascent-20261120/target-round-trip-geometry.json"
        ),
        candidate=4,
        target_deg=90.0,
        resistance_nm=10.0,
        motion_s=args.motion_seconds,
        gain=2.5,
        timestep=args.timestep,
        finger_force_n=100.0,
        pad_mu=0.8,
        slide_loss_n=30.0,
    )
    config = {
        "base_commit": "b704ccf5f19a2ab87efd102fbb7a263f34221f7d",
        "jaw_load_bias_m": 0.0005,
        "spring_gain_n_per_m": args.spring_gain,
        "spring_active_s": [SPRING_START_S, SPRING_END_S],
        "spring_axes": "world_xy",
        "spring_application": "physical stance motor command only: -J_xy.T K e_xy",
        "contact_velocity_modified": False,
        "stance_slip_measurement_modified": False,
        "motion_seconds": args.motion_seconds,
        "motion_contract": "probe duration; rated floor-start mission uses 8.75 s",
        "physics_timestep_s": args.timestep,
        "actuation_period_s": 0.005,
        "runner_sha256_lf": sha256_lf(Path(__file__)),
        "original_execute_sha256_lf": text_sha256_lf(original_execute_source),
        "transformed_execute_sha256_lf": text_sha256_lf(transformed_execute_source),
        "source_sha256_lf": result["source_sha256_lf"],
    }
    config["effective_variant_sha256"] = canonical_hash(config)
    result["torque_stance_spring_probe"] = config
    result["torque_stance_spring_metrics"] = {
        "anchor_count": len(anchors_xy),
        "maximum_xy_error_m": maximum_error_m,
        "maximum_spring_force_n": maximum_force_n,
        "maximum_joint_correction_nm": maximum_joint_torque_nm,
    }
    result["isolated_verdict"] = evaluate_valve(result["trace"], result["metadata"])
    replay_contact_valve.save(args.out, result)
    print(json.dumps(result["isolated_verdict"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
