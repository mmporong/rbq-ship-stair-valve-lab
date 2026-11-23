"""Run a frozen-policy RBQ stair intervention experiment on CPU.

This is an independent RBQ host experiment.  It is not a registered NVIDIA
``rl-triage-agent`` task, check, adaptation, or policy update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))

from eval_stairs_cpu import evaluation_substeps, supporting_contacts  # noqa: E402
from eval_target_stairs import FOOT_NAMES, START_X_M, route_command  # noqa: E402
from probe_stair_entry import EntryPolicy, load_policy  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    speed_mps: float
    phase_offset_steps: int = 0
    edge_speed_mps: float | None = None


CANDIDATES = {
    "no_intervention": Candidate("no_intervention", 0.60),
    "phase_15": Candidate("phase_15", 0.60, phase_offset_steps=15),
    "phase_25": Candidate("phase_25", 0.60, phase_offset_steps=25),
    "edge_slow_rule": Candidate("edge_slow_rule", 0.60, edge_speed_mps=0.22),
}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_cases(cases: list[dict]) -> list[dict]:
    clean = []
    for case in cases:
        if set(case) != {"case_id", "lateral_m", "yaw_deg", "joint_seed", "joint_noise_rad"}:
            raise ValueError("case fields do not match the frozen host contract")
        if not isinstance(case["case_id"], str) or not case["case_id"]:
            raise ValueError("case_id must be non-empty")
        if not isinstance(case["joint_seed"], int):
            raise ValueError("joint_seed must be an integer")
        for field in ("lateral_m", "yaw_deg", "joint_noise_rad"):
            if isinstance(case[field], bool) or not isinstance(case[field], (int, float)):
                raise ValueError(f"{field} must be numeric")
        clean.append(dict(case))
    if len({case["case_id"] for case in clean}) != len(clean):
        raise ValueError("duplicate case_id")
    return clean


def run_ascent(cpu: EntryPolicy, height: float, case: dict, candidate: Candidate) -> dict:
    model = cpu.model
    data = mujoco.MjData(model)
    yaw = math.radians(case["yaw_deg"])
    data.qpos[:7] = [
        START_X_M,
        case["lateral_m"],
        height,
        math.cos(yaw / 2),
        0.0,
        0.0,
        math.sin(yaw / 2),
    ]
    rng = np.random.default_rng(case["joint_seed"])
    data.qpos[cpu.pos_ids] = cpu.defaults + rng.uniform(
        -case["joint_noise_rad"], case["joint_noise_rad"], len(cpu.pos_ids)
    )
    mujoco.mj_forward(model, data)

    cpu.phase_offset_steps = candidate.phase_offset_steps
    cpu.target_adjustment = None
    substeps = evaluation_substeps(cpu, cpu.meta["physics_timestep_s"], False)
    feet = {model.geom(f"robot/{leg}_foot_collision").id: leg for leg in FOOT_NAMES}
    treads = {model.geom(f"tread{k}").id: k for k in range(1, 11)}
    previous = np.zeros(len(cpu.actuator_ids), dtype=np.float32)
    support: dict[str, set[int]] = {leg: set() for leg in FOOT_NAMES}
    contacted_treads: set[int] = set()
    max_stance_s = 0.0
    stance_s = 0.0
    max_forward = START_X_M
    max_z = float(data.qpos[2])
    max_abs_pitch = 0.0
    body_collision = False
    target = {"FL": 5, "FR": 5, "RL": 3, "RR": 3}

    def measure(state: mujoco.MjData) -> None:
        nonlocal stance_s, max_stance_s, body_collision, max_abs_pitch
        for values in support.values():
            values.clear()
        for foot, surface, _ in supporting_contacts(model, state, set(feet)):
            leg = feet[foot]
            if surface in treads:
                tread = treads[surface]
                support[leg].add(tread)
                contacted_treads.add(tread)
        matches = all(target[leg] in support[leg] for leg in FOOT_NAMES)
        stance_s = stance_s + model.opt.timestep if matches else 0.0
        max_stance_s = max(max_stance_s, stance_s)
        max_abs_pitch = max(
            max_abs_pitch,
            abs(math.degrees(math.asin(np.clip(state.xmat[cpu.base, 6], -1.0, 1.0)))),
        )
        forces = state.sensordata[cpu.contact_force_ids]
        body_collision |= bool(np.any(np.sum(forces * forces, axis=1) > cpu.contact_threshold_sq))

    termination = None
    max_steps = cpu.meta["steps"]
    for step in range(max_steps):
        speed = candidate.speed_mps
        if candidate.edge_speed_mps is not None:
            speed = float(
                np.interp(data.xpos[cpu.base, 0], (-0.65, -0.35), (speed, candidate.edge_speed_mps))
            )
        command, _ = route_command(cpu, data, "ascend", speed)
        previous, termination = cpu.control_step(
            data, command, previous, step, substeps, physics_observer=measure
        )
        max_forward = max(max_forward, float(data.qpos[0]))
        max_z = max(max_z, float(data.qpos[2]))
        if termination is not None:
            break

    return {
        "case_id": case["case_id"],
        "candidate_id": candidate.candidate_id,
        "termination": termination or "phase_timeout",
        "work_stance_reached": max_stance_s >= 0.5 - 1e-9,
        "maximum_work_stance_s": max_stance_s,
        "highest_supported_tread": max(contacted_treads, default=0),
        "supported_treads": sorted(contacted_treads),
        "body_collision": body_collision,
        "maximum_forward_x_m": max_forward,
        "maximum_body_z_m": max_z,
        "maximum_abs_pitch_deg": max_abs_pitch,
        "final_body_xyz_m": data.qpos[:3].tolist(),
        "simulated_s": float(data.time),
    }


def summarize(trials: list[dict]) -> dict:
    n = len(trials)
    if not n:
        raise ValueError("cannot summarize an empty trial list")
    return {
        "n": n,
        "work_stance_rate": sum(t["work_stance_reached"] for t in trials) / n,
        "mean_maximum_work_stance_s": float(np.mean([t["maximum_work_stance_s"] for t in trials])),
        "mean_highest_supported_tread": float(
            np.mean([t["highest_supported_tread"] for t in trials])
        ),
        "body_collision_rate": sum(t["body_collision"] for t in trials) / n,
        "mean_maximum_forward_x_m": float(np.mean([t["maximum_forward_x_m"] for t in trials])),
        "mean_maximum_abs_pitch_deg": float(np.mean([t["maximum_abs_pitch_deg"] for t in trials])),
    }


def evaluate(bundle_root: Path, cases: list[dict], candidate_ids: list[str]) -> dict:
    cpu, height = load_policy(bundle_root / "t1-v8-cpu-9100")
    groups = {}
    for candidate_id in candidate_ids:
        candidate = CANDIDATES[candidate_id]
        trials = [run_ascent(cpu, height, case, candidate) for case in cases]
        groups[candidate_id] = {"summary": summarize(trials), "trials": trials}
    return {
        "host_contract": "rbq_stair_intervention_v1",
        "registered_nvidia_task": False,
        "checkpoint": 9100,
        "checkpoint_sha256": cpu.meta["checkpoint_sha256"],
        "cpu_only": True,
        "geometry": {"slope_deg": 50, "rise_m": 0.19, "treads": 10},
        "work_stance": {"FL": 5, "FR": 5, "RL": 3, "RR": 3, "hold_s": 0.5},
        "case_set_sha256": canonical_sha256(cases),
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--candidates", nargs="+", choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--expected-cases-sha256")
    args = parser.parse_args()
    cases = validate_cases(json.loads(args.cases.read_text(encoding="utf-8")))
    digest = canonical_sha256(cases)
    if args.expected_cases_sha256 is not None and digest != args.expected_cases_sha256:
        parser.error("case-set commitment mismatch")
    if args.out.exists():
        parser.error("output path already exists")
    result = evaluate(args.bundle_root.resolve(), cases, args.candidates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
