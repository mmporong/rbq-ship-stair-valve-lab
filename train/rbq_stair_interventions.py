"""Develop PPO-compatible position-target interventions for the RBQ stair task.

The frozen actor always supplies the base action.  A demonstration candidate may
blend saved leg-joint targets into that action after the robot reaches the stair
entrance.  A weight of one is therefore reported as teacher replacement, not as
PPO recovery; weights strictly between zero and one are actor residuals.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from eval_stairs_cpu import evaluation_substeps, supporting_contacts
from eval_target_stairs import FOOT_NAMES, START_X_M, route_command
from probe_stair_entry import EntryPolicy, load_policy
from replay_stair_sequence import segments_and_milestones

from go2_wall import stairs, transition
from go2_wall.statics import joint_names, torque_limits

TARGET_STANCE = {"FL": 5, "FR": 5, "RL": 3, "RR": 3}
HOLD_S = 0.5


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    speed_mps: float = 0.60
    phase_offset_steps: int = 0
    teacher_weight: float = 0.0
    teacher_time_scale: float = 0.18
    activation_x_m: float = -0.46
    teacher_command_speed_mps: float = 0.0
    lateral_gain: float = 1.0
    lateral_limit_mps: float = 0.15
    speed_after_tread3_mps: float | None = None

    @property
    def controller_class(self) -> str:
        if self.teacher_weight == 0:
            return "frozen_ppo"
        if self.teacher_weight < 1:
            return "frozen_ppo_actor_residual"
        return "teacher_leg_target_replacement"


CANDIDATES = {
    "ppo": Candidate("ppo"),
    "phase25": Candidate("phase25", phase_offset_steps=25),
    "phase25_feedback": Candidate(
        "phase25_feedback",
        phase_offset_steps=25,
        lateral_gain=3.0,
        lateral_limit_mps=0.35,
        speed_after_tread3_mps=0.35,
    ),
    "demo_r35": Candidate("demo_r35", phase_offset_steps=25, teacher_weight=0.35),
    "demo_r60": Candidate("demo_r60", phase_offset_steps=25, teacher_weight=0.60),
    "demo_teacher": Candidate("demo_teacher", phase_offset_steps=25, teacher_weight=1.0),
}


def interpolate_rows(times: np.ndarray, values: np.ndarray, query_s: float) -> np.ndarray:
    """Linearly interpolate a sampled joint trajectory, clamping both ends."""
    if times.ndim != 1 or values.ndim != 2 or len(times) != len(values) or len(times) < 1:
        raise ValueError("trajectory arrays must be non-empty and aligned")
    if not np.all(np.diff(times) > 0):
        raise ValueError("trajectory times must be strictly increasing")
    if query_s <= times[0]:
        return values[0].copy()
    if query_s >= times[-1]:
        return values[-1].copy()
    high = int(np.searchsorted(times, query_s, side="right"))
    low = high - 1
    weight = (query_s - times[low]) / (times[high] - times[low])
    return (1.0 - weight) * values[low] + weight * values[high]


def blend_position_targets(
    actor_targets: np.ndarray, teacher_targets: np.ndarray, weight: float
) -> np.ndarray:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("teacher weight must be in [0, 1]")
    if actor_targets.shape != teacher_targets.shape:
        raise ValueError("actor and teacher targets must have the same shape")
    return actor_targets + weight * (teacher_targets - actor_targets)


def build_teacher_track(geometry_path: Path, entry_path: Path, plan_path: Path, scale: float):
    """Build the saved F5/R3 demonstration and compress its timing for development."""
    if not 0 < scale <= 1:
        raise ValueError("teacher time scale must be in (0, 1]")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    sequence = json.loads(plan_path.read_text(encoding="utf-8"))
    # The stance after moves 0..15 is FL/FR tread 5 and RL/RR tread 3.
    plan, milestones = segments_and_milestones(geometry, entry, sequence["moves"][:16])
    scaled = copy.deepcopy(plan)
    for segment in scaled:
        segment["seconds"] = max(transition.SAMPLE_S, segment["seconds"] * scale)
    stairs.ARM = geometry["arm"]
    stairs.use_stow()
    model, data = stairs.build_stairs(geometry["slope_deg"], geometry["rise_m"])
    track = transition.sample_segments(
        model, data, scaled, torque_limits(geometry["arm"]), geometry["arm_stow_rad"]
    )
    leg_columns = [
        joint_names().index(f"{leg}_{part}_joint")
        for leg in FOOT_NAMES
        for part in ("hip", "thigh", "calf")
    ]
    return {
        "time": track["time"],
        "target": track["target"][:, leg_columns],
        "torque": track["torque"][:, leg_columns],
        "duration_s": float(track["time"][-1]),
        "unscaled_duration_s": float(milestones[-1][0]),
    }


class DemonstrationResidualPolicy(EntryPolicy):
    teacher_target: np.ndarray | None = None
    teacher_origin: np.ndarray | None = None
    teacher_weight: float = 0.0

    def applied_action(self, action: np.ndarray) -> np.ndarray:
        self.target_adjustment = None
        applied = super().applied_action(action)
        if self.teacher_target is None or self.teacher_weight == 0:
            return applied
        actor_targets = self.position_targets(applied)
        if self.teacher_weight < 1.0:
            if self.teacher_origin is None:
                raise ValueError("a residual teacher requires an origin target")
            targets = actor_targets + self.teacher_weight * (
                self.teacher_target - self.teacher_origin
            )
        else:
            targets = self.teacher_target
        lo, hi = self.joint_bounds.T
        normalized = (np.clip(targets, lo, hi) - self.offset) / self.meta["action_scale"]
        return super().applied_action(normalized)


def load_residual_policy(bundle: Path) -> tuple[DemonstrationResidualPolicy, float]:
    base, height = load_policy(bundle)
    base.__class__ = DemonstrationResidualPolicy
    return base, height


def run_ascent(cpu, height: float, case: dict, candidate: Candidate, teacher: dict | None) -> dict:
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
    cpu.teacher_weight = candidate.teacher_weight
    cpu.teacher_target = None
    cpu.teacher_origin = None if teacher is None else teacher["target"][0].copy()
    substeps = evaluation_substeps(cpu, cpu.meta["physics_timestep_s"], False)
    feet = {model.geom(f"robot/{leg}_foot_collision").id: leg for leg in FOOT_NAMES}
    treads = {model.geom(f"tread{k}").id: k for k in range(1, 11)}
    previous = np.zeros(len(cpu.actuator_ids), dtype=np.float32)
    support = {leg: set() for leg in FOOT_NAMES}
    contacted: set[int] = set()
    stance_s = max_stance_s = 0.0
    activation_step: int | None = None
    body_collision = False
    first_illegal_sensor: str | None = None
    max_x = START_X_M
    max_z = float(data.qpos[2])
    max_pitch = 0.0

    def measure(state: mujoco.MjData) -> None:
        nonlocal stance_s, max_stance_s, body_collision, max_pitch, first_illegal_sensor
        for values in support.values():
            values.clear()
        for foot, surface, _ in supporting_contacts(model, state, set(feet)):
            if surface in treads:
                tread = treads[surface]
                support[feet[foot]].add(tread)
                contacted.add(tread)
        matches = all(tread in support[leg] for leg, tread in TARGET_STANCE.items())
        stance_s = stance_s + model.opt.timestep if matches else 0.0
        max_stance_s = max(max_stance_s, stance_s)
        max_pitch = max(
            max_pitch, abs(math.degrees(math.asin(np.clip(state.xmat[cpu.base, 6], -1, 1))))
        )
        forces = state.sensordata[cpu.contact_force_ids]
        illegal = np.sum(forces * forces, axis=1) > cpu.contact_threshold_sq
        body_collision |= bool(np.any(illegal))
        if first_illegal_sensor is None and np.any(illegal):
            first_illegal_sensor = cpu.meta["illegal_contact_sensors"][
                int(np.flatnonzero(illegal)[0])
            ]

    termination = None
    for step in range(cpu.meta["steps"]):
        if teacher is not None and candidate.teacher_weight > 0:
            if activation_step is None and data.xpos[cpu.base, 0] >= candidate.activation_x_m:
                activation_step = step
            if activation_step is not None:
                elapsed = (step - activation_step) * cpu.control_dt
                cpu.teacher_target = interpolate_rows(teacher["time"], teacher["target"], elapsed)
        speed = (
            candidate.teacher_command_speed_mps
            if activation_step is not None and candidate.teacher_weight > 0
            else candidate.speed_mps
        )
        if candidate.speed_after_tread3_mps is not None and 3 in contacted:
            speed = candidate.speed_after_tread3_mps
        command, _ = route_command(cpu, data, "ascend", speed)
        if candidate.lateral_gain != 1.0 or candidate.lateral_limit_mps != 0.15:
            heading = math.atan2(data.xmat[cpu.base, 3], data.xmat[cpu.base, 0])
            world_y = float(
                np.clip(
                    -candidate.lateral_gain * data.xpos[cpu.base, 1],
                    -candidate.lateral_limit_mps,
                    candidate.lateral_limit_mps,
                )
            )
            c, s = math.cos(heading), math.sin(heading)
            command[0] += s * world_y
            command[1] += c * world_y
        previous, termination = cpu.control_step(
            data, command, previous, step, substeps, physics_observer=measure
        )
        max_x = max(max_x, float(data.qpos[0]))
        max_z = max(max_z, float(data.qpos[2]))
        if termination is not None:
            break
    return {
        "case_id": case["case_id"],
        "candidate_id": candidate.candidate_id,
        "controller_class": candidate.controller_class,
        "termination": termination or "phase_timeout",
        "work_stance_reached": max_stance_s >= HOLD_S - 1e-9,
        "maximum_work_stance_s": max_stance_s,
        "highest_supported_tread": max(contacted, default=0),
        "supported_treads": sorted(contacted),
        "body_collision": body_collision,
        "first_illegal_sensor": first_illegal_sensor,
        "maximum_forward_x_m": max_x,
        "maximum_body_z_m": max_z,
        "maximum_abs_pitch_deg": max_pitch,
        "teacher_activated": activation_step is not None,
        "teacher_activation_s": None
        if activation_step is None
        else activation_step * cpu.control_dt,
        "simulated_s": float(data.time),
    }


def summarize(trials: list[dict]) -> dict:
    return {
        "n": len(trials),
        "work_stance_rate": float(np.mean([row["work_stance_reached"] for row in trials])),
        "mean_highest_supported_tread": float(
            np.mean([row["highest_supported_tread"] for row in trials])
        ),
        "body_collision_rate": float(np.mean([row["body_collision"] for row in trials])),
        "mean_maximum_forward_x_m": float(np.mean([row["maximum_forward_x_m"] for row in trials])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--candidates", nargs="+", choices=sorted(CANDIDATES), required=True)
    parser.add_argument(
        "--geometry",
        type=Path,
        default=REPO / "evidence/t1-stair-ascent-20261120/target-round-trip-geometry.json",
    )
    parser.add_argument(
        "--entry", type=Path, default=REPO / ".omx/t1-statics-20261120/front-pair.json"
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=REPO / "evidence/t1-stair-ascent-20261120/ascent-and-turn-plan.json",
    )
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output path already exists")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    cpu, height = load_residual_policy(args.bundle_root.resolve() / "t1-v8-cpu-9100")
    groups = {}
    tracks: dict[float, dict] = {}
    for candidate_id in args.candidates:
        candidate = CANDIDATES[candidate_id]
        teacher = None
        if candidate.teacher_weight > 0:
            teacher = tracks.setdefault(
                candidate.teacher_time_scale,
                build_teacher_track(
                    args.geometry, args.entry, args.plan, candidate.teacher_time_scale
                ),
            )
            if "effective_target" not in teacher:
                kp = cpu.model.actuator_gainprm[cpu.actuator_ids, 0]
                if np.any(kp <= 0):
                    raise ValueError("position actuators require positive proportional gains")
                teacher["effective_target"] = teacher["target"] + teacher["torque"] / kp
            teacher = {**teacher, "target": teacher["effective_target"]}
        trials = [run_ascent(cpu, height, case, candidate, teacher) for case in cases]
        groups[candidate_id] = {
            "candidate": candidate.__dict__,
            "summary": summarize(trials),
            "trials": trials,
        }
    result = {
        "scope": "development_only_previous_cases",
        "cpu_only": True,
        "checkpoint": 9100,
        "checkpoint_sha256": cpu.meta["checkpoint_sha256"],
        "teacher_disclosure": (
            "weight 1 replaces PPO leg targets after activation; "
            "0<weight<1 is a PPO actor residual"
        ),
        "groups": groups,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
