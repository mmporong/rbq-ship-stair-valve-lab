"""Learn a three-parameter residual gate from articulated MuJoCo rollout return.

The policy blends a recorded expert torque, measured-state contact feedback and
the frozen checkpoint-9100 leg command.  It never changes the expert trajectory,
the physical scene, the mission endpoint, or the actor weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import secrets
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from rbq_articulated_control import ArticulatedActor, ReferenceControl, digest  # noqa: E402
from rbq_articulated_study import Contacts, canonical_sha256, prepared, validate_cases  # noqa: E402

INITIAL_PARAMETERS = np.array([-2.0, 0.0, -3.0])
PARAMETER_LOWER = np.array([-3.0, -4.0, -6.0])
PARAMETER_UPPER = np.array([6.0, 4.0, 1.0])
TRAIN_HORIZON_S = 20.0
FULL_HORIZON_S = 127.75
CEM_SEED = 20261122
CEM_GENERATIONS = 3
CEM_POPULATION = 8
CEM_ELITE = 3
TARGET_TREADS = np.array([5, 5, 3, 3])
VARIANTS = (
    "learned",
    "initial",
    "fixed_contact",
    "teacher_open_loop",
    "learned_no_ppo",
    "learned_contact_off",
)


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0))))


def gates(parameters: np.ndarray, base_error_m: float) -> tuple[float, float]:
    """Return bounded contact and PPO authorities from one measured observation."""
    values = np.asarray(parameters, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Residual policy needs three finite parameters")
    scaled_error = float(np.clip(base_error_m / 0.03, 0.0, 3.0))
    error_gain = np.logaddexp(0.0, values[1])
    contact = sigmoid(values[0] + error_gain * scaled_error)
    ppo = 0.25 * sigmoid(values[2])
    return contact, ppo


def variant_gates(variant: str, parameters: np.ndarray, base_error_m: float) -> tuple[float, float]:
    if variant == "fixed_contact":
        return 1.0, 0.0
    if variant == "teacher_open_loop":
        return 0.0, 0.0
    if variant == "initial":
        return gates(INITIAL_PARAMETERS, base_error_m)
    contact, ppo = gates(parameters, base_error_m)
    if variant == "learned_no_ppo":
        ppo = 0.0
    elif variant == "learned_contact_off":
        contact = 0.0
    elif variant != "learned":
        raise ValueError(f"unknown residual variant: {variant}")
    return contact, ppo


def initialize(model, reference, case):
    data = mujoco.MjData(model)
    data.qpos[:] = reference["qpos"][0]
    yaw = math.radians(case["yaw_deg"])
    data.qpos[1] += case["lateral_m"]
    turn = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    mujoco.mju_mulQuat(data.qpos[3:7], turn, reference["qpos"][0, 3:7])
    leg_positions = model.jnt_qposadr[model.actuator_trnid[:12, 0]]
    data.qpos[leg_positions] += np.random.default_rng(case["joint_seed"]).uniform(
        -case["joint_noise_rad"], case["joint_noise_rad"], 12
    )
    mujoco.mj_forward(model, data)
    return data


def rollout(
    scene: Path,
    reference_path: Path,
    bundle: Path,
    case: dict,
    parameters: np.ndarray,
    variant: str,
    horizon_s: float,
    state_path: Path | None = None,
) -> dict:
    """Run one physical episode and return its reward and endpoint measurements."""
    model, reference = prepared(scene, reference_path)
    dt, stride = float(reference["dt"]), int(reference["stride"])
    data = initialize(model, reference, case)
    feedback = ReferenceControl(
        model, reference, "contact_gated_reference", case["lateral_m"], case["yaw_deg"]
    )
    actor = ArticulatedActor(model, bundle, "no_intervention")
    contacts = Contacts(model)
    loads = contacts.measure(data)
    limits = model.actuator_ctrlrange
    total = min(len(reference["command"]) * stride, round(horizon_s / dt))
    base = model.body("base").id
    hinge = model.joint("valve_hinge").qposadr[0]
    active = longest = highest = 0
    collision = fell = False
    maximum_angle = 0.0
    contact_sum = ppo_sum = actor_delta_sq = 0.0
    commands = 0
    records: dict[str, np.ndarray] = {}
    if state_path is not None:
        records = {
            name: np.empty((total, width))
            for name, width in (
                ("qpos", model.nq),
                ("qvel", model.nv),
                ("ctrl", model.nu),
                ("requested", model.nu),
            )
        }
    stop = None
    for step in range(total):
        if step % stride == 0:
            k = step // stride
            teacher = reference["command"][k].copy()
            contact_command = feedback.torque(k, data, loads.max(axis=1) >= 1.0)
            actor_command = actor.apply(k, data, teacher.copy())
            position_error = float(np.linalg.norm(data.qpos[:3] - reference["qpos"][k, :3]))
            contact_gate, ppo_gate = variant_gates(variant, parameters, position_error)
            requested = teacher + contact_gate * (contact_command - teacher)
            requested[:12] += ppo_gate * (actor_command[:12] - teacher[:12])
            data.ctrl[:] = np.clip(requested, limits[:, 0], limits[:, 1])
            contact_sum += contact_gate
            ppo_sum += ppo_gate
            actor_delta_sq += float(np.mean((ppo_gate * (actor_command[:12] - teacher[:12])) ** 2))
            commands += 1
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        loads = contacts.measure(data)
        active = active + 1 if (loads[np.arange(4), TARGET_TREADS] >= 1.0).all() else 0
        longest = max(longest, active)
        highest = max(
            highest,
            int(np.max(np.where(loads[:, 1:11] >= 1.0, np.arange(1, 11), 0))),
        )
        collision |= contacts.collision_n > 10.0
        fell |= data.xmat[base, 8] < math.cos(math.radians(70)) or data.qpos[2] < 0.12
        maximum_angle = max(maximum_angle, math.degrees(float(data.qpos[hinge])))
        if records:
            records["qpos"][step] = data.qpos
            records["qvel"][step] = data.qvel
            records["ctrl"][step] = data.ctrl
            records["requested"][step] = requested
        if collision or fell:
            stop = "terrain_body_contact" if collision else "fall"
            break
        if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
            raise ValueError(f"dynamics warning in {case['case_id']} at step {step + 1}")
    steps = step + 1
    work_stance = longest >= 500
    valve_open = maximum_angle >= 89.75
    complete = steps == total and horizon_s >= FULL_HORIZON_S
    mean_contact = contact_sum / commands
    mean_ppo = ppo_sum / commands
    reward = (
        steps * dt
        + 100.0 * highest
        + 5000.0 * work_stance
        + 10.0 * min(maximum_angle, 90.0)
        + 10000.0 * (work_stance and valve_open and complete)
        - 1000.0 * collision
        - 1000.0 * fell
        - 100.0 * mean_contact
    )
    state_sha256 = None
    if state_path is not None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with state_path.open("xb") as stream:
            np.savez(
                stream,
                qpos=records["qpos"][:steps],
                qvel=records["qvel"][:steps],
                ctrl=records["ctrl"][:steps],
                requested_robot_nm=records["requested"][:steps, :18],
                requested_finger_force_n=records["requested"][:steps, 18],
                time_s=(np.arange(steps) + 1) * dt,
                scene_sha256=reference["scene_sha256"],
                case_id=case["case_id"],
                variant=variant,
            )
        state_sha256 = digest(state_path)
    return {
        "case_id": case["case_id"],
        "variant": variant,
        "reward": reward,
        "termination": stop or "horizon",
        "simulated_s": steps * dt,
        "highest_supported_tread": highest,
        "maximum_work_stance_s": longest * dt,
        "work_stance_reached": bool(work_stance),
        "maximum_valve_angle_deg": maximum_angle,
        "full_mission_indicator": bool(work_stance and valve_open and complete and not collision),
        "body_collision": bool(collision),
        "fell": bool(fell),
        "mean_contact_authority": mean_contact,
        "mean_ppo_authority": mean_ppo,
        "ppo_torque_rms_nm": math.sqrt(actor_delta_sq / commands),
        "state_sha256": state_sha256,
    }


def evaluate_batch(scene, reference, bundle, cases, jobs, horizon, workers, state_dir=None):
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = []
        for candidate_id, parameters, variant in jobs:
            for case in cases:
                state = None
                if state_dir is not None:
                    state = state_dir / f"{case['case_id']}--{candidate_id}.npz"
                future = pool.submit(
                    rollout,
                    scene,
                    reference,
                    bundle,
                    case,
                    parameters,
                    variant,
                    horizon,
                    state,
                )
                futures.append((candidate_id, future))
        for candidate_id, future in futures:
            row = future.result()
            row["candidate_id"] = candidate_id
            rows.append(row)
    return rows


def candidate_score(rows: list[dict]) -> float:
    return float(np.mean([row["reward"] for row in rows]))


def train(scene, reference, bundle, cases, workers):
    rng = np.random.default_rng(CEM_SEED)
    mean = np.array([0.5, 0.0, -2.5])
    spread = np.array([2.0, 1.5, 1.5])
    ledger = []
    for generation in range(CEM_GENERATIONS):
        population = np.clip(
            rng.normal(mean, spread, (CEM_POPULATION, 3)), PARAMETER_LOWER, PARAMETER_UPPER
        )
        jobs = [(f"g{generation:02d}-p{i:02d}", row, "learned") for i, row in enumerate(population)]
        trials = evaluate_batch(scene, reference, bundle, cases, jobs, TRAIN_HORIZON_S, workers)
        grouped = []
        for candidate_id, parameters, _ in jobs:
            rows = [row for row in trials if row["candidate_id"] == candidate_id]
            grouped.append((candidate_score(rows), candidate_id, parameters, rows))
        grouped.sort(key=lambda item: (-item[0], item[1]))
        elites = grouped[:CEM_ELITE]
        elite_parameters = np.array([item[2] for item in elites])
        mean = elite_parameters.mean(axis=0)
        spread = np.maximum(elite_parameters.std(axis=0), [0.15, 0.15, 0.15])
        ledger.append(
            {
                "generation": generation,
                "population": [
                    {
                        "candidate_id": candidate_id,
                        "parameters": parameters.tolist(),
                        "mean_return": score,
                        "trials": rows,
                    }
                    for score, candidate_id, parameters, rows in grouped
                ],
                "elite_ids": [item[1] for item in elites],
                "next_mean": mean.tolist(),
                "next_std": spread.tolist(),
            }
        )
    return mean, ledger


def write_json_new(path: Path, value: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def seal_cases(secret_path: Path, commitment_path: Path):
    if secret_path.exists() or commitment_path.exists():
        raise FileExistsError("sealed case output already exists")
    cases = []
    for index in range(12):
        source = secrets.SystemRandom()
        cases.append(
            {
                "case_id": f"reward-holdout-{index + 1:02d}",
                "lateral_m": source.uniform(-0.025, 0.025),
                "yaw_deg": source.uniform(-2.5, 2.5),
                "joint_seed": secrets.randbits(63),
                "joint_noise_rad": source.uniform(0.003, 0.005),
            }
        )
    validate_cases(cases)
    write_json_new(secret_path, cases)
    write_json_new(
        commitment_path,
        {
            "schema": "rbq_reward_policy_holdout_commitment_v1",
            "count": len(cases),
            "case_set_sha256": canonical_sha256(cases),
            "distribution": {
                "lateral_m": [-0.025, 0.025],
                "yaw_deg": [-2.5, 2.5],
                "joint_noise_rad": [0.003, 0.005],
            },
            "comparison_variants": list(VARIANTS),
            "primary_endpoint": (
                "FL/FR tread 5 and RL/RR tread 3 for 0.5 s, then physical 90 degree "
                "passive valve mission"
            ),
            "training": {
                "algorithm": "CEM on physical rollout return",
                "seed": CEM_SEED,
                "generations": CEM_GENERATIONS,
                "population": CEM_POPULATION,
                "elite": CEM_ELITE,
                "horizon_s": TRAIN_HORIZON_S,
            },
            "heldout_horizon_s": FULL_HORIZON_S,
            "heldout_metrics": [
                "work_stance_success_rate",
                "strict_full_mission_success_rate",
                "body_collision_rate",
                "maximum_supported_tread",
                "mean_contact_authority",
                "mean_ppo_authority",
                "ppo_torque_rms_nm",
            ],
            "comparison_contract": (
                "Every variant uses the same scene, start, horizon, motor limits and endpoint. "
                "Learned parameters are frozen after development."
            ),
            "modification_budgets": {
                "development": "code, reward and parameters may change before the freeze commit",
                "heldout": "zero code, reward, parameter, seed or threshold changes after reveal",
            },
            "rule": (
                "Commit code, learned parameters and this commitment before opening the case set."
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("secret", type=Path)
    seal.add_argument("commitment", type=Path)
    for operation in ("train", "evaluate"):
        command = sub.add_parser(operation)
        for name in ("scene", "reference", "bundle", "cases", "output"):
            command.add_argument(name, type=Path)
        command.add_argument("--workers", type=int, default=6)
        if operation == "evaluate":
            command.add_argument("--policy", type=Path, required=True)
            command.add_argument("--state-dir", type=Path, required=True)
            command.add_argument("--expected-cases-sha256", required=True)
    args = parser.parse_args()
    if args.operation == "seal":
        seal_cases(args.secret, args.commitment)
        return
    cases = validate_cases(json.loads(args.cases.read_text(encoding="utf-8")))
    started = time.perf_counter()
    if args.operation == "train":
        parameters, ledger = train(args.scene, args.reference, args.bundle, cases, args.workers)
        result = {
            "schema": "rbq_reward_residual_training_v1",
            "parameters": parameters.tolist(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scene_sha256": digest(args.scene),
            "reference_sha256": digest(args.reference),
            "case_set_sha256": canonical_sha256(cases),
            "cpu_only": True,
            "teacher_assistance_disclosed": True,
            "actor_checkpoint": 9100,
            "training_ledger": ledger,
            "wall_s": time.perf_counter() - started,
        }
    else:
        digest_cases = canonical_sha256(cases)
        if digest_cases != args.expected_cases_sha256:
            parser.error("case commitment mismatch")
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
        parameters = np.asarray(policy["parameters"], dtype=float)
        jobs = [(variant, parameters, variant) for variant in VARIANTS]
        trials = evaluate_batch(
            args.scene,
            args.reference,
            args.bundle,
            cases,
            jobs,
            FULL_HORIZON_S,
            args.workers,
            args.state_dir,
        )
        result = {
            "schema": "rbq_reward_residual_evaluation_v1",
            "policy_sha256": digest(args.policy),
            "case_set_sha256": digest_cases,
            "parameters": parameters.tolist(),
            "variants": list(VARIANTS),
            "trials": sorted(trials, key=lambda row: (row["case_id"], row["variant"])),
            "wall_s": time.perf_counter() - started,
        }
    write_json_new(args.output, result)


if __name__ == "__main__":
    main()
