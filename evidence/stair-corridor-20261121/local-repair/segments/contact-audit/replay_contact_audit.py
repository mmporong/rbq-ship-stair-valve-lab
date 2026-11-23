"""Replay a segment while observing every post-step lever contact without changing state."""

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "src")]

from stair_mission_reference import file_sha256  # noqa: E402

from go2_wall import stair_valve  # noqa: E402

REPLAY_PATH = ROOT / "evidence" / "stair-corridor-20261121" / "tracked" / "segments" / "replay.py"
CHANNELS = ("pad_upper", "pad_lower", "nonpad")


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_replay_module():
    spec = importlib.util.spec_from_file_location(
        "tracked_segment_replay_contact_audit", REPLAY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ContactAudit:
    def __init__(self):
        self.steps = 0
        self.rows = []
        self._model_id = None
        self._lever = None
        self._category = None

    def configure(self, model):
        self._model_id = id(model)
        self._lever = model.geom(stair_valve.VALVE_LEVER).id
        upper = model.geom(stair_valve.GRIPPER_PADS[0]).id
        lower = model.geom(stair_valve.GRIPPER_PADS[1]).id
        self._category = {upper: 0, lower: 1}

    def observe(self, model, data):
        if id(model) != self._model_id:
            self.configure(model)
        mask = 0
        resultant = np.zeros(3, dtype=np.float64)
        normal = np.zeros(3, dtype=np.float64)
        force = np.empty(6, dtype=np.float64)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            first, second = int(contact.geom1), int(contact.geom2)
            if self._lever not in (first, second):
                continue
            other = second if first == self._lever else first
            category = self._category.get(other, 2)
            mujoco.mj_contactForce(model, data, contact_index, force)
            mask |= 1 << category
            resultant[category] = max(resultant[category], float(np.linalg.norm(force[:3])))
            normal[category] = max(normal[category], abs(float(force[0])))
        self.steps += 1
        if mask:
            tick = int(round(float(data.time) / float(model.opt.timestep)))
            self.rows.append((tick, mask, *resultant, *normal))

    def arrays(self):
        if not self.rows:
            return {
                "time_index": np.empty(0, dtype=np.int32),
                "contact_mask": np.empty(0, dtype=np.uint8),
                **{name + "_n": np.empty(0, dtype=np.float64) for name in CHANNELS},
                **{name + "_normal_n": np.empty(0, dtype=np.float64) for name in CHANNELS},
            }
        rows = np.asarray(self.rows, dtype=np.float64)
        return {
            "time_index": rows[:, 0].astype(np.int32),
            "contact_mask": rows[:, 1].astype(np.uint8),
            **{name + "_n": rows[:, 2 + index] for index, name in enumerate(CHANNELS)},
            **{name + "_normal_n": rows[:, 5 + index] for index, name in enumerate(CHANNELS)},
        }


def compare_segments(reference_path, replayed_path):
    with (
        np.load(reference_path, allow_pickle=False) as reference,
        np.load(replayed_path, allow_pickle=False) as replayed,
    ):
        reference_keys = list(reference.files)
        replayed_keys = list(replayed.files)
        arrays = {}
        for name in reference_keys:
            arrays[name] = {
                "shape": list(reference[name].shape),
                "equal": bool(np.array_equal(reference[name], replayed[name], equal_nan=True)),
                "sha256": array_sha256(replayed[name]),
            }
    return {
        "reference_sha256": file_sha256(reference_path),
        "replayed_sha256": file_sha256(replayed_path),
        "file_bytes_equal": reference_path.read_bytes() == replayed_path.read_bytes(),
        "keys_equal": reference_keys == replayed_keys,
        "all_arrays_equal": all(item["equal"] for item in arrays.values()),
        "arrays": arrays,
    }


def contact_summary(arrays, timestep):
    mask = arrays["contact_mask"]
    result = {}
    for index, name in enumerate(CHANNELS):
        present = (mask & (1 << index)) != 0
        force = arrays[name + "_n"]
        normal = arrays[name + "_normal_n"]
        if np.any(present):
            rows = np.flatnonzero(present)
            peak = rows[np.argmax(force[present])]
            peak_normal = rows[np.argmax(normal[present])]
            result[name] = {
                "contact_ticks": int(len(rows)),
                "first_time_s": float(arrays["time_index"][rows[0]] * timestep),
                "last_time_s": float(arrays["time_index"][rows[-1]] * timestep),
                "peak_resultant_n": float(force[peak]),
                "peak_resultant_time_s": float(arrays["time_index"][peak] * timestep),
                "peak_normal_n": float(normal[peak_normal]),
                "peak_normal_time_s": float(arrays["time_index"][peak_normal] * timestep),
            }
        else:
            result[name] = {
                "contact_ticks": 0,
                "first_time_s": None,
                "last_time_s": None,
                "peak_resultant_n": 0.0,
                "peak_resultant_time_s": None,
                "peak_normal_n": 0.0,
                "peak_normal_time_s": None,
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=("up", "down"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--walk", type=Path, required=True)
    parser.add_argument("--reference-segment", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    replay = load_replay_module()
    audit = ContactAudit()
    original_step = mujoco.mj_step

    def observed_step(model, data, *step_args, **step_kwargs):
        if step_args or step_kwargs:
            nstep = step_kwargs.get("nstep", step_args[0] if step_args else 1)
            if nstep != 1:
                raise RuntimeError("contact audit requires one physics step per mj_step call")
        original_step(model, data, *step_args, **step_kwargs)
        audit.observe(model, data)

    with tempfile.TemporaryDirectory(prefix=f"go2-contact-audit-{args.direction}-") as temp:
        replay_output = Path(temp) / "replay"
        mujoco.mj_step = observed_step
        try:
            replay.run(args.walk, args.direction, replay_output)
        finally:
            mujoco.mj_step = original_step
        replay_result = json.loads((replay_output / "result.json").read_text(encoding="utf-8"))
        comparison = compare_segments(args.reference_segment, replay_output / "segment.npz")

    arrays = audit.arrays()
    raw_path = args.output / "contacts.npz"
    with raw_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    timestep = float(replay_result["physics_step_s"])
    expected_steps = int(replay_result["metrics"]["gripper_audit"]["physics_steps"])
    result = {
        "scope": "Post-mj_step lever contact audit using existing MuJoCo constraints",
        "direction": args.direction,
        "walking_sha256": file_sha256(args.walk),
        "physics_step_s": timestep,
        "observed_physics_steps": audit.steps,
        "expected_physics_steps": expected_steps,
        "all_steps_observed": audit.steps == expected_steps,
        "sparse_contact_rows": int(len(arrays["time_index"])),
        "contact_mask_bits": {"pad_upper": 1, "pad_lower": 2, "nonpad": 4},
        "force_value": (
            "maximum resultant norm from mj_contactForce force[0:3] per category and tick"
        ),
        "normal_value": "maximum absolute mj_contactForce force[0] per category and tick",
        "contacts": contact_summary(arrays, timestep),
        "raw_sha256": file_sha256(raw_path),
        "segment_reproduction": comparison,
        "state_mutation": {
            "extra_mj_forward_calls": 0,
            "qpos_writes": 0,
            "qvel_writes": 0,
            "force_writes": 0,
            "command_writes": 0,
        },
    }
    if not result["all_steps_observed"]:
        raise RuntimeError("not every physics step was observed")
    if not comparison["file_bytes_equal"] or not comparison["all_arrays_equal"]:
        raise RuntimeError("instrumented replay changed the preserved 10 ms segment")
    with (args.output / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        f"{args.direction}: steps={audit.steps}, contact_rows={len(arrays['time_index'])}, "
        f"segment={comparison['replayed_sha256']}"
    )


if __name__ == "__main__":
    main()
