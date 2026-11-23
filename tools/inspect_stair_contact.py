"""Locate the first excessive contact and the peak contact in a recorded stair mission."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from render_stair_mission import build_model
from replay_stair_sequence import NONFOOT_CONTACT_LIMIT_N
from stair_mission_reference import file_sha256


def inspect(folder):
    scene_path = folder / "scene.json"
    raw_path = folder / "mission.npz"
    result = json.loads((folder / "result.json").read_bytes())
    if (
        file_sha256(raw_path) != result["raw_sha256"]
        or file_sha256(scene_path) != result["scene_sha256"]
    ):
        raise ValueError("The raw recording or scene differs from its recorded fingerprint")
    model, data = build_model(json.loads(scene_path.read_bytes()))
    eligible = result["metrics"]["unexpected_contacts_n"]
    report = {
        "raw_sha256": result["raw_sha256"],
        "scene_sha256": result["scene_sha256"],
        "limit_n": NONFOOT_CONTACT_LIMIT_N,
        "events": [],
    }
    with np.load(raw_path, allow_pickle=False) as raw:
        force = raw["unexpected_contact_force_n"]
        violations = np.flatnonzero(force > NONFOOT_CONTACT_LIMIT_N)
        if not len(violations):
            return report
        indices = [int(violations[0]), int(np.argmax(force))]
        states = {key: raw[key][indices] for key in ("qpos", "qvel", "ctrl", "time_s")}
    for row, (name, index) in enumerate(zip(("first_over_limit", "peak"), indices, strict=True)):
        data.qpos[:] = states["qpos"][row]
        data.qvel[:] = states["qvel"][row]
        data.ctrl[:] = states["ctrl"][row]
        mujoco.mj_forward(model, data)
        contacts = []
        for number, contact in enumerate(data.contact):
            geoms = [model.geom(int(geom)) for geom in contact.geom]
            bodies = [model.body(int(geom.bodyid[0])).name for geom in geoms]
            if "+".join(sorted(bodies)) in eligible:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, number, wrench)
                contacts.append(
                    {
                        "bodies": bodies,
                        "geoms": [geom.name or None for geom in geoms],
                        "geom_ids": [int(geom.id) for geom in geoms],
                        "location_world_m": contact.pos.tolist(),
                        "signed_distance_m": float(contact.dist),
                        "recomputed_force_n": float(np.linalg.norm(wrench[:3])),
                    }
                )
        report["events"].append(
            {
                "event": name,
                "sample_index": index,
                "state_time_s": float(states["time_s"][row]),
                "recorded_force_n": float(force[index]),
                "contacts_in_forward_solve": sorted(
                    contacts, key=lambda c: -c["recomputed_force_n"]
                ),
            }
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = inspect(args.run_dir)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
