"""Identify the exact controller, route inputs and robot model used by a bypass replay."""

import hashlib
import json
import platform

import mujoco
import numpy as np
import scipy
from stair_mission_route import ASCENT, FRONT_PAIR, PLANNER_SOURCES, VALVE, source_hashes

from go2_wall.robot import ROOT


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def provenance(*extra_paths: str) -> dict:
    paths = {
        *PLANNER_SOURCES,
        *extra_paths,
        "tools/stair_bypass_inputs.py",
        "src/go2_wall/grasp_motion.py",
        "src/go2_wall/gripper_control.py",
        "src/go2_wall/hold.py",
        "src/go2_wall/stair_valve.py",
        "configs/d1_parameters.json",
        "configs/assets.lock.json",
        "pyproject.toml",
    }
    route_inputs = (
        ASCENT / "target-round-trip-geometry.json",
        ASCENT / "target-round-trip-plan.json",
        FRONT_PAIR,
        VALVE,
    )
    paths.update(p.relative_to(ROOT).as_posix() for p in route_inputs)
    assets = json.loads((ROOT / "configs/assets.lock.json").read_bytes())["rbq"]
    for name, expected in assets["files"].items():
        if sha256_file(ROOT / ".assets/rbq" / name) != expected:
            raise ValueError(f"RBQ asset differs from its locked bytes: {name}")
    return {
        "source_sha256_lf": source_hashes(sorted(paths)),
        "rbq_asset_commit": assets["commit"],
        "rbq_asset_files": assets["files"],
        "runtime": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
