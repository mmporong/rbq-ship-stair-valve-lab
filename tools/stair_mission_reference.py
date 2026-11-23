"""Tie independently defined foot placements and landing times to a saved walking reference."""

import hashlib
from pathlib import Path

import numpy as np

from go2_wall.robot import LEGS, ROOT

CORRIDOR_PROFILE = "fixed-side-72-upper-v2"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def corridor_lateral_offsets(x):
    """Shift the next footholds after the working stance and translate the upper walk by 72 mm."""

    def smooth(value):
        z = np.clip(value, 0.0, 1.0)
        return z * z * (3.0 - 2.0 * z)

    anchor_x = np.array([0.70121572, 0.70121572, 0.38235786, 0.38235786])
    lower_shift = np.array([0.262 - 0.19285, -0.118 + 0.19285] * 2)
    entered = smooth((x - anchor_x) / 0.15943)
    upper = smooth((x - 1.495) / 0.135)
    return entered * ((1.0 - upper) * lower_shift + upper * 0.072)


def reference_samples(phase, grid):
    """Locate refined phase samples on the original walking grid, retaining every original knot."""
    phase = np.asarray(phase, dtype=float)
    if (
        phase.ndim != 1
        or len(phase) < len(grid)
        or not np.all(np.isfinite(phase))
        or np.any(np.diff(phase) <= 0)
        or abs(phase[0] - grid[0]) > 1e-8
        or abs(phase[-1] - grid[-1]) > 1e-8
    ):
        raise ValueError("Invalid walking phase or missing original route knot")
    left = np.searchsorted(grid, phase + 1e-8, side="right") - 1
    knot = np.isclose(phase, grid[left], atol=1e-8, rtol=0)
    if not np.array_equal(left[knot], np.arange(len(grid))):
        raise ValueError("Missing or duplicated original route knot")
    right = np.minimum(left + 1, len(grid) - 1)
    fraction = np.zeros_like(phase)
    fraction[~knot] = (phase[~knot] - grid[left[~knot]]) / (grid[right[~knot]] - grid[left[~knot]])
    return left, right, fraction, knot


def scheduled_landings(cache, ends, anchor, slope, rise):
    """Read a monotone physical clock and apply the declared foothold rule to the nominal goals."""
    if "footprint_profile" not in cache:
        return ends, None
    profile = str(cache["footprint_profile"])
    if profile != CORRIDOR_PROFILE or (slope, rise) != (50.0, 0.19):
        raise ValueError("Unknown walking footprint profile or unsupported stair geometry")
    if "tip" in cache:
        tip = np.asarray(cache["tip"])
        if tip.shape != (len(cache["time"]), 3) or np.any(tip != 0.0):
            raise ValueError(
                "The unloaded corridor requires an Nx3 prescribed tip force of exactly zero"
            )
    nominal = np.asarray(cache["reference_time"], dtype=float)
    clock = np.asarray(cache["time"], dtype=float)
    if (
        nominal.ndim != 1
        or clock.shape != nominal.shape
        or len(clock) < 2
        or not np.all(np.isfinite(nominal))
        or not np.all(np.isfinite(clock))
        or nominal[0] != 0.0
        or clock[0] != 0.0
        or np.any(np.diff(clock) <= 0)
        or not np.allclose(clock / 0.005, np.rint(clock / 0.005), atol=1e-6, rtol=0)
        or abs(nominal[-1] - ends[-1][0]) > 1e-7
    ):
        raise ValueError("Invalid walking reference clock or route endpoint")
    grid = np.arange(round(ends[-1][0] / 0.02) + 1) * 0.02
    _, _, _, knot = reference_samples(nominal, grid)
    before_task = nominal <= ends[anchor][0] + 1e-8
    if not knot[before_task].all() or not np.allclose(
        nominal[before_task], clock[before_task], atol=1e-8, rtol=0
    ):
        raise ValueError("Walking clock changes before the fixed manipulation stance")
    mapped = []
    for number, (t, stance, leg) in enumerate(ends):
        index = int(np.argmin(abs(nominal - t)))
        if abs(nominal[index] - t) > 1e-7:
            raise ValueError("A planned landing does not coincide with a reference sample")
        if number <= anchor:
            mapped.append((t, stance, leg))
            continue
        feet = np.array([stance[name][0] for name in LEGS], dtype=float)
        feet[:, 1] += corridor_lateral_offsets(feet[:, 0])
        target = {name: (feet[k], stance[name][1]) for k, name in enumerate(LEGS)}
        mapped.append((float(clock[index]), target, leg))
    return mapped, profile


def describe_reference(path, profile):
    path = Path(path).resolve()
    repo_file = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else None
    return {"sha256": file_sha256(path), "repo_file": repo_file, "footprint_profile": profile}


def verified_landings(reference, path, ends, anchor, slope, rise):
    """Rebuild the prescribed targets before scoring any measurements."""
    if path is None:
        relative = reference.get("repo_file")
        if not relative:
            raise ValueError("Supply --walk for this recorded walking reference")
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Recorded walking reference points outside the repository")
    path = Path(path)
    if file_sha256(path) != reference["sha256"]:
        raise ValueError("Walking reference fingerprint mismatch")
    with np.load(path, allow_pickle=False) as cache:
        if (float(cache["slope_deg"]), float(cache["rise_m"])) != (slope, rise):
            raise ValueError("Walking reference geometry mismatch")
        mapped, profile = scheduled_landings(cache, ends, anchor, slope, rise)
    if profile != reference["footprint_profile"]:
        raise ValueError("Walking footprint profile differs from the saved experiment")
    return mapped
