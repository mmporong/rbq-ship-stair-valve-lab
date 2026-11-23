"""Opt-in physical control profiles for the continuous stair mission."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np

from go2_wall import stairs, transition
from go2_wall.statics import torque_limits

RL_REPLANT_PROFILE = "rl-mid-replant-k3075-v1"
REPLANT_AFTER_LANDING = 15
REPLANT_LEG = "RL"
REPLANT_SECONDS = 2.8
SLIP_SPRING_GAIN_N_PER_M = 3075.0
SLIP_SPRING_ACTIVE_S = (116.65, 129.4)
TERMINAL_OVERTRAVEL_DEG = 1.3
STANCE_SPRING_GAIN_N_PER_M = 2500.0


def with_failed_replant(walking, time_s, *, archived_order=False):
    """Include the measured relanding failure in the walking verdict."""
    failure = {
        "stage": "prewalk",
        "time_s": time_s,
        "metric": "physical_replant",
        "measured": False,
        "limit": "released, positive measured lift, and reloaded within 0.035 m",
    }
    failures = [*walking["failures"], failure]
    if not archived_order:
        failures.sort(key=lambda item: (item["time_s"] is None, item["time_s"] or 0.0))
    first = failure if archived_order else failures[0]
    return {**walking, "verified": False, "first_failure": first, "failures": failures}


def _source_sha256(function) -> str:
    source = inspect.getsource(function).replace("\r\n", "\n").encode()
    return hashlib.sha256(source).hexdigest()


def replant_landings(ends):
    """Insert the physical RL relanding and shift every later scheduled landing."""
    trigger_t, stance, trigger_leg = ends[REPLANT_AFTER_LANDING]
    if trigger_leg != "RR" or abs(trigger_t - 78.8) > 1e-8:
        raise ValueError("The replant profile requires the fixed 78.8 s RR landing")
    shifted = [
        (time_s + (REPLANT_SECONDS if index > REPLANT_AFTER_LANDING else 0.0), state, leg)
        for index, (time_s, state, leg) in enumerate(ends)
    ]
    shifted.insert(
        REPLANT_AFTER_LANDING + 1,
        (trigger_t + REPLANT_SECONDS, stance, REPLANT_LEG),
    )
    return shifted


def apply_replant(geometry, segments, ends, walk):
    """Splice a real unload/swing/load cycle into the fixed walking track."""
    trigger_t, stance, _ = ends[REPLANT_AFTER_LANDING]
    base = segments[80]["base"]
    plan = [
        {
            "kind": "unload",
            "base": base,
            "stance": stance,
            "leg": REPLANT_LEG,
            "seconds": transition.HANDOFF_S,
        },
        {
            "kind": "swing",
            "base": base,
            "before": stance,
            "after": stance,
            "leg": REPLANT_LEG,
            "seconds": segments[73]["seconds"],
        },
        {
            "kind": "load",
            "base": base,
            "stance": stance,
            "leg": REPLANT_LEG,
            "seconds": transition.HANDOFF_S,
        },
    ]
    stairs.ARM = "rb1"
    stairs.use_stow()
    model, data = stairs.build_stairs(geometry["slope_deg"], geometry["rise_m"])
    refresh = transition.sample_segments(
        model,
        data,
        plan,
        torque_limits("rb1"),
        geometry["arm_stow_rad"],
    )
    if abs(float(refresh["time"][-1]) - REPLANT_SECONDS) > 1e-9:
        raise ValueError("The physical replant duration changed")
    boundary = int(np.argmin(abs(walk["time"] - trigger_t)))
    if abs(float(walk["time"][boundary]) - trigger_t) > 1e-7:
        raise ValueError("Walking reference lacks the replant boundary")
    spliced = {}
    for key in walk:
        before = walk[key][: boundary + 1]
        inserted = refresh[key][1:]
        after = walk[key][boundary + 1 :]
        if key == "time":
            inserted = trigger_t + inserted
            after = REPLANT_SECONDS + after
        spliced[key] = np.concatenate((before, inserted, after))
    return spliced, replant_landings(ends), {
        "name": RL_REPLANT_PROFILE,
        "trigger_landing_index": REPLANT_AFTER_LANDING,
        "trigger_time_s": trigger_t,
        "trigger_leg": "RR",
        "replant_leg": REPLANT_LEG,
        "replant_time_s": trigger_t + REPLANT_SECONDS,
        "duration_s": REPLANT_SECONDS,
        "kind": "physical same-tread unload, swing, and relanding",
        "slip_spring_gain_n_per_m": SLIP_SPRING_GAIN_N_PER_M,
        "slip_spring_active_s": list(SLIP_SPRING_ACTIVE_S),
        "slip_spring_leg": REPLANT_LEG,
        "terminal_overtravel_deg": TERMINAL_OVERTRAVEL_DEG,
        "stance_spring_gain_n_per_m": STANCE_SPRING_GAIN_N_PER_M,
        "apply_replant_sha256_lf": _source_sha256(apply_replant),
        "replant_landings_sha256_lf": _source_sha256(replant_landings),
        "source_file": Path(__file__).relative_to(Path(__file__).parents[1]).as_posix(),
    }
