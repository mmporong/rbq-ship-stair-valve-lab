"""Dump selected TensorBoard scalars of an rsl_rl run at fixed iterations as JSON.

Usage (inside the training venv): python tb_trend.py <run dir> <out.json>
"""

import json
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

KEYS = (
    "Curriculum/terrain_levels",
    "Train/mean_episode_length",
    "Train/mean_reward",
    "Episode_Reward/track_linear_velocity",
    "Episode_Termination/time_out",
    "Episode_Termination/fell_over",
    "Episode_Termination/illegal_contact",
)
POINTS = (0, 1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 499, 500, 1000, 2000, 3000, 3999)


def main():
    run, out = sys.argv[1:3]
    events = EventAccumulator(run, size_guidance={"scalars": 0})
    events.Reload()
    tags = set(events.Tags()["scalars"])
    trend = {}
    for key in KEYS:
        if key in tags:
            values = {e.step: e.value for e in events.Scalars(key)}
            trend[key] = {p: round(values[p], 4) for p in POINTS if p in values}
    with open(out, "w") as f:
        json.dump({"run": run, "scalars": trend}, f, indent=2)


if __name__ == "__main__":
    main()
