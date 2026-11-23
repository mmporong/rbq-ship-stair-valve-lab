"""Count mid-swing poses on a ship stair for gait orders, foot widths, and clearances (T0)."""

import multiprocessing

from go2_wall import pose_search as P
from go2_wall import stairs as S
from go2_wall import transition as T


def run(job):
    slope, rise, gap, order, foot_y, clear = job
    P.FOOT_Y = foot_y
    S.FOOT_Y = foot_y
    T.CLEARANCE_M = clear
    S._init(slope, rise)
    start, seq = S.cycle(2, gap, slope, rise, 0.127, 0.022, order=order)
    befores = [start] + [a for _, a in seq[:-1]]
    counts = []
    for k, (leg, after) in enumerate(seq):
        n = 0
        for off in S.GRID["pitch_offsets"]:
            n += len(S._quick((off, befores[k], after, leg, slope)))
        counts.append((leg, n))
    return job, counts


if __name__ == "__main__":
    jobs = []
    for order in (
        ("RL", "RR", "FL", "FR"),
        ("FL", "FR", "RL", "RR"),
        ("RL", "FL", "RR", "FR"),
        ("FR", "RL", "FL", "RR"),
    ):
        for fy in (0.142, 0.20):
            jobs.append((50, 0.2, 2, order, fy, 0.03))
    jobs.append((50, 0.2, 2, ("RL", "RR", "FL", "FR"), 0.20, 0.015))
    jobs.append((60, 0.2, 2, ("RL", "FL", "RR", "FR"), 0.20, 0.015))
    with multiprocessing.Pool(10) as pool:
        for job, counts in pool.imap(run, jobs):
            print(
                job[0],
                job[1],
                "order",
                "-".join(job[3]),
                "foot_y",
                job[4],
                "clear",
                job[5],
                counts,
                flush=True,
            )
