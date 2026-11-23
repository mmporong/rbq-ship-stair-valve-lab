"""Integration error of the wall-climb and lever mission, against a fine RK4 reference.

MuJoCo 4.0 made RK4 fourth-order accurate for free-joint orientation (3.x combined the stage
angular velocities without the Munthe-Kaas correction, so the floating base was second
order). Here the whole mission (climb, lever, climb down, 73 s) runs with several integrators
and timesteps, and joint torques, bracket stress, and base motion are compared with RK4 at the
smallest timestep. Control and logging keep their rates in seconds, so only the integration
changes.

  python tools/integrator_study.py evidence/<run-id>/mujoco-4.0 [--variants euler:0.002 ...]

Running the same command with the WSL training environment (MuJoCo 3.5) into another folder
gives the 3.x comparison; observed_order in the output shows whether RK4 gains its order.

Each variant writes <out>/<name>/mission.json, mission.npz, loads.npz; the comparison goes
to <out>/integrators.json.
"""

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2_wall import lever as lv  # noqa: E402
from go2_wall import loads  # noqa: E402

PLAN = ROOT / "evidence/s6-transition/plan.json"
PITCH_DEG = 40.0
DEFAULT_VARIANTS = (
    "euler:0.002",
    "implicitfast:0.002",
    "rk4:0.002",
    "euler:0.001",
    "rk4:0.001",
    "euler:0.0005",
    "rk4:0.0005",
    "rk4:0.00025",
)


def name_of(variant: str) -> str:
    integrator, dt = variant.split(":")
    return f"{integrator}_{float(dt) * 1e3:g}ms"


def run_variant(job: tuple[str, Path]) -> dict:
    variant, out = job
    integrator, dt = variant.split(":")
    folder = out / name_of(variant)
    folder.mkdir(parents=True, exist_ok=True)
    saved = json.loads(PLAN.read_text(encoding="utf-8"))
    start = time.perf_counter()
    result, track, log = lv.run(
        saved, pitch_deg=PITCH_DEG, physics={"integrator": integrator, "timestep": float(dt)}
    )
    seconds = time.perf_counter() - start
    result["wall_clock_s"] = seconds
    result["mujoco"] = mujoco.__version__
    (folder / "mission.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    npz = folder / "mission.npz"
    np.savez_compressed(npz, **track, **{f"log_{k}": v for k, v in log.items()})
    result["loads"] = loads.analyse(npz)
    (folder / "loads.json").write_text(json.dumps(result["loads"], indent=1), encoding="utf-8")
    print(f"{name_of(variant)}: {seconds:.0f} s", flush=True)
    return result


def series(folder: Path) -> dict:
    mission = np.load(folder / "mission.npz")
    load = np.load(folder / "loads.npz")
    return {
        "time": mission["log_time"],
        "torque": mission["log_torque"],
        "base": mission["log_base"][:, :3],
        "bracket": load["bracket_mpa"],
        "bend": load["joint_bend"],
    }


def observed_order(rows: dict[str, dict], reference: str) -> dict:
    """log2 of the error ratio when the timestep halves, per integrator and metric.

    A first-order method gives about 1, and RK4 on a smooth problem about 4; contacts and the
    50 Hz control steps can cap what is observed.

    >>> rows = {
    ...     "euler_2ms": {"vs_reference": {"base_max_mm": 4.0, "torque_rms_nm": 0.8}},
    ...     "euler_1ms": {"vs_reference": {"base_max_mm": 2.0, "torque_rms_nm": 0.4}},
    ...     "rk4_0.5ms": {"vs_reference": {"base_max_mm": 0.0, "torque_rms_nm": 0.0}},
    ... }
    >>> observed_order(rows, "rk4_0.5ms")
    {'euler': {'2ms->1ms': {'base_max_mm': 1.0, 'torque_rms_nm': 1.0}}}
    """
    ladders: dict[str, list[tuple[float, str]]] = {}
    for name in rows:
        if name == reference:
            continue
        integrator, dt = name.rsplit("_", 1)
        ladders.setdefault(integrator, []).append((float(dt.removesuffix("ms")), name))
    out = {}
    for integrator, ladder in ladders.items():
        ladder.sort(reverse=True)
        steps = {}
        for (dt_a, a), (dt_b, b) in zip(ladder, ladder[1:]):
            if not np.isclose(dt_a, 2 * dt_b):
                continue
            ratio = {}
            for metric in ("base_max_mm", "torque_rms_nm"):
                ea, eb = rows[a]["vs_reference"][metric], rows[b]["vs_reference"][metric]
                ratio[metric] = round(float(np.log2(ea / eb)), 2) if ea > 0 and eb > 0 else None
            steps[f"{dt_a:g}ms->{dt_b:g}ms"] = ratio
        if steps:
            out[integrator] = steps
    return out


def compare(out: Path, names: list[str], results: dict[str, dict], reference: str) -> dict:
    ref = series(out / reference)
    rows = {}
    for name in names:
        cur = series(out / name)
        n = min(len(cur["time"]), len(ref["time"]))
        assert np.allclose(cur["time"][:n], ref["time"][:n], atol=1e-6), name
        r = results[name]
        rows[name] = {
            "mujoco": r["mujoco"],
            "wall_clock_s": round(r["wall_clock_s"], 1),
            "mission_passed": r["round_trip"]["passed"] and r["lever"]["passed"],
            "fell": r["round_trip"]["fell"],
            "bracket_peak_mpa": r["loads"]["bracket"]["peak_von_mises_mpa"],
            "max_torque_ratio": round(r["round_trip"]["max_ratio"], 4),
            "max_base_track_mm": round(r["round_trip"]["max_base_track_mm"], 3),
            "lever_max_deg": round(r["lever"]["lever_max_deg"], 3),
            "max_stance_slip_mm": max(r["round_trip"]["max_stance_slip_mm"].values()),
            "vs_reference": {
                "torque_max_abs_nm": round(
                    float(np.abs(cur["torque"][:n] - ref["torque"][:n]).max()), 4
                ),
                "torque_rms_nm": round(
                    float(np.sqrt(np.mean((cur["torque"][:n] - ref["torque"][:n]) ** 2))), 5
                ),
                "bracket_max_abs_mpa": round(
                    float(np.abs(cur["bracket"][:n] - ref["bracket"][:n]).max()), 4
                ),
                "joint_bend_max_abs_nm": round(
                    float(np.abs(cur["bend"][:n] - ref["bend"][:n]).max()), 4
                ),
                "base_max_mm": round(
                    float(np.linalg.norm(cur["base"][:n] - ref["base"][:n], axis=1).max()) * 1e3, 4
                ),
                "samples": int(n),
            },
        }
    return {
        "observed_order": observed_order(rows, reference),
        "reference": reference,
        "mission": f"{PLAN.relative_to(ROOT).as_posix()}, lever pitch {PITCH_DEG:g} deg",
        "note": "Control (QP every 0.02 s) and logging (50 Hz) keep their rate in seconds.",
        "variants": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--reference", default=None, help="variant name; default: the last one")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = [name_of(v) for v in args.variants]
    if args.compare_only:
        results = {}
        for name in names:
            results[name] = json.loads((args.out / name / "mission.json").read_text("utf-8"))
            results[name]["loads"] = json.loads((args.out / name / "loads.json").read_text("utf-8"))
    else:
        with multiprocessing.Pool(args.workers) as pool:
            done = pool.map(run_variant, [(v, args.out) for v in args.variants], chunksize=1)
        results = dict(zip(names, done))
    table = compare(args.out, names, results, args.reference or names[-1])
    (args.out / "integrators.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    for name, row in table["variants"].items():
        print(name, json.dumps(row))


if __name__ == "__main__":
    main()
