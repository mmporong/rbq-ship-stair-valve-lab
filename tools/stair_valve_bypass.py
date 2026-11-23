"""Search a local, support-preserving body/foot bypass around the fixed shut valve."""

import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

from go2_wall import pose_search, stair_valve, transition
from go2_wall.robot import LEGS
from go2_wall.statics import Contact, joint_names, solve_static, torque_limits

FOOT_Y_LIMIT_M = 0.275  # 610 mm clear width minus the 30 mm foot radius.


def adjusted_feet(
    feet: np.ndarray,
    offset_m: float = 0.0,
    width_delta_m: float = 0.0,
    *,
    x_delta_m: np.ndarray | None = None,
    y_delta_m: np.ndarray | None = None,
) -> dict:
    """Shift and narrow/widen cached contacts without changing their x/z landing coordinates."""
    out = {}
    x_delta_m = np.zeros(4) if x_delta_m is None else np.asarray(x_delta_m)
    y_delta_m = np.zeros(4) if y_delta_m is None else np.asarray(y_delta_m)
    for index, (leg, point) in enumerate(zip(LEGS, feet, strict=True)):
        target = point.copy()
        target[0] += x_delta_m[index]
        target[1] += (
            offset_m + (width_delta_m if leg[1] == "L" else -width_delta_m) + y_delta_m[index]
        )
        if abs(target[1]) > FOOT_Y_LIMIT_M:
            raise ValueError("foot leaves the 610 mm tread support width")
        out[leg] = target
    return out


def _pitch(qpos: np.ndarray) -> float:
    return float(-2 * np.arctan2(qpos[5], qpos[3]))


def horizontal_stance(feet: dict, active: np.ndarray, lift_leg=None) -> dict:
    """Keep scheduled x/z landings while assigning every horizontal tread an upward normal."""
    return {
        leg: (feet[leg], "floor")
        for leg, supported in zip(LEGS, active, strict=True)
        if supported and leg != lift_leg
    }


def support_margin_m(points: np.ndarray, com_xy: np.ndarray) -> float:
    """Signed distance to the convex support polygon: positive means inside."""
    points = sorted({tuple(point) for point in np.asarray(points)})
    if len(points) < 3:
        return -np.inf

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    margins = []
    for first, second in zip(hull, np.roll(hull, -1, axis=0), strict=True):
        edge = second - first
        delta = com_xy - first
        margins.append((edge[0] * delta[1] - edge[1] * delta[0]) / np.linalg.norm(edge))
    return float(min(margins))


def _friction_diagnostic(model, data, contacts, limits) -> dict:
    relaxed = [Contact(item.leg, item.normal, 10.0) for item in contacts]
    solution = solve_static(model, data, relaxed, limits * 100.0)
    if solution is None:
        return {"static_bottleneck": "support_wrench"}
    actual_ratio = float(np.max(np.abs(solution.torque) / limits))
    uses, normals = [], []
    for contact in relaxed:
        force = solution.forces[contact.leg]
        normal = float(force @ contact.normal)
        tangent = np.linalg.norm(force - normal * contact.normal)
        normals.append(normal)
        uses.append(float(tangent / max(normal, 1e-9) / 0.42))
    return {
        "static_bottleneck": "friction" if actual_ratio <= 1.0 else "friction_and_joint_limit",
        "relaxed_actual_torque_ratio": actual_ratio,
        "relaxed_max_normal_n": max(normals),
        "required_friction_use": max(uses),
    }


def lever_clearance_m(model, data) -> float:
    """Minimum signed distance from the fixed lever to any robot collision geometry."""
    lever = model.geom(stair_valve.VALVE_LEVER).id
    robot_root = model.body_rootid[model.body("base").id]
    segment = np.empty(6)
    gaps = []
    for geom in range(model.ngeom):
        if geom == lever or model.body_rootid[model.geom_bodyid[geom]] != robot_root:
            continue
        enabled = (model.geom_contype[lever] & model.geom_conaffinity[geom]) or (
            model.geom_contype[geom] & model.geom_conaffinity[lever]
        )
        if not enabled:
            continue
        gaps.append(mujoco.mj_geomDistance(model, data, lever, geom, 1.0, segment))
    return float(min(gaps))


def unexpected_collision(model, data) -> dict | None:
    feet = {model.geom(leg).id for leg in LEGS}
    world = set(range(model.ngeom)) - {
        geom
        for geom in range(model.ngeom)
        if model.body_rootid[model.geom_bodyid[geom]] == model.body_rootid[model.body("base").id]
    }
    rows = []
    for contact in data.contact[: data.ncon]:
        pair = {int(contact.geom1), int(contact.geom2)}
        if contact.dist >= -1e-4 or (pair & feet and pair & world):
            continue
        names = []
        for geom in (int(contact.geom1), int(contact.geom2)):
            body_name = model.body(model.geom_bodyid[geom]).name
            names.append(model.geom(geom).name or f"{body_name}:geom_{geom}")
        rows.append({"geoms": names, "penetration_m": float(-contact.dist)})
    return max(rows, key=lambda item: item["penetration_m"], default=None)


def evaluate_candidate(model, data, row: dict, candidate: dict, rise_m: float) -> dict:
    """Evaluate IK, collision and static support at one cached physical sample."""
    data.qpos[:] = model.qpos0
    data.qpos[: len(row["qpos"])] = row["qpos"]
    base = data.qpos[:3].copy()
    pose_search.set_base(
        model,
        data,
        _pitch(row["qpos"]) + np.radians(candidate.get("pitch_delta_deg", 0.0)),
        base + [candidate.get("body_x_m", 0.0), candidate["body_y_m"], candidate["body_z_m"]],
        yaw=np.radians(candidate.get("yaw_deg", 0.0)),
        roll=np.radians(candidate["roll_deg"]),
    )
    try:
        feet = adjusted_feet(
            row["feet"],
            candidate.get("foot_offset_m", 0.0),
            candidate.get("foot_width_delta_m", 0.0),
            x_delta_m=candidate.get("foot_x_delta_m"),
            y_delta_m=candidate.get("foot_y_delta_m"),
        )
    except ValueError:
        return {"ok": False, "reason": "foot_width"}
    error = pose_search.solve_leg_ik(model, data, feet, reset=False)
    if error > 2e-3:
        return {"ok": False, "reason": "ik", "ik_error_m": error}
    hinge = model.joint(stair_valve.VALVE_HINGE).qposadr[0]
    data.qpos[hinge] = np.pi / 2
    mujoco.mj_forward(model, data)
    lever = model.geom(stair_valve.VALVE_LEVER).id
    overlap = max(
        (max(0.0, -float(c.dist)) for c in data.contact[: data.ncon] if lever in c.geom),
        default=0.0,
    )
    clearance = lever_clearance_m(model, data)
    if overlap > 1e-4 or not pose_search.clear_of_collisions(model, data):
        return {
            "ok": False,
            "reason": "collision",
            "overlap_m": overlap,
            "lever_clearance_m": clearance,
            "unexpected_collision": unexpected_collision(model, data),
        }
    lift_leg = candidate.get("lift_leg")
    stance = horizontal_stance(feet, row["stance"], lift_leg)
    contacts = transition.contacts_for(stance)
    limits = torque_limits("rb1")
    solution = solve_static(model, data, contacts, limits, smooth=True)
    if solution is None:
        points = np.array([feet[leg][:2] for leg in stance])
        return {
            "ok": False,
            "reason": "static",
            "support_margin_m": support_margin_m(points, data.subtree_com[1, :2]),
            **_friction_diagnostic(model, data, contacts, limits),
        }
    if solution.ratio > 1.0:
        return {"ok": False, "reason": "joint_saturation", "ratio": float(solution.ratio)}
    return {
        "ok": True,
        "ratio": float(solution.ratio),
        "ik_error_m": float(error),
        "lever_clearance_m": clearance,
    }


def candidate_grid():
    for body_y, roll, body_z, foot_offset, width in itertools.product(
        (0.10, 0.14, 0.18, 0.22, 0.26),
        (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0),
        (-0.10, -0.05, 0.0, 0.05),
        (0.0, 0.04, 0.08),
        (-0.08, -0.04, 0.0, 0.04),
    ):
        yield {
            "body_y_m": body_y,
            "roll_deg": roll,
            "body_z_m": body_z,
            "foot_offset_m": foot_offset,
            "foot_width_delta_m": width,
        }


def multistart_grid(count: int, seed: int):
    """Sample both sides plus body attitude and independent foot placement freedoms."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        yield {
            "body_x_m": float(rng.uniform(-0.12, 0.12)),
            "body_y_m": float(rng.choice((-1.0, 1.0)) * rng.uniform(0.06, 0.27)),
            "body_z_m": float(rng.uniform(-0.12, 0.08)),
            "pitch_delta_deg": float(rng.uniform(-12.0, 12.0)),
            "yaw_deg": float(rng.uniform(-15.0, 15.0)),
            "roll_deg": float(rng.uniform(-18.0, 18.0)),
            "foot_x_delta_m": rng.uniform(-0.03, 0.03, 4).round(5).tolist(),
            "foot_y_delta_m": rng.uniform(-0.08, 0.08, 4).round(5).tolist(),
            "lift_leg": rng.choice((None, None, None, *LEGS)),
        }


def scaled_candidate(candidate: dict, scale: float) -> dict:
    out = {}
    for name, value in candidate.items():
        if name in {"ok", "ratio", "ik_error_m", "lift_leg"}:
            continue
        if isinstance(value, list):
            out[name] = (scale * np.asarray(value)).tolist()
        elif isinstance(value, (int, float)):
            out[name] = scale * value
    out["lift_leg"] = None
    return out


def validate_neighbour_path(model, data, walk, index: int, candidate: dict, rise_m: float) -> dict:
    """Ramp into a four-foot bypass across the preceding/current/following plan segments."""
    target_segment = int(walk["segment"][index])
    selected = np.flatnonzero(
        (walk["segment"] >= target_segment - 1) & (walk["segment"] <= target_segment + 1)
    )
    selected = selected[::5]
    first, last = int(selected[0]), int(selected[-1])
    failures, worst_ratio = [], 0.0
    for row_index in selected:
        segment = int(walk["segment"][row_index])
        if segment == target_segment - 1:
            scale = (row_index - first) / max(index - first, 1)
        elif segment == target_segment + 1:
            scale = (last - row_index) / max(last - index, 1)
        else:
            scale = 1.0
        row = {name: walk[name][row_index] for name in ("qpos", "feet", "stance")}
        result = evaluate_candidate(model, data, row, scaled_candidate(candidate, scale), rise_m)
        if result["ok"]:
            worst_ratio = max(worst_ratio, result["ratio"])
        else:
            failures.append(
                {
                    "time_s": float(walk["time"][row_index]),
                    "segment": segment,
                    "scale": float(scale),
                    **result,
                }
            )
            break
    return {
        "segments": [target_segment - 1, target_segment, target_segment + 1],
        "sample_stride_s": float(5 * np.median(np.diff(walk["time"]))),
        "passed": not failures,
        "worst_ratio": worst_ratio,
        "first_failure": failures[0] if failures else None,
    }


BODY_KEYS = (
    "body_x_m",
    "body_y_m",
    "body_z_m",
    "pitch_delta_deg",
    "yaw_deg",
    "roll_deg",
)


def zero_candidate() -> dict:
    return {
        **dict.fromkeys(BODY_KEYS, 0.0),
        "foot_x_delta_m": [0.0] * 4,
        "foot_y_delta_m": [0.0] * 4,
        "lift_leg": None,
    }


def _continuation_neighbours(candidate: dict, stance: np.ndarray, rng, branches: int):
    yield candidate
    for _ in range(branches - 1):
        out = {**candidate}
        steps = (0.012, 0.016, 0.012, 1.5, 1.5, 1.8)
        bounds = (0.14, 0.29, 0.14, 15.0, 18.0, 20.0)
        for name, step, bound in zip(BODY_KEYS, steps, bounds, strict=True):
            out[name] = float(np.clip(candidate[name] + rng.normal(0.0, step), -bound, bound))
        x = np.asarray(candidate["foot_x_delta_m"], dtype=float).copy()
        y = np.asarray(candidate["foot_y_delta_m"], dtype=float).copy()
        for leg_index, active in enumerate(stance):
            if not active:
                x[leg_index] = np.clip(x[leg_index] + rng.normal(0.0, 0.008), -0.033, 0.033)
                y[leg_index] = np.clip(y[leg_index] + rng.normal(0.0, 0.012), -0.08, 0.08)
        out["foot_x_delta_m"] = x.round(6).tolist()
        out["foot_y_delta_m"] = y.round(6).tolist()
        yield out


def _candidate_effort(candidate: dict) -> float:
    body = np.array([candidate[name] for name in BODY_KEYS], dtype=float)
    scale = np.array([0.14, 0.29, 0.14, 15.0, 18.0, 20.0])
    feet = np.r_[candidate["foot_x_delta_m"], candidate["foot_y_delta_m"]]
    return float(np.linalg.norm(body / scale) + np.linalg.norm(feet / 0.08))


def _beam_score(candidate: dict, result: dict) -> float:
    effort = _candidate_effort(candidate)
    return 20.0 * result["lever_clearance_m"] - 0.4 * result["ratio"] - 0.04 * effort


def continuation_search(
    model,
    data,
    walk,
    rise_m: float,
    *,
    start_s: float,
    end_s: float,
    sample_s: float,
    beam_width: int,
    branches: int,
    seed: int,
    initial_candidates: list[dict] | None = None,
) -> tuple[dict, dict | None]:
    """Continue locally feasible offsets while only moving the currently swinging foot."""
    times = np.arange(start_s, end_s + sample_s / 2, sample_s)
    indices = np.unique(np.searchsorted(walk["time"], times).clip(max=len(walk["time"]) - 1))
    rng = np.random.default_rng(seed)
    beam = [
        {"candidate": candidate, "path": []}
        for candidate in (initial_candidates or [zero_candidate()])
    ]
    first_failure = None
    evaluated = 0
    for index in indices:
        row = {name: walk[name][index] for name in ("qpos", "feet", "stance")}
        next_beam = []
        for state in beam:
            for candidate in _continuation_neighbours(
                state["candidate"], row["stance"], rng, branches
            ):
                evaluated += 1
                result = evaluate_candidate(model, data, row, candidate, rise_m)
                if not result["ok"]:
                    continue
                record = {
                    "time_s": float(walk["time"][index]),
                    "candidate": candidate,
                    "ratio": result["ratio"],
                    "lever_clearance_m": result["lever_clearance_m"],
                    "qpos": data.qpos.copy(),
                }
                next_beam.append(
                    {
                        "candidate": candidate,
                        "path": [*state["path"], record],
                        "score": _beam_score(candidate, result),
                    }
                )
        if not next_beam:
            first_failure = {"time_s": float(walk["time"][index]), "attempts": evaluated}
            break
        next_beam.sort(key=lambda item: item["score"], reverse=True)
        anchor = min(next_beam, key=lambda item: _candidate_effort(item["candidate"]))
        diverse, bins = [anchor], set()
        for state in next_beam:
            if state is anchor:
                continue
            candidate = state["candidate"]
            key = (
                round(candidate["body_y_m"] / 0.025),
                round(candidate["roll_deg"] / 3.0),
                round(candidate["yaw_deg"] / 3.0),
            )
            if key in bins and len(diverse) < beam_width // 2:
                continue
            bins.add(key)
            diverse.append(state)
            if len(diverse) == beam_width:
                break
        beam = diverse
    best = max(beam, key=lambda item: len(item["path"])) if beam else None
    path = None if best is None else best["path"]
    report = {
        "start_s": start_s,
        "requested_end_s": end_s,
        "sample_s": sample_s,
        "beam_width": beam_width,
        "branches": branches,
        "seed": seed,
        "evaluated": evaluated,
        "last_valid_s": None if not path else path[-1]["time_s"],
        "crossed_first_collision_boundary": bool(path and path[-1]["time_s"] >= 106.3),
        "first_failure": first_failure,
        "minimum_clearance_m": None
        if not path
        else min(item["lever_clearance_m"] for item in path),
        "maximum_ratio": None if not path else max(item["ratio"] for item in path),
    }
    if path:
        frontier = [state["candidate"] for state in beam]
        route = {
            "time_s": np.array([item["time_s"] for item in path]),
            "qpos": np.array([item["qpos"] for item in path]),
            "body_offset": np.array(
                [[item["candidate"][name] for name in BODY_KEYS] for item in path]
            ),
            "foot_x_delta_m": np.array([item["candidate"]["foot_x_delta_m"] for item in path]),
            "foot_y_delta_m": np.array([item["candidate"]["foot_y_delta_m"] for item in path]),
            "ratio": np.array([item["ratio"] for item in path]),
            "lever_clearance_m": np.array([item["lever_clearance_m"] for item in path]),
            "frontier_body_offset": np.array(
                [[candidate[name] for name in BODY_KEYS] for candidate in frontier]
            ),
            "frontier_foot_x_delta_m": np.array(
                [candidate["foot_x_delta_m"] for candidate in frontier]
            ),
            "frontier_foot_y_delta_m": np.array(
                [candidate["foot_y_delta_m"] for candidate in frontier]
            ),
        }
    else:
        route = None
    return report, route


def dense_validate_route(model, data, walk, route, rise_m: float, tracking_error_m: float) -> dict:
    """Re-evaluate an interpolated continuation route on every 20 ms cached walk sample."""
    start, end = route["time_s"][[0, -1]]
    indices = np.flatnonzero((walk["time"] >= start) & (walk["time"] <= end))
    minimum_clearance, maximum_ratio = np.inf, 0.0
    previous_x = previous_y = None
    for index in indices:
        timestamp = walk["time"][index]
        body = [np.interp(timestamp, route["time_s"], route["body_offset"][:, k]) for k in range(6)]
        foot_x = np.array(
            [np.interp(timestamp, route["time_s"], route["foot_x_delta_m"][:, k]) for k in range(4)]
        )
        foot_y = np.array(
            [np.interp(timestamp, route["time_s"], route["foot_y_delta_m"][:, k]) for k in range(4)]
        )
        if previous_x is not None:
            fixed = walk["stance"][index]
            # JIT target decoration: only the scheduled swing foot may follow the interpolated
            # offset; supported feet retain their last world target exactly.
            foot_x[fixed] = previous_x[fixed]
            foot_y[fixed] = previous_y[fixed]
        candidate = {
            **dict(zip(BODY_KEYS, body, strict=True)),
            "foot_x_delta_m": foot_x.tolist(),
            "foot_y_delta_m": foot_y.tolist(),
            "lift_leg": None,
        }
        row = {name: walk[name][index] for name in ("qpos", "feet", "stance")}
        result = evaluate_candidate(model, data, row, candidate, rise_m)
        if not result["ok"]:
            return {
                "passed": False,
                "time_s": float(timestamp),
                "last_valid_s": float(walk["time"][indices[np.where(indices == index)[0][0] - 1]])
                if index != indices[0]
                else None,
                "minimum_clearance_m_before_failure": float(minimum_clearance),
                "tracking_margin_m_before_failure": float(minimum_clearance - tracking_error_m),
                "maximum_ratio_before_failure": float(maximum_ratio),
                **result,
            }
        minimum_clearance = min(minimum_clearance, result["lever_clearance_m"])
        maximum_ratio = max(maximum_ratio, result["ratio"])
        previous_x, previous_y = foot_x, foot_y
    return {
        "passed": True,
        "samples": len(indices),
        "minimum_clearance_m": float(minimum_clearance),
        "tracking_error_m": tracking_error_m,
        "tracking_margin_m": float(minimum_clearance - tracking_error_m),
        "maximum_ratio": float(maximum_ratio),
    }


def build_control_track(model, data, walk, route, rise_m: float, end_s: float) -> dict:
    """Decorate the accepted walk interval in transition.execute's normal command format."""
    start = float(route["time_s"][0])
    indices = np.flatnonzero((walk["time"] >= start) & (walk["time"] <= end_s))
    addresses = np.array([model.joint(name).qposadr[0] for name in joint_names()])
    rows = {
        name: []
        for name in ("time", "qpos", "target", "torque", "tip", "stance", "wall", "cap", "feet")
    }
    previous_x = previous_y = None
    for index in indices:
        timestamp = walk["time"][index]
        body = [np.interp(timestamp, route["time_s"], route["body_offset"][:, k]) for k in range(6)]
        foot_x = np.array(
            [np.interp(timestamp, route["time_s"], route["foot_x_delta_m"][:, k]) for k in range(4)]
        )
        foot_y = np.array(
            [np.interp(timestamp, route["time_s"], route["foot_y_delta_m"][:, k]) for k in range(4)]
        )
        if previous_x is not None:
            fixed = walk["stance"][index]
            foot_x[fixed], foot_y[fixed] = previous_x[fixed], previous_y[fixed]
        candidate = {
            **dict(zip(BODY_KEYS, body, strict=True)),
            "foot_x_delta_m": foot_x.tolist(),
            "foot_y_delta_m": foot_y.tolist(),
            "lift_leg": None,
        }
        row = {name: walk[name][index] for name in ("qpos", "feet", "stance")}
        result = evaluate_candidate(model, data, row, candidate, rise_m)
        if not result["ok"]:
            raise ValueError(f"accepted track invalid at {timestamp:.2f}s: {result}")
        if unexpected_collision(model, data) is not None:
            raise ValueError(f"active unexpected contact at {timestamp:.2f}s")
        feet = adjusted_feet(row["feet"], x_delta_m=foot_x, y_delta_m=foot_y)
        stance = horizontal_stance(feet, row["stance"])
        caps = {
            leg: float(value)
            for leg, value in zip(LEGS, walk["cap"][index], strict=True)
            if np.isfinite(value) and leg in stance
        }
        solution = solve_static(
            model,
            data,
            transition.contacts_for(stance),
            torque_limits("rb1"),
            max_normal=caps,
            smooth=True,
        )
        if solution is None or solution.ratio > 1.0:
            raise ValueError(f"static command invalid at {timestamp:.2f}s")
        rows["time"].append(timestamp)
        rows["qpos"].append(data.qpos[:27].copy())
        rows["target"].append(data.qpos[addresses].copy())
        rows["torque"].append(solution.torque.copy())
        rows["tip"].append(data.site("arm_tip").xpos.copy())
        rows["stance"].append(walk["stance"][index].copy())
        rows["wall"].append(walk["wall"][index].copy())
        rows["cap"].append(walk["cap"][index].copy())
        rows["feet"].append(np.array([data.geom_xpos[model.geom(leg).id].copy() for leg in LEGS]))
        previous_x, previous_y = foot_x, foot_y
    out = {name: np.asarray(values) for name, values in rows.items()}
    out["collision_validation"] = np.array(
        "active MuJoCo contacts; model collision filters applied"
    )
    out["distance_label"] = np.array(
        "mj_geomDistance is geometric only and is not an active-contact or collision-filter gate"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("walk", type=Path)
    parser.add_argument("study", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--candidate", type=int, default=4)
    parser.add_argument("--time", type=float, default=156.68, help="coarse bottleneck sample")
    parser.add_argument("--multistart", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument(
        "--connect", action="store_true", help="validate best four-foot neighbour path"
    )
    parser.add_argument("--continuation", action="store_true")
    parser.add_argument("--start", type=float, default=90.0)
    parser.add_argument("--end", type=float, default=130.0)
    parser.add_argument("--sample", type=float, default=0.2)
    parser.add_argument("--beam", type=int, default=12)
    parser.add_argument("--branches", type=int, default=12)
    parser.add_argument("--route-out", type=Path)
    parser.add_argument("--seed-route", type=Path)
    parser.add_argument("--dense-route", type=Path)
    parser.add_argument("--tracking-error", type=float, default=0.035)
    parser.add_argument("--control-track", type=Path)
    parser.add_argument("--control-end", type=float, default=107.52)
    args = parser.parse_args()
    study = json.loads(args.study.read_text(encoding="utf-8"))
    saved = study["candidates"][args.candidate]
    model, data = stair_valve.build_grasp_scene(
        study["slope_deg"],
        study["rise_m"],
        saved["centre_m"],
        saved["start_deg"],
        saved["sense"],
        saved["handle_length_m"],
        saved["resistance_nm"],
    )
    with np.load(args.walk, allow_pickle=False) as walk:
        if args.control_track is not None:
            with np.load(args.control_track, allow_pickle=False) as route:
                track = build_control_track(
                    model, data, walk, route, study["rise_m"], args.control_end
                )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.out, **track)
            print(json.dumps({"samples": len(track["time"]), "end_s": float(track["time"][-1])}))
            return
        if args.dense_route is not None:
            with np.load(args.dense_route, allow_pickle=False) as route:
                report = dense_validate_route(
                    model, data, walk, route, study["rise_m"], args.tracking_error
                )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report))
            return
        if args.continuation:
            initial = None
            if args.seed_route is not None:
                with np.load(args.seed_route, allow_pickle=False) as seed_route:
                    body_key = (
                        "frontier_body_offset"
                        if "frontier_body_offset" in seed_route
                        else "body_offset"
                    )
                    foot_x_key = (
                        "frontier_foot_x_delta_m"
                        if "frontier_foot_x_delta_m" in seed_route
                        else "foot_x_delta_m"
                    )
                    foot_y_key = (
                        "frontier_foot_y_delta_m"
                        if "frontier_foot_y_delta_m" in seed_route
                        else "foot_y_delta_m"
                    )
                    bodies = np.atleast_2d(seed_route[body_key])
                    foot_x = np.atleast_2d(seed_route[foot_x_key])
                    foot_y = np.atleast_2d(seed_route[foot_y_key])
                    if body_key == "body_offset":
                        bodies, foot_x, foot_y = bodies[-1:], foot_x[-1:], foot_y[-1:]
                    initial = [
                        {
                            **dict(zip(BODY_KEYS, body, strict=True)),
                            "foot_x_delta_m": x.tolist(),
                            "foot_y_delta_m": y.tolist(),
                            "lift_leg": None,
                        }
                        for body, x, y in zip(bodies, foot_x, foot_y, strict=True)
                    ]
            report, route = continuation_search(
                model,
                data,
                walk,
                study["rise_m"],
                start_s=args.start,
                end_s=args.end,
                sample_s=args.sample,
                beam_width=args.beam,
                branches=args.branches,
                seed=args.seed,
                initial_candidates=initial,
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            if route is not None and args.route_out is not None:
                args.route_out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(args.route_out, **route)
            print(json.dumps(report))
            return
        index = int(np.argmin(np.abs(walk["time"] - args.time)))
        row = {name: walk[name][index] for name in ("qpos", "feet", "stance")}
        attempts, passing = [], []
        grid = (
            candidate_grid()
            if args.multistart == 0
            else multistart_grid(args.multistart, args.seed)
        )
        for candidate in grid:
            result = evaluate_candidate(model, data, row, candidate, study["rise_m"])
            attempts.append({**candidate, **result})
            if result["ok"]:
                passing.append(attempts[-1])
        four_foot = sorted(
            (item for item in passing if item.get("lift_leg") is None),
            key=lambda item: item["ratio"],
        )
        connection = (
            validate_neighbour_path(model, data, walk, index, four_foot[0], study["rise_m"])
            if args.connect and four_foot
            else None
        )
    reasons = ("foot_width", "ik", "collision", "static", "joint_saturation")
    counts = {reason: sum(a.get("reason") == reason for a in attempts) for reason in reasons}
    report = {
        "scope": "fixed candidate4 shut-valve local bypass probe",
        "sample_time_s": float(args.time),
        "grid_size": len(attempts),
        "passing_count": len(passing),
        "failure_counts": counts,
        "static_failures": [a for a in attempts if a.get("reason") == "static"][:50],
        "passing": sorted(passing, key=lambda item: item["ratio"])[:20],
        "neighbour_path": connection,
        "best_by_overlap": sorted(
            (a for a in attempts if "overlap_m" in a), key=lambda item: item["overlap_m"]
        )[:20],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("grid_size", "failure_counts", "passing")}))


if __name__ == "__main__":
    main()
