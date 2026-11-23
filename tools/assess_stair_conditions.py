"""Aggregate frozen stair-condition rows from existing run evidence only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import zlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def _slug(value: object) -> str:
    text = format(value, "g") if isinstance(value, float) else str(value)
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def protocol_rows(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand single-factor, sensitivity, and optional boundary cases in source order."""
    rows: list[dict[str, Any]] = []
    for case_id, changes in protocol.get("single_factor_cases", {}).items():
        rows.append({"id": case_id, "group": "single_factor", "changes": dict(changes)})

    for parameter, values in protocol.get("sensitivity", {}).items():
        for value in values:
            rows.append(
                {
                    "id": f"sensitivity_{_slug(parameter)}_{_slug(value)}",
                    "group": "sensitivity",
                    "changes": {parameter: value},
                }
            )

    boundary = protocol.get("boundary_cases", {})
    items = (
        boundary.items()
        if isinstance(boundary, Mapping)
        else ((item["id"], item.get("changes", {})) for item in boundary)
    )
    for case_id, changes in items:
        rows.append({"id": case_id, "group": "boundary", "changes": dict(changes)})
    return rows


def discover_artifacts(roots: Iterable[Path]) -> list[Path]:
    """Find only documented run outputs: result.json and ``*.failure.json``."""
    found: set[Path] = set()
    for root in roots:
        if root.is_file() and (root.name == "result.json" or root.name.endswith(".failure.json")):
            found.add(root.resolve())
        elif root.is_dir():
            found.update(path.resolve() for path in root.rglob("result.json"))
            found.update(path.resolve() for path in root.rglob("*.failure.json"))
    return sorted(found)


def _explicit_id(path: Path, payload: Mapping[str, Any]) -> str | None:
    for key in ("condition_id", "case_id", "id"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    if path.name == "result.json":
        return path.parent.name
    stem = path.name.removesuffix(".failure.json")
    aliases = {"condition-60": "slope_60", "condition-165": "rise_165", "condition-200": "rise_200"}
    return aliases.get(stem, stem)


def _matches_changes(payload: Mapping[str, Any], changes: Mapping[str, Any]) -> bool:
    if not changes:
        return False
    aliases = {
        "floor_mu": ("floor_mu", "floor_friction"),
        "resistance_nm": ("resistance_nm", "valve_resistance_nm"),
    }
    for key, expected in changes.items():
        keys = aliases.get(key, (key,))
        actual = next((payload[name] for name in keys if name in payload), None)
        if actual is None:
            return False
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
                return False
        elif actual != expected:
            return False
    return True


def _control_profile_verified(payload: Mapping[str, Any]) -> bool | None:
    """Validate the measured replant structure when a result declares a control profile."""
    profile = payload.get("control_profile")
    if profile is None:
        return None
    if not isinstance(profile, Mapping):
        return False
    physical = profile.get("physical_replant")
    if not isinstance(physical, Mapping):
        return False
    original = physical.get("original_scheduled_landings")
    auxiliary = physical.get("auxiliary_replant_landings")
    total = physical.get("total_scheduled_landings")
    lift = physical.get("maximum_foot_lift_m")
    minimum_load = physical.get("minimum_total_replant_leg_load_n")
    relanding = physical.get("relanding")
    if (
        physical.get("verified") is not True
        or (original, auxiliary, total) != (156, 1, 157)
        or total != original + auxiliary
        or physical.get("released_from_support") is not True
        or not isinstance(lift, (int, float))
        or not math.isfinite(lift)
        or lift <= 0
        or not isinstance(minimum_load, (int, float))
        or not math.isfinite(minimum_load)
        or minimum_load >= 1.0
        or not isinstance(relanding, Mapping)
        or relanding.get("moved_leg") != profile.get("replant_leg")
    ):
        return False
    support = relanding.get("support")
    error = relanding.get("terminal_foot_xy_error_m")
    return bool(
        isinstance(support, Mapping)
        and support.get("all_expected_feet_ge_1n") is True
        and isinstance(error, (int, float))
        and math.isfinite(error)
        and error <= 0.035
    )


def _frozen_scene_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Map the protocol vocabulary to fields recorded in a mission scene."""
    fields = {
        "name": "name",
        "replant_after_landing_index": "trigger_landing_index",
        "replant_trigger_s": "trigger_time_s",
        "replant_duration_s": "duration_s",
        "replant_leg": "replant_leg",
        "slip_spring_gain_n_per_m": "slip_spring_gain_n_per_m",
        "slip_spring_active_s": "slip_spring_active_s",
        "stance_spring_gain_n_per_m": "stance_spring_gain_n_per_m",
        "terminal_overtravel_deg": "terminal_overtravel_deg",
    }
    return {scene_key: profile[protocol_key] for protocol_key, scene_key in fields.items()}


def _outcome(payload: Mapping[str, Any]) -> str:
    extent = payload.get("extent")
    stated = payload.get("outcome") or payload.get("status")
    aliases = {
        "planning_failed": "planning_failed",
        "plan_failed": "planning_failed",
        "physical_failed": "physical_failed",
        "physics_failed": "physical_failed",
        "failed": "physical_failed",
        "success": "success",
        "passed": "success",
        "verified": "success",
    }
    judgments = [
        payload[k]["verified"]
        for k in ("validation", "mission", "walking", "valve")
        if isinstance(payload.get(k), Mapping) and "verified" in payload[k]
    ]
    if False in judgments:
        if extent == "full" or isinstance(_first_failure_summary(payload), Mapping):
            return "physical_failed"
    if _control_profile_verified(payload) is False:
        return "physical_failed"
    if isinstance(stated, str) and aliases.get(stated.lower()) in (
        "planning_failed",
        "physical_failed",
    ):
        return aliases[stated.lower()]
    if extent == "full" and (payload.get("mission") or {}).get("verified") is True:
        return "success"
    if extent == "through-task" and all(
        (payload.get(component) or {}).get("verified") is True for component in ("walking", "valve")
    ):
        return "success"

    valve = payload.get("valve")
    if isinstance(valve, Mapping) and valve.get("verified") is False:
        if isinstance(valve.get("first_failure"), Mapping):
            return "physical_failed"
        failures = valve.get("failures")
        if isinstance(failures, list) and failures:
            return "physical_failed"
    if extent in ("approach", "through-task"):
        return "pending"

    if payload.get("verified") is False:
        return "physical_failed"
    return "pending"


def observed_settings(protocol, row, path, payload):
    """Bind the row to measured inputs. A named directory is only a lookup hint."""
    g, a, hand = (
        protocol.get("geometry", {}),
        protocol.get("actuation", {}),
        protocol.get("gripper", {}),
    )
    base = {
        "slope_deg": g.get("slope_deg"),
        "rise_m": g.get("rise_m"),
        "floor_mu": hand.get("floor_mu"),
        "pad_mu": hand.get("pad_mu"),
        "finger_force_n": hand.get("finger_force_limit_n"),
        "slide_loss_n": hand.get("finger_slide_frictionloss_n"),
        "handle_length_m": a.get("lever_m"),
        "resistance_nm": a.get("nominal_hinge_resistance_nm"),
        "timestep_s": protocol.get("physics", {}).get("timestep_s"),
    }
    target = {k: v for k, v in base.items() if v is not None}
    if target:
        target.update(impratio=100.0, arm_limits="rb1")
    aliases = {
        "lever_m": "handle_length_m",
        "finger_force_limit_n": "finger_force_n",
        "finger_slide_frictionloss_n": "slide_loss_n",
    }
    for key, value in row["changes"].items():
        target[aliases.get(key, key)] = "lims-ex" if key == "arm_limits" else value
    if path.name.endswith(".failure.json"):
        if set(row["changes"]) - {"slope_deg", "rise_m"}:
            raise ValueError(f"A geometry planning failure cannot certify {row['id']}")
        target = {k: v for k, v in target.items() if k in ("slope_deg", "rise_m")}
        settings = payload
        hashes = {}
        planner_hashes = payload.get("planner_source_sha256")
    else:
        scene_path = path.with_name("scene.json")
        scene_bytes = scene_path.read_bytes()
        if hashlib.sha256(scene_bytes).hexdigest() != payload.get("scene_sha256"):
            raise ValueError(f"Scene hash mismatch for {row['id']}")
        scene = json.loads(scene_bytes)
        with path.with_name("mission.npz").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != payload.get("raw_sha256"):
                raise ValueError(f"Raw trace hash mismatch for {row['id']}")
        planner_hashes = scene.get("planner_source_sha256")
        if scene["settings"] != payload.get("settings") or scene["extent"] != payload.get("extent"):
            raise ValueError(f"Result and scene disagree for {row['id']}")
        settings = {
            **scene["settings"],
            **{k: scene.get(k) for k in ("timestep_s", "impratio", "arm_limits")},
        }
        hashes = scene["source_sha256"]
        if not hashes:
            raise ValueError(f"Missing controller source hashes for {row['id']}")
        frozen_hashes = protocol.get("source_sha256")
        if frozen_hashes is not None and hashes != frozen_hashes:
            raise ValueError(
                f"Controller source hashes differ from the frozen protocol for {row['id']}"
            )
        frozen_profile = protocol.get("control_profile")
        if frozen_profile is not None:
            if not isinstance(frozen_profile, Mapping):
                raise ValueError("Frozen control profile must be an object")
            expected_profile = _frozen_scene_profile(frozen_profile)
            scene_profile = scene.get("control_profile")
            result_profile = payload.get("control_profile")
            if not isinstance(scene_profile, Mapping) or any(
                scene_profile.get(key) != value for key, value in expected_profile.items()
            ):
                raise ValueError(
                    f"Scene control profile differs from the frozen protocol for {row['id']}"
                )
            if not isinstance(result_profile, Mapping) or any(
                result_profile.get(key) != scene_profile.get(key) for key in scene_profile
            ):
                raise ValueError(f"Result and scene control profiles disagree for {row['id']}")
    for k, v in target.items():
        if settings.get(k) != v:
            raise ValueError(f"{row['id']}: expected {k}={v}, observed {settings.get(k)}")
    if not planner_hashes:
        raise ValueError(f"Missing planner source hashes for {row['id']}")
    return hashes, planner_hashes


def _reported_failures(payload: Mapping[str, Any]):
    first = payload.get("first_failure")
    if isinstance(first, Mapping):
        yield first
    failures = payload.get("failures")
    if isinstance(failures, list):
        yield from (failure for failure in failures if isinstance(failure, Mapping))
    for key in ("walking", "valve", "mission", "validation"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            yield from _reported_failures(nested)


def _first_failure_summary(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    def time_order(failure):
        for time_s in (failure.get("time_s"), failure.get("planned_time_s")):
            if type(time_s) in (int, float) and math.isfinite(time_s):
                return time_s
        return math.inf

    failure = min(_reported_failures(payload), key=time_order, default=None)
    if failure is None:
        return None

    stage = failure.get("stage") or failure.get("kind")
    if stage is None and "segment" in failure:
        stage = f"segment_{failure['segment']}"
    time_s = time_order(failure)
    metric = failure.get("metric")
    measured = failure.get("measured")
    limit = failure.get("limit")

    if metric is None:
        overlaps = failure.get("unexpected_overlaps")
        if isinstance(overlaps, list) and overlaps:
            metric = "unexpected_overlap_m"
            measured = max(float(item["penetration_m"]) for item in overlaps)
            limit = failure.get("collision_tolerance_m")
        elif isinstance(failure.get("foot_ik_error_m"), Mapping):
            metric = "max_foot_ik_error_m"
            measured = max(float(value) for value in failure["foot_ik_error_m"].values())
            limit = failure.get("foot_ik_limit_m")

    return {
        "stage": stage,
        "time_s": time_s if math.isfinite(time_s) else None,
        "metric": metric,
        "measured": measured,
        "limit": limit,
    }


def aggregate(
    protocol: Mapping[str, Any],
    artifacts: Iterable[Path],
    *,
    relative_to: Path,
    assigned_ids: Mapping[Path, str] | None = None,
) -> dict[str, Any]:
    """Associate evidence with protocol rows without manufacturing missing outcomes."""
    rows = protocol_rows(protocol)
    loaded: list[tuple[Path, Mapping[str, Any]]] = []
    for path in artifacts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"result must be a JSON object: {path}")
        loaded.append((path, payload))

    assigned = {path.resolve(): case_id for path, case_id in (assigned_ids or {}).items()}
    used: set[Path] = set()
    controller = None
    planner = None
    row_ids = {row["id"] for row in rows}
    for row in rows:
        candidates = [
            (path, payload)
            for path, payload in loaded
            if assigned.get(path.resolve(), _explicit_id(path, payload)) == row["id"]
            or (
                path.resolve() not in assigned
                and _explicit_id(path, payload) not in row_ids
                and _matches_changes(payload, row["changes"])
            )
        ]
        if len(candidates) > 1:
            names = ", ".join(str(path) for path, _ in candidates)
            raise ValueError(f"multiple results match {row['id']}: {names}")
        if not candidates:
            row.update(outcome="pending", evidence_path=None, first_failure=None)
            continue
        path, payload = candidates[0]
        hashes, planner_hashes = observed_settings(protocol, row, path, payload)
        if planner is not None and planner != planner_hashes:
            raise ValueError("Rows use different walking planners")
        planner = planner_hashes
        if hashes:
            if controller is not None and hashes != controller:
                raise ValueError(
                    "Rows use different source revisions; start a separate condition table"
                )
            controller = hashes
        used.add(path)
        outcome = _outcome(payload)
        row.update(
            outcome=outcome,
            evidence_path=path.relative_to(relative_to.resolve()).as_posix(),
            first_failure=_first_failure_summary(payload) if outcome != "success" else None,
        )

    counts = {
        name: sum(row["outcome"] == name for row in rows)
        for name in ("success", "planning_failed", "physical_failed", "pending")
    }
    return {
        "protocol_plan_commit": protocol.get("plan_commit"),
        "counts": counts,
        "conditions": rows,
        "unmatched_evidence": [
            path.relative_to(relative_to.resolve()).as_posix()
            for path, _ in loaded
            if path not in used
        ],
    }


def run_assignment(value: str) -> tuple[str, Path]:
    """Parse ``CASE_ID=RUN_DIR`` and require one result artifact in the run directory."""
    case_id, separator, raw_path = value.partition("=")
    if not separator or not case_id or not raw_path:
        raise argparse.ArgumentTypeError("run must be CASE_ID=RUN_DIR")
    root = Path(raw_path)
    artifacts = discover_artifacts([root])
    if len(artifacts) != 1:
        raise argparse.ArgumentTypeError(
            f"{root} must contain exactly one result.json or *.failure.json; found {len(artifacts)}"
        )
    return case_id, artifacts[0]


def _write_png_rgb(path: Path, pixels: Any) -> None:
    """Write an HxWx3 uint8 array without adding an image dependency."""
    height, width, channels = pixels.shape
    if channels != 3:
        raise ValueError("snapshot must be RGB")
    raw = b"".join(b"\0" + pixels[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )


def render_planning_snapshot(failure_path: Path, output: Path) -> None:
    """Render the saved planning pose in its RBQ stair geometry, without simulation."""
    import numpy as np

    from go2_wall import stairs

    payload = json.loads(failure_path.read_text(encoding="utf-8"))
    first = payload.get("first_failure", {})
    qpos = np.asarray(first.get("planning_qpos", []), dtype=float)
    if qpos.ndim != 1 or not qpos.size:
        raise ValueError(f"planning_qpos missing from {failure_path}")
    stairs.ARM = "rb1"
    stairs.use_stow()
    model, data = stairs.build_stairs(float(payload["slope_deg"]), float(payload["rise_m"]))
    if qpos.size != model.nq:
        raise ValueError(f"planning_qpos has {qpos.size} values; RBQ scene requires {model.nq}")
    data.qpos[:] = qpos
    _render_pose(model, data, output)


def render_physical_snapshot(result_path, failure_time, output):
    import numpy as np
    from render_stair_mission import build_model

    scene = json.loads(result_path.with_name("scene.json").read_bytes())
    model, data = build_model(scene)
    with np.load(result_path.with_name("mission.npz"), allow_pickle=False) as arrays:
        times = arrays["time_s"]
        index = max(0, min(len(times) - 1, int(np.searchsorted(times, failure_time)) - 1))
        data.qpos[:] = arrays["qpos"][index]
        data.qvel[:] = arrays["qvel"][index]
        recorded_time = float(times[index])
    _render_pose(model, data, output)
    return {"raw_index": index, "time_s": recorded_time}


def _render_pose(model, data, output):
    import mujoco
    import numpy as np

    mujoco.mj_forward(model, data)

    width, height = 1280, 720
    for name in ("stringerL", "stringerR"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            model.geom_rgba[geom_id, 3] = 0.0
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    model.vis.headlight.ambient[:] = [0.55, 0.55, 0.55]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = data.qpos[:3] + np.array([0.12, 0.0, 0.0])
    camera.distance, camera.azimuth, camera.elevation = 2.1, 125, -18
    with mujoco.Renderer(model, height, width) as renderer:
        renderer.update_scene(data, camera)
        _write_png_rgb(output, renderer.render())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("run_roots", type=Path, nargs="*")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="CASE_ID=RUN_DIR",
        help="bind a protocol category to one run directory or failure file",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-to", type=Path, default=Path.cwd())
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        help="render failed planned poses or the last physical sample before failure",
    )
    args = parser.parse_args()
    assignments = [run_assignment(value) for value in args.run]
    if not args.run_roots and not assignments:
        parser.error("provide a run_root or --run CASE_ID=RUN_DIR")
    artifacts = discover_artifacts(args.run_roots)
    artifacts.extend(path for _, path in assignments)
    assigned_ids = {path: case_id for case_id, path in assignments}
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    report = aggregate(
        protocol,
        sorted(set(artifacts)),
        relative_to=args.relative_to,
        assigned_ids=assigned_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.snapshot_dir:
        by_path = {
            path.resolve(): payload
            for path, payload in (
                (path, json.loads(path.read_text(encoding="utf-8"))) for path in artifacts
            )
        }
        for row in report["conditions"]:
            if (
                row["outcome"] not in ("planning_failed", "physical_failed")
                or not row["evidence_path"]
            ):
                continue
            source = (args.relative_to / row["evidence_path"]).resolve()
            png = args.snapshot_dir / f"{row['id']}.png"
            if row["outcome"] == "planning_failed" and "planning_qpos" in by_path[source].get(
                "first_failure", {}
            ):
                render_planning_snapshot(source, png)
                row["snapshot"] = {"type": "planned pose", "time_s": row["first_failure"]["time_s"]}
            elif (
                row["outcome"] == "physical_failed"
                and (row.get("first_failure") or {}).get("time_s") is not None
            ):
                row["snapshot"] = render_physical_snapshot(
                    source, row["first_failure"]["time_s"], png
                )
            if "snapshot" in row:
                row["snapshot"]["path"] = (
                    png.resolve().relative_to(args.relative_to.resolve()).as_posix()
                )
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
