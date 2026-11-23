"""Link loads over the mission (M5): joint reactions, bracket stress, and phase peaks.

The mission log stores qpos, qvel, and motor commands at 50 Hz. Each sample is replayed with
mj_forward and mj_rnePostConstraint, which gives cfrc_int, the force and moment each body
receives from its parent. Moments are moved to the joint anchor or the bracket bolt line.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from . import lever as lv
from .robot import LEG_PARTS, LEGS, load_parameters
from .statics import joint_names, torque_limits

PHASES = ("climb", "lever", "descend")


def moment_about(data, body: int, point: np.ndarray, root: int) -> tuple[np.ndarray, np.ndarray]:
    """Force and moment on `body` from its parent, with the moment taken about `point`.

    cfrc_int is a spatial force about the subtree centre of mass of the tree root, in world
    axes: M_point = M_com + (com - point) x F.
    """
    torque, force = data.cfrc_int[body][:3], data.cfrc_int[body][3:]
    centre = data.subtree_com[root]
    return force.copy(), torque + np.cross(centre - point, force)


def bracket_section() -> dict:
    """Section moduli of the bracket plate.

    >>> s = bracket_section()
    >>> round(s["z_vertical_m3"] * 1e9, 1), round(s["z_lateral_m3"] * 1e9, 1)
    (480.0, 6400.0)
    """
    plate = load_parameters()["front_bracket"]["value"]
    width, thick = plate["plate_width_m"], plate["plate_thickness_m"]
    return {
        "z_vertical_m3": width * thick**2 / 6,
        "z_lateral_m3": thick * width**2 / 6,
        "torsion_m3": width * thick**2 / 3,
        "width_m": width,
        "thickness_m": thick,
    }


def bracket_stress(force_world, moment_world, rotation, cantilever: float) -> dict:
    """Upper-bound plate stress at the bolt line from the arm base load, in MPa.

    The arm load is moved back by `cantilever` along the trunk axis. Vertical and lateral
    bending use the plate section moduli, torsion uses the thin-strip value b t^2 / 3, and the
    three combine as von Mises.

    >>> eye = np.eye(3)
    >>> out = bracket_stress(np.array([0, 0, -10.0]), np.zeros(3), eye, 0.15)
    >>> round(out["bending_vertical_mpa"], 3), round(out["von_mises_mpa"], 3)
    (3.125, 3.125)
    """
    section = bracket_section()
    force = rotation.T @ force_world
    moment = rotation.T @ moment_world
    moment = moment + np.cross(np.array([cantilever, 0.0, 0.0]), force)
    vertical = abs(moment[1]) / section["z_vertical_m3"] / 1e6
    lateral = abs(moment[2]) / section["z_lateral_m3"] / 1e6
    shear = abs(moment[0]) / section["torsion_m3"] / 1e6
    return {
        "bending_vertical_mpa": float(vertical),
        "bending_lateral_mpa": float(lateral),
        "torsion_mpa": float(shear),
        "von_mises_mpa": float(np.sqrt((vertical + lateral) ** 2 + 3 * shear**2)),
        "moment_bolt_line_nm": moment.tolist(),
        "force_base_frame_n": force.tolist(),
    }


def phase_of(track: dict, times: np.ndarray) -> np.ndarray:
    window = np.isin(track["segment"], np.arange(100, 100 + len(lv.TIMES)))
    start, stop = track["time"][window][[0, -1]]
    return np.where(times < start, 0, np.where(times <= stop, 1, 2))


def analyse(npz: Path, lever: dict | None = None, wall_x: float = 0.75) -> dict:
    data_in = np.load(npz)
    model, data = lv.build_scene(wall_x, lever=lever)
    params = load_parameters()
    cantilever = params["bracket_bolt_line"]["value"]["cantilever_m"]
    yield_mpa = params["aluminum_6061_t6_yield_mpa"]["value"]
    root = model.body_rootid[model.body("base").id]
    mount = model.body("arm_mount").id
    base = model.body("base").id
    joints = {
        f"{leg}_{part}": model.joint(f"{leg}_{part}_joint") for leg in LEGS for part in LEG_PARTS
    }
    times = data_in["log_time"]
    phases = phase_of({k: data_in[k] for k in ("segment", "time")}, times)
    limits = torque_limits()
    names = joint_names()
    rows = {"stress": [], "joint_bend": [], "joint_force": [], "torque_ratio": []}
    for k in range(len(times)):
        data.qpos[:] = data_in["log_qpos"][k]
        data.qvel[:] = data_in["log_qvel"][k]
        data.ctrl[:] = data_in["log_ctrl"][k]
        mujoco.mj_forward(model, data)
        mujoco.mj_rnePostConstraint(model, data)
        force, moment = moment_about(data, mount, data.xpos[mount], root)
        rotation = data.xmat[base].reshape(3, 3)
        stress = bracket_stress(-force, -moment, rotation, cantilever)
        rows["stress"].append(stress["von_mises_mpa"])
        bend, push = [], []
        for name, joint in joints.items():
            body = model.jnt_bodyid[joint.id]
            anchor = data.xanchor[joint.id]
            axis = data.xaxis[joint.id]
            f, m = moment_about(data, body, anchor, root)
            bend.append(float(np.linalg.norm(m - (m @ axis) * axis)))
            push.append(float(np.linalg.norm(f)))
        rows["joint_bend"].append(bend)
        rows["joint_force"].append(push)
        rows["torque_ratio"].append(np.abs(data_in["log_torque"][k]) / limits)
    stress = np.array(rows["stress"])
    bend = np.array(rows["joint_bend"])
    push = np.array(rows["joint_force"])
    ratio = np.array(rows["torque_ratio"])
    torque = np.abs(data_in["log_torque"])
    out = {"phases": {}, "bracket": {}, "yield_mpa": yield_mpa, "cantilever_m": cantilever}
    for index, phase in enumerate(PHASES):
        mask = phases == index
        if not mask.any():
            continue
        out["phases"][phase] = {
            "seconds": [float(times[mask][0]), float(times[mask][-1])],
            "joint_torque_peak_nm": {
                n: round(float(torque[mask, j].max()), 3) for j, n in enumerate(names)
            },
            "joint_torque_peak_ratio": {
                n: round(float(ratio[mask, j].max()), 3) for j, n in enumerate(names)
            },
            "joint_bending_peak_nm": {
                n: round(float(bend[mask, j].max()), 3) for j, n in enumerate(joints)
            },
            "joint_reaction_peak_n": {
                n: round(float(push[mask, j].max()), 2) for j, n in enumerate(joints)
            },
            "bracket_von_mises_peak_mpa": round(float(stress[mask].max()), 3),
        }
    peak = int(np.argmax(stress))
    out["bracket"] = {
        "peak_von_mises_mpa": round(float(stress.max()), 3),
        "peak_time_s": float(times[peak]),
        "safety_factor_to_yield": round(yield_mpa / float(stress.max()), 1),
    }
    np.savez_compressed(
        npz.with_name("loads.npz"),
        time=times,
        phase=phases,
        bracket_mpa=stress,
        joint_bend=bend,
        joint_force=push,
        torque_ratio=ratio,
        joint_names=np.array(list(joints)),
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=Path, required=True, help="mission.npz from lever")
    args = parser.parse_args()
    result = analyse(args.mission)
    args.mission.with_name("loads.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result["bracket"], indent=1))
    for phase, item in result["phases"].items():
        top = sorted(item["joint_torque_peak_ratio"].items(), key=lambda kv: -kv[1])[:3]
        print(phase, "bracket", item["bracket_von_mises_peak_mpa"], "MPa", "top ratios", top)


if __name__ == "__main__":
    main()
