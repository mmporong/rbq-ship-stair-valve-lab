"""CPU probes for thin-tread contacts and the actuator force-limit derivative."""

import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

# Foot radius from the RBQ model, plate thickness from train/rbq_rb1_stairs.py.
FOOT_RADIUS_M = 0.03
TREAD_THICKNESS_M = 0.03
TREAD_HALF_DEPTH_M = 0.125
TREAD_HALF_WIDTH_M = 0.4

CONTACT_TIMESTEPS_S = (0.005, 0.01, 0.02)
IMPACT_SPEEDS_M_S = (1.0, 3.0, 6.0, 12.0, 24.0)
CLEARANCE_PHASES = (0.1, 0.5, 0.9)
CONTACT_WINDOW_S = 0.2

ACTUATOR_TIMESTEP_S = 0.02
ACTUATOR_KP = 20.0
ACTUATOR_KV = 4.0
FORCE_LIMIT_N = 5.0


def foot_model(timestep: float, integrator: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(f"""
        <mujoco>
          <option timestep="{timestep}" gravity="0 0 0" integrator="{integrator}"/>
          <worldbody>
            <geom name="tread" type="box" pos="0 0 {-TREAD_THICKNESS_M / 2}"
                  size="{TREAD_HALF_DEPTH_M} {TREAD_HALF_WIDTH_M} {TREAD_THICKNESS_M / 2}"/>
            <body name="foot">
              <freejoint/>
              <geom name="sphere" type="sphere" size="{FOOT_RADIUS_M}" mass="1"
                    priority="1" condim="3" friction="0.6 0.005 0.0001"
                    solimp="0.9 0.95 0.023"/>
            </body>
          </worldbody>
        </mujoco>
    """)


def drop_foot(
    model: mujoco.MjModel, speed: float, phase: float, speculative: bool, beside_tread: bool
) -> dict:
    flag = int(mujoco.mjtEnableBit.mjENBL_SPECULATIVE)
    model.opt.enableflags = (
        model.opt.enableflags | flag if speculative else model.opt.enableflags & ~flag
    )
    dt = model.opt.timestep
    data = mujoco.MjData(model)
    data.qpos[0] = TREAD_HALF_DEPTH_M + 2 * FOOT_RADIUS_M if beside_tread else 0.0
    data.qpos[2] = FOOT_RADIUS_M + phase * speed * dt
    data.qvel[2] = -speed

    contact_steps = 0
    first_contact_dist = None
    greatest_top_overlap = 0.0
    below_step = None
    for step in range(1, round(CONTACT_WINDOW_S / dt) + 1):
        mujoco.mj_step(model, data)
        active_contacts = [contact for contact in data.contact if contact.efc_address >= 0]
        if active_contacts:
            contact_steps += 1
            if first_contact_dist is None:
                first_contact_dist = float(active_contacts[0].dist)
        if data.qpos[2] + FOOT_RADIUS_M < -TREAD_THICKNESS_M:
            below_step = step
            break
        if not beside_tread:
            greatest_top_overlap = max(greatest_top_overlap, FOOT_RADIUS_M - float(data.qpos[2]))

    return {
        "integrator": mujoco.mjtIntegrator(model.opt.integrator).name,
        "timestep_s": dt,
        "initial_speed_m_s": speed,
        "clearance_as_fraction_of_step_travel": phase,
        "speculative": speculative,
        "beside_tread": beside_tread,
        "fully_below_tread_step": below_step,
        "tunneled_without_active_contact": below_step is not None
        and not beside_tread
        and contact_steps == 0,
        "active_contact_steps": contact_steps,
        "first_contact_dist_m": first_contact_dist,
        "max_top_overlap_before_full_crossing_m": greatest_top_overlap,
        "final_center_z_m": float(data.qpos[2]),
        "final_vz_m_s": float(data.qvel[2]),
    }


def contact_study() -> dict:
    rows = []
    for integrator, dt in itertools.product(("Euler", "implicitfast"), CONTACT_TIMESTEPS_S):
        model = foot_model(dt, integrator)
        for speed, phase, enabled in itertools.product(
            IMPACT_SPEEDS_M_S, CLEARANCE_PHASES, (False, True)
        ):
            rows.append(drop_foot(model, speed, phase, enabled, beside_tread=False))
        for enabled in (False, True):
            rows.append(drop_foot(model, 6.0, 0.5, enabled, beside_tread=True))
    return {
        "mujoco": mujoco.__version__,
        "gravity_m_s2": [0, 0, 0],
        "foot_radius_m": FOOT_RADIUS_M,
        "tread_thickness_m": TREAD_THICKNESS_M,
        "tread_half_depth_m": TREAD_HALF_DEPTH_M,
        "tread_half_width_m": TREAD_HALF_WIDTH_M,
        "duration_limit_s": CONTACT_WINDOW_S,
        "rows": rows,
    }


def actuator_probe(limit: str, integrator: str, command: float) -> dict:
    limits = {
        "none": ("", ""),
        "control": ("", 'ctrlrange="-0.1 0.1"'),
        "actuator_force": ("", f'forcerange="{-FORCE_LIMIT_N} {FORCE_LIMIT_N}"'),
        "joint_force": (f'actuatorfrcrange="{-FORCE_LIMIT_N} {FORCE_LIMIT_N}"', ""),
    }
    joint_limit, actuator_limit = limits[limit]
    model = mujoco.MjModel.from_xml_string(f"""
        <mujoco>
          <option timestep="{ACTUATOR_TIMESTEP_S}" gravity="0 0 0" integrator="{integrator}"/>
          <worldbody>
            <body>
              <joint name="slider" type="slide" axis="1 0 0" {joint_limit}/>
              <geom type="sphere" size="0.05" mass="1"/>
            </body>
          </worldbody>
          <actuator>
            <position joint="slider" kp="{ACTUATOR_KP}" kv="{ACTUATOR_KV}" {actuator_limit}/>
          </actuator>
        </mujoco>
    """)
    data = mujoco.MjData(model)
    data.ctrl[0] = command
    mujoco.mj_step(model, data)
    return {
        "limit": limit,
        "integrator": integrator,
        "command_m": command,
        "actuator_force_n": float(data.actuator_force[0]),
        "joint_force_n": float(data.qfrc_actuator[0]),
        "implicit_velocity_derivative_ns_per_m": (
            float(data.qDeriv[0]) if integrator == "implicitfast" else None
        ),
        "next_velocity_m_s": float(data.qvel[0]),
    }


def actuator_inventory(model: mujoco.MjModel) -> dict:
    affine = int(mujoco.mjtBias.mjBIAS_AFFINE)
    actuators = []
    for i in range(model.nu):
        actuators.append(
            {
                "name": model.actuator(i).name,
                "ctrlrange": model.actuator_ctrlrange[i].tolist()
                if model.actuator_ctrllimited[i]
                else None,
                "forcerange": model.actuator_forcerange[i].tolist()
                if model.actuator_forcelimited[i]
                else None,
                "affine_velocity_bias": float(model.actuator_biasprm[i, 2])
                if model.actuator_biastype[i] == affine
                else None,
            }
        )
    return {
        "integrator": mujoco.mjtIntegrator(model.opt.integrator).name,
        "timestep_s": model.opt.timestep,
        "joint_force_limits": {
            model.joint(i).name: model.jnt_actfrcrange[i].tolist()
            for i in np.flatnonzero(model.jnt_actfrclimited)
        },
        "tendon_force_limits": {
            model.tendon(i).name: model.tendon_actfrcrange[i].tolist()
            for i in np.flatnonzero(model.tendon_actfrclimited)
        },
        "actuators": actuators,
    }


def actuator_study(probes_only: bool, rbq_scene: Path | None) -> dict:
    models = {}
    if not probes_only:
        from go2_wall.robot import build_robot

        for arm in ("d1", "z1", "rb1"):
            models[f"mission_{arm}"] = actuator_inventory(build_robot(arm=arm).compile())
    if rbq_scene is not None:
        model = mujoco.MjModel.from_xml_path(str(rbq_scene))
        models["rbq_stairs"] = actuator_inventory(model)
    return {
        "mujoco": mujoco.__version__,
        "probe": {
            "mass_kg": 1.0,
            "timestep_s": ACTUATOR_TIMESTEP_S,
            "position_gain_n_m": ACTUATOR_KP,
            "velocity_gain_ns_m": ACTUATOR_KV,
            "force_limit_n": FORCE_LIMIT_N,
            "rows": [
                actuator_probe(limit, integrator, command)
                for limit, integrator, command in itertools.product(
                    ("none", "control", "actuator_force", "joint_force"),
                    ("Euler", "implicitfast"),
                    (0.1, 1.0),
                )
            ],
        },
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    study = parser.add_subparsers(dest="study", required=True)
    contacts = study.add_parser("contacts")
    contacts.add_argument("out", type=Path)
    actuators = study.add_parser("actuators")
    actuators.add_argument("out", type=Path)
    actuators.add_argument("--probes-only", action="store_true")
    actuators.add_argument("--rbq-scene", type=Path)
    args = parser.parse_args()
    if args.study == "contacts":
        result = contact_study()
    else:
        result = actuator_study(args.probes_only, args.rbq_scene)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
