"""Quasi-static contact and joint-load solver for a floating-base robot."""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import linprog, minimize

from .robot import ARM_JOINTS, LEG_LIMITS, LEG_PARTS, LEGS, load_parameters

FLOOR_NORMAL = np.array([0.0, 0.0, 1.0])
WALL_NORMAL = np.array([-1.0, 0.0, 0.0])


def joint_names() -> list[str]:
    return [f"{leg}_{part}_joint" for leg in LEGS for part in LEG_PARTS] + ARM_JOINTS


def torque_limits(arm: str = "derived") -> np.ndarray:
    """Leg limits of the body (Go2, or RBQ-10 for "rb1") joined with one graded arm limit set."""
    parameters = load_parameters()
    sets = {
        "derived": parameters["arm_torque_lower_bound_nm"]["value"],
        "urdf": parameters["joint_torque_limit_public_nm"]["urdf"],
        "osu": parameters["joint_torque_limit_public_nm"]["osu"],
        "z1": parameters["z1_torque_limit_nm"]["value"],
        "rb1": parameters["rb1_torque_limit_nm"]["value"],
    }
    legs = LEG_LIMITS
    if arm == "rb1":
        legs = np.tile(parameters["rbq_leg_torque_limit_nm"]["value"], len(LEGS))
    return np.r_[legs, np.array(sets[arm])]


@dataclass
class Contact:
    leg: str
    normal: np.ndarray
    mu: float


@dataclass
class StaticSolution:
    ratio: float
    torque: np.ndarray
    forces: dict[str, np.ndarray]
    residual: float


def contact_point(model: mujoco.MjModel, data: mujoco.MjData, contact: Contact) -> np.ndarray:
    geom = model.geom(contact.leg).id
    return data.geom_xpos[geom] - model.geom_size[geom][0] * contact.normal


def _point_jacobian(model, data, point: np.ndarray, body: int) -> np.ndarray:
    jacobian = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacobian, None, point, body)
    return jacobian


def solve_static(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contacts: list[Contact],
    limits: np.ndarray,
    tip_force: np.ndarray | None = None,
    max_normal: dict[str, float] | None = None,
    com_force: np.ndarray | None = None,
    smooth: bool = False,
    extra_qfrc: np.ndarray | None = None,
    min_normal: dict[str, float] | None = None,
) -> StaticSolution | None:
    """Find contact forces that balance the robot and minimise the largest joint-limit ratio.

    Friction uses the inscribed pyramid |f1| + |f2| <= mu * fn, which is stricter than the cone.
    Static balance is qfrc_bias = S'tau + J_c'f + J_tip'f_tip. tip_force acts on the robot.
    max_normal caps the normal force of named contacts, used to hand load to or from a foot.
    com_force is an extra force on the whole-robot centre of mass, used as a disturbance test.
    extra_qfrc is any other external generalized force on the robot, such as a feedback wrench.
    min_normal sets a lower bound on the normal force of named contacts (a preload).

    The min-max LP has many optimal force sets when more than three feet touch, and it may jump
    between them as the pose changes. smooth=True instead returns the unique minimum of the sum
    of squared joint-limit ratios inside the same constraints, which changes continuously with
    the pose and is used for feedforward torques.
    """
    max_normal = max_normal or {}
    min_normal = min_normal or {}
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    bias = data.qfrc_bias.copy()
    if tip_force is not None:
        site = model.site("arm_tip").id
        tip = _point_jacobian(model, data, data.site_xpos[site], model.site_bodyid[site])
        bias = bias - tip.T @ tip_force
    if com_force is not None:
        com = np.zeros((3, model.nv))
        mujoco.mj_jacSubtreeCom(model, data, com, 1)
        bias = bias - com.T @ com_force
    if extra_qfrc is not None:
        bias = bias - extra_qfrc
    blocks = []
    for contact in contacts:
        body = model.geom_bodyid[model.geom(contact.leg).id]
        blocks.append(_point_jacobian(model, data, contact_point(model, data, contact), body))
    jacobian = np.vstack(blocks).T
    count = jacobian.shape[1]
    dofs = [model.joint(name).dofadr[0] for name in joint_names()]
    actuated, load = jacobian[dofs], bias[dofs]
    upper, lower = [], []
    for row, value, limit in zip(actuated, load, limits):
        upper.append(np.r_[-row, -limit])
        lower.append(-value)
        upper.append(np.r_[row, -limit])
        lower.append(value)
    for index, contact in enumerate(contacts):
        normal = contact.normal
        first = np.cross(normal, [0.0, 1.0, 0.0])
        if np.linalg.norm(first) < 1e-6:
            first = np.cross(normal, [1.0, 0.0, 0.0])
        first = first / np.linalg.norm(first)
        second = np.cross(normal, first)
        span = slice(3 * index, 3 * index + 3)
        for a in (1, -1):
            for b in (1, -1):
                row = np.zeros(count + 1)
                row[span] = a * first + b * second - contact.mu * normal
                upper.append(row)
                lower.append(0.0)
        row = np.zeros(count + 1)
        row[span] = -normal
        upper.append(row)
        lower.append(-min_normal.get(contact.leg, 0.0))
        if contact.leg in max_normal:
            row = np.zeros(count + 1)
            row[span] = normal
            upper.append(row)
            lower.append(max_normal[contact.leg])
    cost = np.r_[np.zeros(count), 1.0]
    balance = np.c_[jacobian[:6], np.zeros(6)]
    result = linprog(
        cost,
        A_ub=np.array(upper),
        b_ub=np.array(lower),
        A_eq=balance,
        b_eq=bias[:6],
        bounds=[(None, None)] * count + [(0, None)],
        method="highs",
    )
    if result.status != 0:
        return None
    forces = result.x[:count]
    if smooth:
        forces = _smooth_forces(
            forces, np.array(upper), np.array(lower), jacobian[:6], bias[:6], actuated, load, limits
        )
    torque = load - actuated @ forces
    residual = float(np.abs(jacobian[:6] @ forces - bias[:6]).max())
    return StaticSolution(
        float(np.max(np.abs(torque) / limits)) if smooth else float(result.x[-1]),
        torque,
        {c.leg: forces[3 * i : 3 * i + 3] for i, c in enumerate(contacts)},
        residual,
    )


def _smooth_forces(start, upper, lower, balance, bias, actuated, load, limits) -> np.ndarray:
    """Minimise sum((tau / limit)^2) over contact forces, keeping torques inside the limits."""
    a, b = upper[:, :-1], lower - upper[:, -1]
    weight = 1.0 / limits**2
    regular = 1e-6

    def cost(forces):
        torque = load - actuated @ forces
        value = float(weight @ torque**2 + regular * forces @ forces)
        grad = -2 * actuated.T @ (weight * torque) + 2 * regular * forces
        return value, grad

    result = minimize(
        cost,
        start,
        jac=True,
        method="SLSQP",
        constraints=[
            {"type": "eq", "fun": lambda f: balance @ f - bias, "jac": lambda f: balance},
            {"type": "ineq", "fun": lambda f: b - a @ f, "jac": lambda f: -a},
        ],
        options={"maxiter": 200, "ftol": 1e-12},
    )
    feasible = np.abs(balance @ result.x - bias).max() < 1e-6 and (b - a @ result.x).min() > -1e-6
    return result.x if feasible else start


def max_tip_force(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contacts: list[Contact],
    limits: np.ndarray,
    direction: np.ndarray,
    point: np.ndarray,
    body: int,
) -> float | None:
    """Largest force the robot can push with at `point` along `direction`, or None.

    The valve pushes back with -s * direction. Unknowns are the contact forces and s; the
    constraints are static balance of the floating base, friction pyramids, and joint
    torque limits. Returns s, or None if the pose cannot even stand. tests/test_valve.py checks
    that solve_static with the returned force sits exactly at a torque or friction limit.
    """
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    bias = data.qfrc_bias.copy()
    unit = np.asarray(direction, dtype=float)
    unit = unit / np.linalg.norm(unit)
    push = _point_jacobian(model, data, point, body).T @ unit
    blocks = []
    for contact in contacts:
        owner = model.geom_bodyid[model.geom(contact.leg).id]
        blocks.append(_point_jacobian(model, data, contact_point(model, data, contact), owner))
    jacobian = np.vstack(blocks).T
    count = jacobian.shape[1]
    dofs = [model.joint(name).dofadr[0] for name in joint_names()]
    actuated, load, gain = jacobian[dofs], bias[dofs], push[dofs]
    upper, lower = [], []
    for row, value, extra, limit in zip(actuated, load, gain, limits):
        upper.append(np.r_[-row, extra])
        lower.append(limit - value)
        upper.append(np.r_[row, -extra])
        lower.append(limit + value)
    for index, contact in enumerate(contacts):
        normal = contact.normal
        first = np.cross(normal, [0.0, 1.0, 0.0])
        if np.linalg.norm(first) < 1e-6:
            first = np.cross(normal, [1.0, 0.0, 0.0])
        first = first / np.linalg.norm(first)
        second = np.cross(normal, first)
        span = slice(3 * index, 3 * index + 3)
        for a in (1, -1):
            for b in (1, -1):
                row = np.zeros(count + 1)
                row[span] = a * first + b * second - contact.mu * normal
                upper.append(row)
                lower.append(0.0)
        row = np.zeros(count + 1)
        row[span] = -normal
        upper.append(row)
        lower.append(0.0)
    balance = np.c_[jacobian[:6], -push[:6]]
    result = linprog(
        np.r_[np.zeros(count), -1.0],
        A_ub=np.array(upper),
        b_ub=np.array(lower),
        A_eq=balance,
        b_eq=bias[:6],
        bounds=[(None, None)] * count + [(0, 1000.0)],
        method="highs",
    )
    return float(result.x[-1]) if result.status == 0 else None
