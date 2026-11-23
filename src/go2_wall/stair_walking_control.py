"""Motor-only whole-body control for the ascent and descent alongside the released valve."""

import mujoco
import numpy as np

from .pose_search import FRICTION_MARGIN
from .robot import LEGS
from .statics import joint_names


def distributed_forces(balance, load, reference, constraints, bounds):
    """Project the planned forces onto floating-base balance and a capped friction pyramid."""
    active = []
    row_norms = np.linalg.norm(constraints, axis=1)
    for _ in range(25):
        equality = np.vstack((balance, constraints[active])) if active else balance
        target = np.r_[load, bounds[active]] if active else load
        multipliers = np.linalg.lstsq(
            equality @ equality.T, equality @ reference - target, rcond=1e-11
        )[0]
        forces = reference - equality.T @ multipliers
        if np.max(np.abs(equality @ forces - target)) > 0.04:
            return None
        excess = (constraints @ forces - bounds) / row_norms
        if excess.max() <= 1e-5:
            if active and multipliers[6:].min() < -0.0001:
                del active[int(np.argmin(multipliers[6:]))]
                continue
            return forces
        index = int(np.argmax(excess))
        if index in active:
            return None
        active.append(index)
    return None


def friction_constraints(leg_count, mu):
    rows = np.array([[1, 1, -mu], [1, -1, -mu], [-1, 1, -mu], [-1, -1, -mu], [0, 0, -1], [0, 0, 1]])
    return np.kron(np.eye(leg_count), rows)


def inward_normals(polygon):
    edges = np.roll(polygon, -1, axis=0) - polygon
    return np.column_stack((-edges[:, 1], edges[:, 0])) / np.maximum(
        np.linalg.norm(edges, axis=1, keepdims=True), 1e-8
    )


class StairWalkingControl:
    """Follow a frozen foothold schedule using measured posture and stance contact velocity."""

    def __init__(self, model, track, windows, servo_s, floor_mu):
        self.windows = windows
        self.servo_s = servo_s
        self.mu = floor_mu * FRICTION_MARGIN
        self.base = model.body("base").id
        self.feet = np.array([model.geom(leg).id for leg in LEGS])
        self.joint_ids = np.array([model.joint(name).id for name in joint_names()])
        self.dofs = model.jnt_dofadr[self.joint_ids]
        self.positions = model.jnt_qposadr[self.joint_ids]
        self.hips = np.array([model.body(f"{leg}_hip").pos for leg in LEGS])
        self.thigh_y = np.array([model.body(f"{leg}_thigh").pos[1] for leg in LEGS])
        self.upper = np.array([-model.body(f"{leg}_calf").pos[2] for leg in LEGS])
        tips = np.array([model.body(f"{leg}_foot").pos + model.geom(leg).pos for leg in LEGS])
        self.lower = np.linalg.norm(tips, axis=1)
        self.ankle = np.arctan2(-tips[:, 0], -tips[:, 2])
        self.force_constraints = {n: friction_constraints(n, self.mu) for n in (3, 4)}
        self.slip = np.zeros((4, 3))
        self.shift = np.zeros(2)
        self.previous_window = None
        self.max_shift = 0.0
        self.attempts = [0] * 5
        self.unsolved = 0
        times = track["time"]
        rows = np.arange(len(times))
        flags = track["stance"]
        self.support = flags & np.vstack((flags[1:], flags[-1:]))
        future = np.where(self.support.sum(axis=1) == 3, rows, rows[-1])
        self.next_tripod = np.minimum.accumulate(future[::-1])[::-1]
        landing = np.where(flags, rows[:, None], rows[-1])
        self.next_touchdown = times[np.minimum.accumulate(landing[::-1], axis=0)[::-1]]
        speed = np.gradient(track["target"][:, :12], times, axis=0)
        self.swing_accel = np.clip(np.gradient(speed, times, axis=0), -70, 70)
        self.planned_com = np.zeros((len(times), 2))
        scratch = mujoco.MjData(model)
        for start, stop, _ in windows:
            for row in np.flatnonzero((times >= start - 0.03) & (times <= stop + 0.03)):
                pose = track["qpos"][row]
                scratch.qpos[: len(pose)] = pose
                mujoco.mj_fwdPosition(model, scratch)
                self.planned_com[row] = scratch.subtree_com[self.base, :2]

    def describe(self):
        return {
            "windows_s_and_tripod_margin_m": self.windows,
            "friction_pyramid_mu": self.mu,
            "servo_s": self.servo_s,
            "attempt_counts": self.attempts,
            "failed_distributions": self.unsolved,
            "largest_support_correction_m": float(self.max_shift),
        }

    def support_shift(self, data, track, row, now, desired, margin):
        future = self.next_tripod[row]
        delay = max(0.0, track["time"][future] - now)
        requested = np.zeros(2)
        if margin and delay < 1.8 and self.support[future].sum() == 3:
            # Counterclockwise in base coordinates: FL, RL, RR, FR.
            legs = [k for k in (0, 2, 3, 1) if self.support[future, k]]
            measured = data.geom_xpos[self.feet[legs], :2]
            planned = track["feet"][row, legs, :2]
            edges = inward_normals(measured)
            target_margin = np.sum(
                (self.planned_com[row] - planned) * inward_normals(planned), axis=1
            )
            target_margin = np.minimum(margin, target_margin + 0.65 * margin)
            com = data.subtree_com[self.base, :2] + desired[:2] - data.qpos[:2]
            for _ in range(2):
                for point, normal, minimum in zip(measured, edges, target_margin, strict=True):
                    gap = (com + requested - point) @ normal
                    requested += max(0.0, minimum - gap) * normal
            length = np.linalg.norm(requested)
            requested *= min(1.0, 0.024 / max(length, 1e-10))
            blend = 1 - delay / 1.8
            requested *= blend**2 * (3 - 2 * blend)
        self.shift += self.servo_s / 0.16 * (requested - self.shift)
        self.max_shift = max(self.max_shift, np.linalg.norm(self.shift))
        return self.shift

    def swing_joints(self, model, data, leg, target, reference):
        point = (target - data.qpos[:3]) @ data.xmat[self.base].reshape(3, 3) - self.hips[leg]
        x, y, z = point
        side, upper, lower = self.thigh_y[leg], self.upper[leg], self.lower[leg]
        projected = -np.sqrt(max(y * y + z * z - side * side, 0.0))
        hip = np.arctan2(z, y) - np.arctan2(projected, side)
        knee = -np.arccos(
            np.clip((x * x + projected**2 - upper**2 - lower**2) / (2 * upper * lower), -1, 1)
        )
        thigh = np.arctan2(-x, -projected) - np.arctan2(
            lower * np.sin(knee), upper + lower * np.cos(knee)
        )
        solution = np.array([hip, thigh, knee - self.ankle[leg]])
        solution = reference + (solution - reference + np.pi) % (2 * np.pi) - np.pi
        limits = model.jnt_range[self.joint_ids[3 * leg : 3 * leg + 3]]
        return np.clip(solution, limits[:, 0], limits[:, 1])

    def command(self, model, data, track, row, weight, stance, nominal_forces, now, original):
        window = next((w for w in self.windows if w[0] <= now <= w[1]), None)
        if window is None:
            self.previous_window = None
            return original, 0
        if self.previous_window != window:
            self.slip.fill(0)
            self.shift.fill(0)
            self.previous_window = window
        start, stop, margin = window
        times = track["time"]
        dt = times[row + 1] - times[row]
        q0, q1 = track["qpos"][row : row + 2]
        desired = (1 - weight) * q0 + weight * q1
        desired[3:7] /= np.linalg.norm(desired[3:7])
        first, second = data.qpos.copy(), data.qpos.copy()
        first[: len(q0)], second[: len(q1)] = q0, q1
        velocity = np.zeros(model.nv)
        error = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, velocity, dt, first, second)
        first[: len(desired)] = desired
        mujoco.mj_differentiatePos(model, error, 1.0, data.qpos, first)
        error[:2] += self.support_shift(data, track, row, now, desired, margin)
        speed_error = velocity - data.qvel
        accel = np.zeros(model.nv)
        accel[:3] = np.clip(10.5**2 * error[:3] + 21 * speed_error[:3], -2.3, 2.3)
        accel[3:6] = np.clip(15**2 * error[3:6] + 30 * speed_error[3:6], -9, 9)
        arm = self.dofs[12:]
        accel[arm] = np.clip(65 * error[arm] + 15 * speed_error[arm], -25, 25)
        targets = (1 - weight) * track["feet"][row] + weight * track["feet"][row + 1]
        contacts = []
        servo = np.zeros(len(self.dofs))
        for leg, geom in enumerate(self.feet):
            body = model.geom_bodyid[geom]
            centre = data.geom_xpos[geom]
            point = centre.copy()
            if stance[leg]:
                point[2] -= model.geom_size[geom, 0]
            jac = np.zeros((3, model.nv))
            derivative = np.zeros_like(jac)
            mujoco.mj_jac(model, data, jac, None, point, body)
            mujoco.mj_jacDot(model, data, derivative, None, point, body)
            columns = slice(3 * leg, 3 * leg + 3)
            dofs = self.dofs[columns]
            if stance[leg]:
                contacts.append(jac.T)
                foot_velocity = jac @ data.qvel
                self.slip[leg] += foot_velocity * self.servo_s
                goal = -18.0 * foot_velocity
                goal[:2] += np.clip(15.0 * (targets[leg, :2] - centre[:2]), -0.2, 0.2)
                goal[:2] -= np.clip(140.0 * self.slip[leg, :2], -0.7, 0.7)
                local = jac[:, dofs]
                residual = goal - jac[:, :6] @ accel[:6] - derivative @ data.qvel
                accel[dofs] = np.clip(
                    np.linalg.solve(local.T @ local + np.eye(3) * 0.00004, local.T @ residual),
                    -80,
                    80,
                )
            else:
                self.slip[leg].fill(0)
                target = targets[leg].copy()
                blend = np.clip(1 - (self.next_touchdown[row + 1, leg] - now) / 0.22, 0, 1)
                target[2] -= 0.005 * blend**2 * (3 - 2 * blend)
                positions = self.positions[columns]
                joints = self.swing_joints(model, data, leg, target, desired[positions])
                joint_error = joints - data.qpos[positions]
                feed = (1 - weight) * self.swing_accel[row, columns] + weight * self.swing_accel[
                    row + 1, columns
                ]
                accel[dofs] = np.clip(feed + 310 * joint_error + 34 * speed_error[dofs], -95, 95)
                servo[columns] = 68 * joint_error + 0.25 * np.sqrt(68) * speed_error[dofs]
        count = len(contacts)
        if count not in self.force_constraints:
            self.unsolved += 1
            return original, -1
        jacobian = np.concatenate(contacts, axis=1)
        limits = track["cap"][row, stance]
        bounds = np.zeros(6 * count)
        reserve = 6.0 if margin else 1.0
        bounds[4::6] = -np.minimum(reserve, limits)
        bounds[5::6] = np.minimum(limits, 650.0)
        inertial = np.zeros(model.nv)
        mujoco.mj_mulM(model, data, inertial, accel)
        bias = data.qfrc_bias - data.qfrc_passive
        for attempt, scale in enumerate((1.0, 0.7, 0.4, 0.15, 0.0)):
            total = bias + scale * inertial
            forces = distributed_forces(
                jacobian[:6],
                total[:6],
                nominal_forces[stance].ravel(),
                self.force_constraints[count],
                bounds,
            )
            if forces is not None:
                self.attempts[attempt] += 1
                blend = np.clip(min(now - start, stop - now) / 0.45, 0, 1)
                blend = blend**2 * (3 - 2 * blend)
                torque = (total - jacobian @ forces)[self.dofs] + servo
                return (1 - blend) * original + blend * torque, attempt
        self.unsolved += 1
        return original, -1
