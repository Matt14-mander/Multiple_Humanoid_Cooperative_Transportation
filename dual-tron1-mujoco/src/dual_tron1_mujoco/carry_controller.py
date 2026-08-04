"""Object-level impedance and adaptive quadratic load sharing."""

from typing import Dict

import math

import mujoco
import numpy as np

from .model_index import ModelIndex


ARM_JOINTS = ("J1", "J2", "J3", "J4", "J5", "J6")


def _vector(config: Dict[str, object], name: str, size: int) -> np.ndarray:
    value = np.asarray(config[name], dtype=float)
    if value.shape != (size,):
        raise ValueError("carry_impedance.{} must contain {} values".format(name, size))
    return value


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _orientation_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Return the world-frame SO(3) logarithm from current to desired.

    The former cross-product approximation tends back to zero near 180 degrees,
    exactly when the payload most needs a recovery moment.
    """
    relative = desired @ current.T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    angle = math.acos(float(cosine))
    vee = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    )
    if angle < 1e-7:
        return 0.5 * vee
    if math.pi - angle < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eigh(relative)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        return angle * axis / max(np.linalg.norm(axis), 1e-12)
    return angle * vee / (2.0 * math.sin(angle))


def _solve_inequality_qp(
    quadratic: np.ndarray,
    linear_rhs: np.ndarray,
    inequalities: np.ndarray,
    bounds: np.ndarray,
    maximum_iterations: int = 80,
) -> np.ndarray:
    """Small active-set solver for min 0.5*x'Q*x-rhs'x, C*x<=d."""
    active = []
    solution = np.linalg.solve(quadratic, linear_rhs)
    for _ in range(maximum_iterations):
        violation = inequalities @ solution - bounds
        candidate = int(np.argmax(violation))
        if violation[candidate] <= 1e-8:
            return solution
        if candidate not in active:
            active.append(candidate)
        active_matrix = inequalities[active]
        kkt = np.block(
            [
                [quadratic, active_matrix.T],
                [active_matrix, np.zeros((len(active), len(active)))],
            ]
        )
        rhs = np.concatenate((linear_rhs, bounds[active]))
        result = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
        solution = result[: quadratic.shape[0]]
        multipliers = result[quadratic.shape[0] :]
        if multipliers.size and np.min(multipliers) < -1e-8:
            del active[int(np.argmin(multipliers))]
    return solution


class CooperativeCarryHoldController:
    """Hold one payload using object impedance and quadratic load sharing.

    The commanded end-effector wrenches sum to one desired payload wrench.
    Payload mass and body-frame COM can be updated from the independent
    payload estimator; the controller never folds them into either arm model.
    """

    def __init__(self, model, data, config: Dict[str, object]):
        self.model = model
        self.index = ModelIndex(model)
        self.config = config
        self.payload_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "payload_body"
        )
        if self.payload_id < 0:
            raise KeyError("MuJoCo body not found: payload_body")
        self.payload_mass = float(model.body_subtreemass[self.payload_id])
        self.payload_parameter_source = str(
            config.get("payload_parameter_source", "model_truth")
        )
        if self.payload_parameter_source == "model_truth":
            self.estimated_payload_mass = self.payload_mass
            self.estimated_payload_com_body = model.body_ipos[
                self.payload_id
            ].copy()
        elif self.payload_parameter_source == "external_estimator":
            self.estimated_payload_mass = float(
                config.get("payload_prior_mass_kg", self.payload_mass)
            )
            self.estimated_payload_com_body = np.asarray(
                config.get("payload_prior_com_m", [0.0, 0.0, 0.0]),
                dtype=float,
            )
        else:
            raise ValueError(
                "payload_parameter_source must be model_truth or external_estimator"
            )
        self.payload_position_ref = data.xpos[self.payload_id].copy()
        self.payload_rotation_ref = data.xmat[self.payload_id].reshape(3, 3).copy()
        self.reference_frame = str(config.get("reference_frame", "world"))
        if self.reference_frame not in {"world", "base_centroid"}:
            raise ValueError(
                "carry_impedance.reference_frame must be world or base_centroid"
            )
        self.actuation_mode = str(config.get("actuation_mode", "absolute"))
        if self.actuation_mode not in {"absolute", "additive_policy"}:
            raise ValueError(
                "carry_impedance.actuation_mode must be absolute or additive_policy"
            )
        self.load_ramp_s = float(config.get("load_ramp_s", 0.0))
        self.base_ids = [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_Link"
            )
            for prefix in ("r1_", "r2_")
        ]
        self.base_centroid_ref = np.mean(data.xpos[self.base_ids], axis=0)
        self.base_yaw_ref = self._average_base_yaw(data)
        self.translation_stiffness = _vector(config, "translation_stiffness", 3)
        self.translation_damping = _vector(config, "translation_damping", 3)
        self.rotation_stiffness = _vector(config, "rotation_stiffness", 3)
        self.rotation_damping = _vector(config, "rotation_damping", 3)
        self.joint_posture_kp = float(config.get("joint_posture_kp", 0.5))
        self.joint_damping = float(config.get("joint_damping", 0.2))
        self.torque_rate_limit = float(config.get("torque_rate_limit_nm_s", 500.0))
        self.max_wrench_force = float(config.get("max_wrench_force_n", 40.0))
        self.max_wrench_torque = float(config.get("max_wrench_torque_nm", 5.0))
        self.wrench_tracking_weight = float(
            config.get("wrench_tracking_weight", 1e5)
        )
        self.feasibility_tolerance = float(
            config.get("wrench_feasibility_tolerance", 0.03)
        )
        force_weights = np.asarray(
            config.get("load_sharing_force_weights", [1.0, 1.0]), dtype=float
        )
        if force_weights.shape != (2,) or np.any(force_weights <= 0.0):
            raise ValueError(
                "carry_impedance.load_sharing_force_weights must contain "
                "two positive values"
            )
        moment_weight = float(config.get("load_sharing_moment_weight", 100.0))
        if moment_weight <= 0.0:
            raise ValueError(
                "carry_impedance.load_sharing_moment_weight must be positive"
            )
        self.load_sharing_weights = np.concatenate(
            [
                np.concatenate((np.full(3, weight), np.full(3, moment_weight)))
                for weight in force_weights
            ]
        )
        self.arms = []
        for prefix in ("r1_", "r2_"):
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, prefix + "link6"
            )
            joints = [self.index.joint(prefix + name) for name in ARM_JOINTS]
            actuators = [self.index.actuator(prefix + name) for name in ARM_JOINTS]
            q_ref = np.array([data.qpos[joint.qpos_adr] for joint in joints])
            limits = np.array(
                [
                    float(config["arm_j123_torque_limit"])
                    if number < 3
                    else float(config["arm_j456_torque_limit"])
                    for number in range(6)
                ]
            )
            self.arms.append(
                {
                    "body_id": body_id,
                    "joints": joints,
                    "actuators": actuators,
                    "q_ref": q_ref,
                    "limits": limits,
                    "previous_torque": np.zeros(6),
                }
            )
        self.last_payload_wrench = np.zeros(6)
        self.last_payload_com_world = data.xpos[self.payload_id].copy()
        self.last_arm_wrenches = np.zeros((2, 6))
        self.last_arm_torques = np.zeros((2, 6))
        self.last_predicted_arm_torques = np.zeros((2, 6))
        self.last_achieved_payload_wrench = np.zeros(6)
        self.last_allocation_residual = np.zeros(6)
        self.last_vertical_support_ratio = 0.0
        self.last_allocation_feasible = True
        self.peak_abs_arm_torques = np.zeros((2, 6))
        self.saturation_steps = np.zeros((2, 6), dtype=np.int64)
        self.update_count = 0
        self._first_update = True

    def set_payload_estimate(
        self, mass_kg: float, com_body_m: np.ndarray
    ) -> None:
        """Apply one accepted object-level estimate to load compensation."""
        mass = float(mass_kg)
        com = np.asarray(com_body_m, dtype=float)
        if mass <= 0.0:
            raise ValueError("estimated payload mass must be positive")
        if com.shape != (3,) or not np.all(np.isfinite(com)):
            raise ValueError("estimated payload COM must contain three values")
        self.estimated_payload_mass = mass
        self.estimated_payload_com_body = com.copy()

    def assess_static_capacity(self, data) -> Dict[str, object]:
        """Evaluate the initial gravity-support request without actuating motors."""
        payload_wrench = self._payload_wrench(data)
        torque_maps, base_torques = self._arm_torque_maps(data)
        self._share_wrench(data, payload_wrench, torque_maps, base_torques)
        return {
            "feasible": self.last_allocation_feasible,
            "desired_wrench": payload_wrench.copy(),
            "achieved_wrench": self.last_achieved_payload_wrench.copy(),
            "residual": self.last_allocation_residual.copy(),
            "vertical_support_ratio": self.last_vertical_support_ratio,
            "predicted_arm_torques": self.last_predicted_arm_torques.copy(),
        }

    def _average_base_yaw(self, data) -> float:
        yaws = [
            math.atan2(
                data.xmat[body_id].reshape(3, 3)[1, 0],
                data.xmat[body_id].reshape(3, 3)[0, 0],
            )
            for body_id in self.base_ids
        ]
        return math.atan2(
            sum(math.sin(yaw) for yaw in yaws),
            sum(math.cos(yaw) for yaw in yaws),
        )

    def _reference_pose(self, data):
        if self.reference_frame == "world":
            return self.payload_position_ref, self.payload_rotation_ref
        centroid = np.mean(data.xpos[self.base_ids], axis=0)
        position = self.payload_position_ref + (
            centroid - self.base_centroid_ref
        )
        delta_yaw = self._average_base_yaw(data) - self.base_yaw_ref
        cosine, sine = math.cos(delta_yaw), math.sin(delta_yaw)
        yaw_rotation = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        return position, yaw_rotation @ self.payload_rotation_ref

    def _reference_velocity(self, data):
        if self.reference_frame == "world":
            return np.zeros(3), np.zeros(3)
        base_velocities = []
        for body_id in self.base_ids:
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
                velocity,
                0,
            )
            base_velocities.append(velocity)
        average = np.mean(base_velocities, axis=0)
        angular_reference = np.array([0.0, 0.0, average[2]])
        return angular_reference, average[3:]

    def _payload_wrench(self, data) -> np.ndarray:
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.payload_id,
            velocity,
            0,
        )
        angular_velocity = velocity[:3]
        linear_velocity = velocity[3:]
        angular_velocity_ref, linear_velocity_ref = self._reference_velocity(data)
        position_ref, rotation_ref = self._reference_pose(data)
        position_error = position_ref - data.xpos[self.payload_id]
        rotation = data.xmat[self.payload_id].reshape(3, 3)
        rotation_error = _orientation_error(rotation, rotation_ref)
        force = (
            self.translation_stiffness * position_error
            + self.translation_damping * (linear_velocity_ref - linear_velocity)
            - self.estimated_payload_mass * np.asarray(self.model.opt.gravity)
        )
        moment = (
            self.rotation_stiffness * rotation_error
            + self.rotation_damping * (angular_velocity_ref - angular_velocity)
        )
        force = np.clip(force, -self.max_wrench_force, self.max_wrench_force)
        moment = np.clip(moment, -self.max_wrench_torque, self.max_wrench_torque)
        return np.concatenate((force, moment))

    def _arm_torque_maps(self, data):
        maps = []
        base_torques = []
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        for arm in self.arms:
            jacobian_position.fill(0.0)
            jacobian_rotation.fill(0.0)
            mujoco.mj_jacBody(
                self.model,
                data,
                jacobian_position,
                jacobian_rotation,
                arm["body_id"],
            )
            dofs = np.array([joint.dof_adr for joint in arm["joints"]])
            maps.append(
                np.hstack(
                    (
                        jacobian_position[:, dofs].T,
                        jacobian_rotation[:, dofs].T,
                    )
                )
            )
            if self.actuation_mode == "additive_policy":
                base_torques.append(
                    np.array([float(data.ctrl[a]) for a in arm["actuators"]])
                )
            else:
                posture = np.array(
                    [
                        self.joint_posture_kp
                        * (
                            arm["q_ref"][number]
                            - float(data.qpos[joint.qpos_adr])
                        )
                        - self.joint_damping * float(data.qvel[joint.dof_adr])
                        for number, joint in enumerate(arm["joints"])
                    ]
                )
                base_torques.append(data.qfrc_bias[dofs] + posture)
        return np.asarray(maps), np.asarray(base_torques)

    def _share_wrench(
        self,
        data,
        payload_wrench: np.ndarray,
        torque_maps=None,
        base_torques=None,
    ) -> np.ndarray:
        payload_position = data.xpos[self.payload_id]
        payload_rotation = data.xmat[self.payload_id].reshape(3, 3)
        payload_com = (
            payload_position
            + payload_rotation @ self.estimated_payload_com_body
        )
        grasp_matrix = np.zeros((6, 12))
        for arm_number, arm in enumerate(self.arms):
            offset = data.xpos[arm["body_id"]] - payload_com
            column = arm_number * 6
            grasp_matrix[:3, column : column + 3] = np.eye(3)
            grasp_matrix[3:, column : column + 3] = _skew(offset)
            grasp_matrix[3:, column + 3 : column + 6] = np.eye(3)
        inverse_weights = 1.0 / self.load_sharing_weights
        weighted_dual = (
            grasp_matrix
            @ (inverse_weights[:, None] * grasp_matrix.T)
        )
        solution = inverse_weights * (
            grasp_matrix.T @ np.linalg.lstsq(
                weighted_dual, payload_wrench, rcond=None
            )[0]
        )
        if torque_maps is None or base_torques is None:
            torque_maps, base_torques = self._arm_torque_maps(data)
        predicted = np.asarray(
            [
                base + mapping @ solution[number * 6 : number * 6 + 6]
                for number, (mapping, base) in enumerate(
                    zip(torque_maps, base_torques)
                )
            ]
        )
        within_limits = all(
            np.all(np.abs(torque) <= arm["limits"] + 1e-8)
            for torque, arm in zip(predicted, self.arms)
        )
        if not within_limits:
            torque_matrix = np.zeros((12, 12))
            limits = np.zeros(12)
            base = base_torques.reshape(-1)
            for number, (arm, mapping) in enumerate(zip(self.arms, torque_maps)):
                rows = slice(number * 6, number * 6 + 6)
                torque_matrix[rows, rows] = mapping
                limits[rows] = arm["limits"]
            inequalities = np.vstack((torque_matrix, -torque_matrix))
            inequality_bounds = np.concatenate((limits - base, limits + base))
            quadratic = np.diag(self.load_sharing_weights) + (
                self.wrench_tracking_weight * grasp_matrix.T @ grasp_matrix
            )
            linear_rhs = (
                self.wrench_tracking_weight * grasp_matrix.T @ payload_wrench
            )
            solution = _solve_inequality_qp(
                quadratic,
                linear_rhs,
                inequalities,
                inequality_bounds,
            )
            predicted = np.asarray(
                [
                    base_torques[number]
                    + torque_maps[number]
                    @ solution[number * 6 : number * 6 + 6]
                    for number in range(2)
                ]
            )
        achieved = grasp_matrix @ solution
        residual = achieved - payload_wrench
        desired_vertical = abs(float(payload_wrench[2]))
        self.last_achieved_payload_wrench = achieved
        self.last_allocation_residual = residual
        self.last_vertical_support_ratio = (
            float(achieved[2] / payload_wrench[2])
            if desired_vertical > 1e-9
            else 1.0
        )
        relative_residual = np.linalg.norm(residual) / max(
            np.linalg.norm(payload_wrench), 1.0
        )
        self.last_allocation_feasible = bool(
            relative_residual <= self.feasibility_tolerance
            and all(
                np.all(np.abs(torque) <= arm["limits"] + 1e-6)
                for torque, arm in zip(predicted, self.arms)
            )
        )
        self.last_predicted_arm_torques = predicted
        self.last_payload_com_world = payload_com
        return solution.reshape(2, 6)

    def update(self, data, arm_wrench_correction=None) -> None:
        payload_wrench = self._payload_wrench(data)
        torque_maps, base_torques = self._arm_torque_maps(data)
        arm_wrenches = self._share_wrench(
            data, payload_wrench, torque_maps, base_torques
        )
        if arm_wrench_correction is not None:
            correction = np.asarray(arm_wrench_correction, dtype=float)
            if correction.shape != (2, 6):
                raise ValueError("arm_wrench_correction must have shape (2, 6)")
            arm_wrenches = arm_wrenches + correction
        load_scale = (
            min(1.0, float(data.time) / self.load_ramp_s)
            if self.load_ramp_s > 0.0
            else 1.0
        )
        applied_arm_wrenches = arm_wrenches * load_scale
        timestep = float(self.model.opt.timestep)
        max_delta = self.torque_rate_limit * timestep
        for arm_number, (arm, wrench) in enumerate(
            zip(self.arms, applied_arm_wrenches)
        ):
            commanded = base_torques[arm_number] + (
                torque_maps[arm_number] @ wrench
            )
            if self.actuation_mode == "absolute" and not self._first_update:
                commanded = np.clip(
                    commanded,
                    arm["previous_torque"] - max_delta,
                    arm["previous_torque"] + max_delta,
                )
            commanded = np.clip(commanded, -arm["limits"], arm["limits"])
            for actuator, torque in zip(arm["actuators"], commanded):
                data.ctrl[actuator] = torque
            arm["previous_torque"] = commanded.copy()
            self.last_arm_torques[arm_number] = commanded
            self.peak_abs_arm_torques[arm_number] = np.maximum(
                self.peak_abs_arm_torques[arm_number], np.abs(commanded)
            )
            self.saturation_steps[arm_number] += (
                np.abs(commanded) >= arm["limits"] - 1e-9
            )

        self.last_payload_wrench = payload_wrench
        self.last_arm_wrenches = applied_arm_wrenches
        self.update_count += 1
        self._first_update = False

    @property
    def saturation_fractions(self) -> np.ndarray:
        if self.update_count == 0:
            return np.zeros((2, 6))
        return self.saturation_steps / float(self.update_count)
