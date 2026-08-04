"""Object-level impedance and symmetric load sharing for static carrying."""

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
    return 0.5 * sum(
        (np.cross(current[:, axis], desired[:, axis]) for axis in range(3)),
        start=np.zeros(3),
    )


class CooperativeCarryHoldController:
    """Hold one payload using object impedance and minimum-norm load sharing.

    The commanded end-effector wrenches sum to one desired payload wrench.
    No null-space/internal grasp wrench is deliberately introduced in this
    first milestone.
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
        self.last_arm_wrenches = np.zeros((2, 6))
        self.last_arm_torques = np.zeros((2, 6))
        self.peak_abs_arm_torques = np.zeros((2, 6))
        self.saturation_steps = np.zeros((2, 6), dtype=np.int64)
        self.update_count = 0

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
            - self.payload_mass * np.asarray(self.model.opt.gravity)
        )
        moment = (
            self.rotation_stiffness * rotation_error
            + self.rotation_damping * (angular_velocity_ref - angular_velocity)
        )
        force = np.clip(force, -self.max_wrench_force, self.max_wrench_force)
        moment = np.clip(moment, -self.max_wrench_torque, self.max_wrench_torque)
        return np.concatenate((force, moment))

    def _share_wrench(self, data, payload_wrench: np.ndarray) -> np.ndarray:
        payload_position = data.xpos[self.payload_id]
        grasp_matrix = np.zeros((6, 12))
        for arm_number, arm in enumerate(self.arms):
            offset = data.xpos[arm["body_id"]] - payload_position
            column = arm_number * 6
            grasp_matrix[:3, column : column + 3] = np.eye(3)
            grasp_matrix[3:, column : column + 3] = _skew(offset)
            grasp_matrix[3:, column + 3 : column + 6] = np.eye(3)
        return (np.linalg.pinv(grasp_matrix) @ payload_wrench).reshape(2, 6)

    def update(self, data, arm_wrench_correction=None) -> None:
        payload_wrench = self._payload_wrench(data)
        arm_wrenches = self._share_wrench(data, payload_wrench)
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
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))

        for arm_number, (arm, wrench) in enumerate(
            zip(self.arms, applied_arm_wrenches)
        ):
            jacobian_position.fill(0.0)
            jacobian_rotation.fill(0.0)
            mujoco.mj_jacBody(
                self.model,
                data,
                jacobian_position,
                jacobian_rotation,
                arm["body_id"],
            )
            generalized_torque = (
                jacobian_position.T @ wrench[:3]
                + jacobian_rotation.T @ wrench[3:]
            )
            commanded = np.zeros(6)
            for joint_number, (joint, actuator) in enumerate(
                zip(arm["joints"], arm["actuators"])
            ):
                if self.actuation_mode == "additive_policy":
                    commanded[joint_number] = (
                        float(data.ctrl[actuator])
                        + generalized_torque[joint.dof_adr]
                    )
                else:
                    q = float(data.qpos[joint.qpos_adr])
                    dq = float(data.qvel[joint.dof_adr])
                    commanded[joint_number] = (
                        generalized_torque[joint.dof_adr]
                        + float(data.qfrc_bias[joint.dof_adr])
                        + self.joint_posture_kp
                        * (arm["q_ref"][joint_number] - q)
                        - self.joint_damping * dq
                    )
            if self.actuation_mode == "absolute":
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

    @property
    def saturation_fractions(self) -> np.ndarray:
        if self.update_count == 0:
            return np.zeros((2, 6))
        return self.saturation_steps / float(self.update_count)
