"""Gravity-compensated Cartesian sanity controller; not a learned policy."""

import mujoco
import numpy as np

from .model_index import ModelIndex


LEG_JOINTS = (
    "proximal_pitch_L_Joint", "proximal_roll_L_Joint",
    "proximal_yaw_L_Joint", "knee_L_Joint", "wheel_L_Joint",
    "proximal_pitch_R_Joint", "proximal_roll_R_Joint",
    "proximal_yaw_R_Joint", "knee_R_Joint", "wheel_R_Joint",
)
ARM_JOINTS = tuple("arm{}_Joint".format(i) for i in range(1, 7))
GRIPPER_JOINTS = ("gripper1_Joint", "gripper2_Joint")
JOINTS = LEG_JOINTS + ARM_JOINTS + GRIPPER_JOINTS


class AnalyticGoalController:
    """Cartesian spring-damper surrogate used only to wire the CHIP pipeline."""

    def __init__(self, model, data, config):
        self.model = model
        self.config = config
        index = ModelIndex(model)
        self.joints = [index.joint(name) for name in JOINTS]
        self.actuators = [index.actuator(name.removesuffix("_Joint")) for name in JOINTS]
        self.arm_dofs = np.array([index.joint(name).dof_adr for name in ARM_JOINTS])
        self.q_ref = np.array([data.qpos[j.qpos_adr] for j in self.joints], dtype=float)
        self.q_nominal = self.q_ref.copy()
        self.gravity_data = mujoco.MjData(model)
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_pick")
        self.cartesian_target = data.xpos[self.ee_body_id].copy()
        self.cartesian_stiffness = np.asarray(
            self.config["cartesian_default_stiffness_n_per_m"], dtype=float
        )

    def update_cartesian_reference(self, data, target_world, compliance=None):
        self.cartesian_target = np.asarray(target_world, dtype=float).copy()
        if compliance is None:
            return
        value = np.asarray(compliance, dtype=float)
        if value.ndim == 0:
            value = np.repeat(value, 3)
        elif value.shape == (3, 3):
            value = np.diag(value)
        if value.shape != (3,):
            raise ValueError("analytic controller requires scalar, diagonal or length-3 compliance")
        default = np.asarray(
            self.config["cartesian_default_stiffness_n_per_m"], dtype=float
        )
        minimum = float(self.config["cartesian_min_stiffness_n_per_m"])
        maximum = float(self.config["cartesian_max_stiffness_n_per_m"])
        stiffness = default.copy()
        enabled = value > 1e-12
        stiffness[enabled] = 1.0 / value[enabled]
        self.cartesian_stiffness = np.clip(stiffness, minimum, maximum)

    def apply(self, data):
        feedforward = self._feedforward(data)
        task_torque = self._cartesian_task_torque(data)
        max_fraction = 0.0
        for number, (name, joint, actuator) in enumerate(zip(JOINTS, self.joints, self.actuators)):
            q = float(data.qpos[joint.qpos_adr])
            dq = float(data.qvel[joint.dof_adr])
            kp, kd, limit = self._gains(name)
            torque = feedforward[joint.dof_adr] + kp * (self.q_ref[number] - q) - kd * dq
            if name in ARM_JOINTS:
                torque += task_torque[joint.dof_adr]
            data.ctrl[actuator] = np.clip(torque, -limit, limit)
            max_fraction = max(max_fraction, abs(float(data.ctrl[actuator])) / limit)
        return max_fraction

    def _cartesian_task_torque(self, data):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, data, jacp, jacr, self.ee_body_id)
        velocity = jacp @ data.qvel
        error = self.cartesian_target - data.xpos[self.ee_body_id]
        damping = np.asarray(self.config["cartesian_damping_ns_per_m"], dtype=float)
        force = self.cartesian_stiffness * error - damping * velocity
        force_limit = float(self.config["cartesian_force_limit_n"])
        force_norm = float(np.linalg.norm(force))
        if force_norm > force_limit:
            force *= force_limit / force_norm
        torque = np.zeros(self.model.nv)
        torque[self.arm_dofs] = jacp[:, self.arm_dofs].T @ force
        return torque

    def _feedforward(self, data):
        if self.config.get("gravity_compensation", "gravity_only") == "full_bias":
            return data.qfrc_bias
        scratch = self.gravity_data
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = 0.0
        scratch.qacc[:] = 0.0
        mujoco.mj_forward(self.model, scratch)
        return scratch.qfrc_bias

    def _gains(self, name):
        if name.startswith("wheel_"):
            return 0.0, float(self.config["wheel_kd"]), float(self.config["wheel_torque_limit"])
        if name.startswith("arm"):
            if name in ARM_JOINTS[3:]:
                return float(self.config["wrist_kp"]), float(self.config["wrist_kd"]), float(self.config["wrist_torque_limit"])
            return float(self.config["arm_kp"]), float(self.config["arm_kd"]), float(self.config["arm_torque_limit"])
        if name.startswith("gripper"):
            return float(self.config["gripper_kp"]), float(self.config["gripper_kd"]), float(self.config["gripper_force_limit"])
        return float(self.config["leg_kp"]), float(self.config["leg_kd"]), float(self.config["leg_torque_limit"])
