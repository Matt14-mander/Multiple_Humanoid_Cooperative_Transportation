"""Deterministic fixed-base joint-hold baseline for WFYG_TRON2A."""

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


def actuator_name(joint_name: str) -> str:
    return joint_name.removesuffix("_Joint")


class JointHoldController:
    def __init__(self, model, data, prefix, config):
        self.model = model
        self.prefix = prefix
        self.config = config
        index = ModelIndex(model)
        self.joints = [index.joint(prefix + name) for name in JOINTS]
        self.actuators = [
            index.actuator(prefix + actuator_name(name)) for name in JOINTS
        ]
        self.q_ref = np.array(
            [data.qpos[joint.qpos_adr] for joint in self.joints], dtype=float
        )

    def _gains(self, name):
        if name.startswith("wheel_"):
            return 0.0, self.config["wheel_kd"], self.config["wheel_torque_limit"]
        if name.startswith("arm"):
            return self.config["arm_kp"], self.config["arm_kd"], self.config["arm_torque_limit"]
        if name.startswith("gripper"):
            return self.config["gripper_kp"], self.config["gripper_kd"], self.config["gripper_force_limit"]
        return self.config["leg_kp"], self.config["leg_kd"], self.config["leg_torque_limit"]

    def update(self, data):
        for number, (name, joint, actuator) in enumerate(
            zip(JOINTS, self.joints, self.actuators)
        ):
            q = float(data.qpos[joint.qpos_adr])
            dq = float(data.qvel[joint.dof_adr])
            kp, kd, limit = [float(value) for value in self._gains(name)]
            gravity_and_velocity_bias = float(data.qfrc_bias[joint.dof_adr])
            data.ctrl[actuator] = np.clip(
                gravity_and_velocity_bias
                + kp * (self.q_ref[number] - q)
                - kd * dq,
                -limit,
                limit,
            )
