"""Baseline torque controller used before enabling the ONNX policy."""

from typing import Dict, List

import numpy as np

from .model_index import ModelIndex


ACTION_ORDER = [
    "J1",
    "abad_L_Joint",
    "abad_R_Joint",
    "J2",
    "hip_L_Joint",
    "hip_R_Joint",
    "J3",
    "knee_L_Joint",
    "knee_R_Joint",
    "J4",
    "J5",
    "J6",
    "wheel_L_Joint",
    "wheel_R_Joint",
]


class JointHoldController:
    """Hold the initial articulated pose with deployment-like gains.

    This is intentionally not a locomotion controller.  It is the deterministic
    baseline for model compilation, closed-chain settling and force diagnostics.
    """

    def __init__(self, model, data, prefix: str, config: Dict[str, float]):
        self.model = model
        self.prefix = prefix
        self.index = ModelIndex(model)
        self.config = config
        self.names: List[str] = [prefix + item for item in ACTION_ORDER]
        self.index.require_joints(self.names)
        self.index.require_actuators(self.names)
        self.q_ref = np.array(
            [data.qpos[self.index.joint(name).qpos_adr] for name in self.names],
            dtype=float,
        )

    def _gains(self, short_name: str):
        if short_name.startswith("wheel_"):
            return 0.0, float(self.config["wheel_kd"]), float(
                self.config["wheel_torque_limit"]
            )
        if short_name in {"J1", "J2", "J3"}:
            return (
                float(self.config["arm_j123_kp"]),
                float(self.config["arm_j123_kd"]),
                float(self.config["arm_j123_torque_limit"]),
            )
        if short_name in {"J4", "J5", "J6"}:
            return (
                float(self.config["arm_j456_kp"]),
                float(self.config["arm_j456_kd"]),
                float(self.config["arm_j456_torque_limit"]),
            )
        return (
            float(self.config["leg_kp"]),
            float(self.config["leg_kd"]),
            float(self.config["leg_torque_limit"]),
        )

    def update(self, data) -> None:
        for index, full_name in enumerate(self.names):
            short_name = full_name[len(self.prefix) :]
            address = self.index.joint(full_name)
            actuator_id = self.index.actuator(full_name)
            q = float(data.qpos[address.qpos_adr])
            dq = float(data.qvel[address.dof_adr])
            kp, kd, effort_limit = self._gains(short_name)
            torque = kp * (self.q_ref[index] - q) - kd * dq
            data.ctrl[actuator_id] = np.clip(
                torque, -effort_limit, effort_limit
            )

