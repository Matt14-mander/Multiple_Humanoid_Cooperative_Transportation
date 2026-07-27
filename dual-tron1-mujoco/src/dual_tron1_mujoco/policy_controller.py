"""Dual-instance ONNX controller matching the upstream WF action semantics."""

from pathlib import Path

import numpy as np

from .control import ACTION_ORDER
from .model_index import ModelIndex
from .observation import ObservationBuilder


class OnnxPolicyBackend:
    """Stateless, shareable encoder and policy inference sessions."""

    def __init__(self, policy_dir: Path):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("Install the policy optional dependency") from error
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(
            (Path(policy_dir) / "encoder.onnx").read_bytes(),
            sess_options=options,
            providers=providers,
        )
        self.policy = ort.InferenceSession(
            (Path(policy_dir) / "policy.onnx").read_bytes(),
            sess_options=options,
            providers=providers,
        )

    def infer(self, observation, history):
        latent = self.encoder.run(
            ["latent"], {"obs_history": np.asarray(history, np.float32)}
        )[0]
        actions = self.policy.run(
            ["actions"],
            {
                "obs": np.asarray(observation, np.float32).reshape(1, 74),
                "latent": np.asarray(latent, np.float32),
            },
        )[0].reshape(14)
        if not np.all(np.isfinite(actions)):
            raise FloatingPointError("ONNX policy returned non-finite actions")
        return actions.astype(float)


class RobotPolicyController:
    def __init__(self, model, prefix, config, backend: OnnxPolicyBackend):
        self.model = model
        self.prefix = prefix
        self.config = config
        self.backend = backend
        self.observation = ObservationBuilder(model, prefix)
        self.index = ModelIndex(model)
        self.names = [prefix + name for name in ACTION_ORDER]
        self.joints = [self.index.joint(name) for name in self.names]
        self.actuators = [self.index.actuator(name) for name in self.names]
        self.actions = np.zeros(14, dtype=float)
        self.last_actions = np.zeros(14, dtype=float)
        self.decimation = int(config.get("policy_decimation", 10))
        self.action_scale_position = float(
            config.get("action_scale_position", 0.25)
        )
        self.action_scale_velocity = float(
            config.get("action_scale_velocity", 3.0)
        )
        self.update_count = 0

    def _gains(self, action_index):
        name = ACTION_ORDER[action_index]
        if name.startswith("wheel_"):
            return 0.0, float(self.config["wheel_kd"]), float(
                self.config["wheel_torque_limit"]
            )
        if name in {"J1", "J2", "J3"}:
            # The C++ controller doubles damping and halves stiffness when the
            # commanded end-effector pose is stationary.
            return (
                float(self.config["arm_j123_kp"]) / 2.0,
                float(self.config["arm_j123_kd"]) * 2.0,
                float(self.config["arm_j123_torque_limit"]),
            )
        if name in {"J4", "J5", "J6"}:
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

    def update(self, data, command, actuate_arms=True):
        if self.update_count % self.decimation == 0:
            obs = self.observation.build(data, command, self.last_actions)
            history = self.observation.push_history(obs)
            self.actions = self.backend.infer(obs, history)

        for action_index, (joint, actuator) in enumerate(
            zip(self.joints, self.actuators)
        ):
            q = float(data.qpos[joint.qpos_adr])
            dq = float(data.qvel[joint.dof_adr])
            kp, kd, torque_limit = self._gains(action_index)
            raw_action = float(self.actions[action_index])
            if ACTION_ORDER[action_index].startswith("wheel_"):
                lower = (
                    dq - torque_limit / kd
                ) / self.action_scale_velocity
                upper = (
                    dq + torque_limit / kd
                ) / self.action_scale_velocity
                action = float(np.clip(raw_action, lower, upper))
                desired_velocity = action * self.action_scale_velocity
                torque = kd * (desired_velocity - dq)
            else:
                lower = (q + (kd * dq - torque_limit) / kp) / self.action_scale_position
                upper = (q + (kd * dq + torque_limit) / kp) / self.action_scale_position
                action = float(np.clip(raw_action, lower, upper))
                desired_position = action * self.action_scale_position
                torque = kp * (desired_position - q) - kd * dq
            if actuate_arms or ACTION_ORDER[action_index] not in {
                "J1", "J2", "J3", "J4", "J5", "J6"
            }:
                data.ctrl[actuator] = np.clip(
                    torque, -torque_limit, torque_limit
                )
            self.last_actions[action_index] = action
        self.update_count += 1
