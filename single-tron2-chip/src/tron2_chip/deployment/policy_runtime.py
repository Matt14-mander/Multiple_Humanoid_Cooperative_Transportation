"""Simulator-independent policy inference runtime."""

from pathlib import Path

import numpy as np


class CallablePolicyBackend:
    def __init__(self, function):
        self.function = function

    def infer(self, observation):
        return np.asarray(self.function(np.asarray(observation, dtype=np.float32)), dtype=float)


class OnnxPolicyBackend:
    def __init__(self, model_path: Path, providers=None):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "onnxruntime is required for ONNX deployment; install the deployment extra"
            ) from error
        provider_list = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(Path(model_path)), providers=provider_list)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer(self, observation):
        batch = np.asarray(observation, dtype=np.float32)[None, :]
        return np.asarray(
            self.session.run([self.output_name], {self.input_name: batch})[0][0],
            dtype=float,
        )


class PolicyRuntime:
    def __init__(self, spec, backend, normalizer=None, safety_filter=None):
        self.spec = spec
        self.backend = backend
        self.normalizer = normalizer
        self.safety_filter = safety_filter

    def infer(self, observation, dt_s=None):
        observation = np.asarray(observation, dtype=float)
        if observation.shape != (self.spec.actor_observation_size,):
            raise ValueError("observation does not match PolicySpec")
        policy_input = observation if self.normalizer is None else self.normalizer.transform(observation)
        action = np.asarray(self.backend.infer(policy_input), dtype=float)
        if action.shape != (self.spec.action_size,):
            raise RuntimeError("policy output does not match PolicySpec")
        if self.safety_filter is not None:
            action = self.safety_filter.filter(action, dt_s or self.spec.control_dt_s)
        return action

