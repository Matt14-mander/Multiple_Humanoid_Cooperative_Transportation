"""Simulator-independent online inference for the intent estimator."""

from pathlib import Path

import numpy as np

from .history import IntentHistoryBuffer


class CallableIntentBackend:
    def __init__(self, function):
        self.function = function

    def infer(self, features):
        return np.asarray(self.function(np.asarray(features, dtype=np.float32)), dtype=np.float32)


class OnnxIntentBackend:
    def __init__(self, model_path: Path, providers=None):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("install the deployment extra to use ONNX intent inference") from error
        self.session = ort.InferenceSession(
            str(Path(model_path)), providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer(self, features):
        batch = np.asarray(features, dtype=np.float32)[None, :]
        return np.asarray(
            self.session.run([self.output_name], {self.input_name: batch})[0][0],
            dtype=np.float32,
        )


class IntentEstimatorRuntime:
    """Own causal history and return [Fx, Fy, Mz] in declared physical units."""

    def __init__(self, spec, backend, input_normalizer=None, output_normalizer=None, output_limits=None, smoothing=1.0):
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0,1]")
        self.spec = spec
        self.backend = backend
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer
        self.output_limits = None if output_limits is None else np.asarray(output_limits, dtype=np.float32)
        if self.output_limits is not None and (self.output_limits.shape != (3,) or np.any(self.output_limits <= 0.0)):
            raise ValueError("output_limits must be a positive length-3 vector")
        self.smoothing = float(smoothing)
        self.history = IntentHistoryBuffer(spec)
        self._filtered = None

    def reset(self, joint_position, joint_velocity=None, previous_action=None):
        self.history.reset(joint_position, joint_velocity, previous_action)
        self._filtered = None

    def infer(self, joint_position, joint_velocity, previous_action):
        self.history.append(joint_position, joint_velocity, previous_action)
        features = self.history.vector()
        if self.input_normalizer is not None:
            features = self.input_normalizer.transform(features)
        prediction = np.asarray(self.backend.infer(features), dtype=np.float32)
        if prediction.shape != (3,) or not np.all(np.isfinite(prediction)):
            raise RuntimeError("intent backend must return one finite [Fx,Fy,Mz] vector")
        if self.output_normalizer is not None:
            prediction = self.output_normalizer.inverse(prediction)
        if self.output_limits is not None:
            prediction = np.clip(prediction, -self.output_limits, self.output_limits)
        self._filtered = prediction if self._filtered is None else (
            (1.0 - self.smoothing) * self._filtered + self.smoothing * prediction
        )
        return self._filtered.copy()

