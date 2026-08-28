"""Serializable observation normalization for ONNX deployment parity."""

from pathlib import Path

import numpy as np


class ObservationNormalizer:
    def __init__(self, mean, standard_deviation, clip=10.0, epsilon=1e-6):
        self.mean = np.asarray(mean, dtype=float)
        self.standard_deviation = np.asarray(standard_deviation, dtype=float)
        self.clip = float(clip)
        self.epsilon = float(epsilon)
        if self.mean.shape != self.standard_deviation.shape:
            raise ValueError("normalization arrays must have matching shapes")
        if np.any(self.standard_deviation < 0.0):
            raise ValueError("standard deviation cannot be negative")

    def transform(self, observation):
        value = np.asarray(observation, dtype=float)
        if value.shape != self.mean.shape:
            raise ValueError("observation shape does not match normalizer")
        normalized = (value - self.mean) / np.maximum(self.standard_deviation, self.epsilon)
        return np.clip(normalized, -self.clip, self.clip)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, standard_deviation=self.standard_deviation,
                 clip=self.clip, epsilon=self.epsilon)

    @classmethod
    def load(cls, path: Path):
        with np.load(Path(path)) as payload:
            return cls(payload["mean"], payload["standard_deviation"],
                       float(payload["clip"]), float(payload["epsilon"]))

