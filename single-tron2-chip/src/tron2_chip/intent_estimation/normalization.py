"""Serializable feature/label normalization with NumPy and Torch parity."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray
    standard_deviation: np.ndarray
    clip: float = 10.0
    epsilon: float = 1e-6

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.standard_deviation, dtype=np.float32)
        if mean.ndim != 1 or mean.shape != std.shape:
            raise ValueError("normalization statistics must be matching 1-D arrays")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("normalization statistics must be finite")
        if np.any(std < 0.0) or self.clip <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("invalid standard deviation, clip or epsilon")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "standard_deviation", std)

    @property
    def size(self) -> int:
        return self.mean.size

    @classmethod
    def fit(cls, values, clip=10.0, epsilon=1e-6):
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] < 1:
            raise ValueError("fit values must have shape [samples, features]")
        return cls(array.mean(axis=0), array.std(axis=0), clip, epsilon)

    def transform(self, values):
        array = np.asarray(values, dtype=np.float32)
        self._check_last_dimension(array)
        normalized = (array - self.mean) / np.maximum(self.standard_deviation, self.epsilon)
        return np.clip(normalized, -self.clip, self.clip)

    def inverse(self, values):
        array = np.asarray(values, dtype=np.float32)
        self._check_last_dimension(array)
        return array * np.maximum(self.standard_deviation, self.epsilon) + self.mean

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean,
            standard_deviation=self.standard_deviation,
            clip=np.float32(self.clip),
            epsilon=np.float32(self.epsilon),
        )

    @classmethod
    def load(cls, path: Path):
        with np.load(Path(path)) as payload:
            return cls(
                payload["mean"], payload["standard_deviation"],
                float(payload["clip"]), float(payload["epsilon"]),
            )

    def _check_last_dimension(self, value):
        if value.ndim < 1 or value.shape[-1] != self.size:
            raise ValueError("value does not match normalization statistics")

