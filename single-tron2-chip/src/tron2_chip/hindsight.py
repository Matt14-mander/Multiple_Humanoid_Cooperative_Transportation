"""CHIP hindsight-goal transformation and deployment history buffers."""

from collections import deque

import numpy as np


def compliance_matrix(compliance_m_per_n) -> np.ndarray:
    value = np.asarray(compliance_m_per_n, dtype=float)
    if value.ndim == 0:
        if value < 0.0:
            raise ValueError("compliance must be non-negative")
        return np.eye(3) * float(value)
    if value.shape == (3,):
        if np.any(value < 0.0):
            raise ValueError("compliance must be non-negative")
        return np.diag(value)
    if value.shape == (3, 3):
        if not np.allclose(value, value.T):
            raise ValueError("compliance matrix must be symmetric")
        if np.min(np.linalg.eigvalsh(value)) < -1e-12:
            raise ValueError("compliance matrix must be positive semidefinite")
        return value
    raise ValueError("compliance must be a scalar, length-3 vector or 3x3 matrix")


def hindsight_goal(reference_position, force_world_n, compliance_m_per_n):
    """Return g_hind = g_ref - C f, all expressed in the world frame."""
    reference = np.asarray(reference_position, dtype=float)
    force = np.asarray(force_world_n, dtype=float)
    if reference.shape != (3,) or force.shape != (3,):
        raise ValueError("reference position and force must have shape (3,)")
    return reference - compliance_matrix(compliance_m_per_n) @ force


class HistoryBuffer:
    """Fixed-size oldest-to-newest observation history for a future actor."""

    def __init__(self, steps: int, sample_shape):
        if steps < 1:
            raise ValueError("steps must be positive")
        self.steps = int(steps)
        self.sample_shape = tuple(sample_shape)
        self._samples = deque(maxlen=self.steps)

    def reset(self, sample):
        sample = self._validated(sample)
        self._samples.clear()
        self._samples.extend(sample.copy() for _ in range(self.steps))

    def append(self, sample):
        sample = self._validated(sample)
        if not self._samples:
            self.reset(sample)
        else:
            self._samples.append(sample.copy())

    def array(self):
        if len(self._samples) != self.steps:
            raise RuntimeError("history buffer has not been initialized")
        return np.stack(self._samples, axis=0)

    def _validated(self, sample):
        value = np.asarray(sample, dtype=float)
        if value.shape != self.sample_shape:
            raise ValueError("unexpected sample shape: {}".format(value.shape))
        return value

