"""Normalized policy-action conversion shared by simulation and hardware."""

import numpy as np


class ActionScaler:
    def __init__(self, scale_rad, lower_rad=None, upper_rad=None):
        self.scale = np.asarray(scale_rad, dtype=float)
        if self.scale.ndim != 1 or np.any(self.scale <= 0.0):
            raise ValueError("scale_rad must be a positive vector")
        self.lower = None if lower_rad is None else np.asarray(lower_rad, dtype=float)
        self.upper = None if upper_rad is None else np.asarray(upper_rad, dtype=float)
        if (self.lower is None) != (self.upper is None):
            raise ValueError("lower and upper limits must be provided together")
        if self.lower is not None and (
            self.lower.shape != self.scale.shape or self.upper.shape != self.scale.shape
        ):
            raise ValueError("action limits must match scale shape")

    def residual(self, normalized_action):
        action = np.asarray(normalized_action, dtype=float)
        if action.shape != self.scale.shape:
            raise ValueError("unexpected action shape")
        return np.clip(action, -1.0, 1.0) * self.scale

    def position_target(self, nominal_position, normalized_action):
        target = np.asarray(nominal_position, dtype=float) + self.residual(normalized_action)
        if self.lower is not None:
            target = np.clip(target, self.lower, self.upper)
        return target

