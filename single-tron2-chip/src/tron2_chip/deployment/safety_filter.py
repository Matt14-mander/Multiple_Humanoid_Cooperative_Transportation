"""Final normalized-action magnitude and slew-rate guard."""

import numpy as np


class ActionSafetyFilter:
    def __init__(self, action_size, magnitude_limit=1.0, rate_limit_per_s=10.0):
        self.magnitude_limit = np.broadcast_to(
            np.asarray(magnitude_limit, dtype=float), (action_size,)
        ).copy()
        self.rate_limit = np.broadcast_to(
            np.asarray(rate_limit_per_s, dtype=float), (action_size,)
        ).copy()
        if np.any(self.magnitude_limit <= 0.0) or np.any(self.rate_limit <= 0.0):
            raise ValueError("safety limits must be positive")
        self.previous = np.zeros(action_size, dtype=float)

    def reset(self, action=None):
        self.previous[:] = 0.0 if action is None else action

    def filter(self, action, dt_s):
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        clipped = np.clip(np.asarray(action, dtype=float), -self.magnitude_limit, self.magnitude_limit)
        delta_limit = self.rate_limit * float(dt_s)
        result = self.previous + np.clip(clipped - self.previous, -delta_limit, delta_limit)
        self.previous = result.copy()
        return result

