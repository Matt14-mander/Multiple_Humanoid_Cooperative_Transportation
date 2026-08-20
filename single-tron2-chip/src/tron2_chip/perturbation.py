"""External-force schedules used by hindsight-perturbation training."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForcePulse:
    force_world_n: np.ndarray
    start_s: float
    duration_s: float

    def __post_init__(self):
        force = np.asarray(self.force_world_n, dtype=float)
        if force.shape != (3,):
            raise ValueError("force_world_n must have shape (3,)")
        if self.start_s < 0.0 or self.duration_s <= 0.0:
            raise ValueError("pulse timing must be non-negative with positive duration")
        object.__setattr__(self, "force_world_n", force)

    def at(self, time_s: float) -> np.ndarray:
        if self.start_s <= time_s < self.start_s + self.duration_s:
            return self.force_world_n.copy()
        return np.zeros(3, dtype=float)


def apply_body_force(data, body_id: int, force_world_n):
    data.xfrc_applied[body_id, :] = 0.0
    data.xfrc_applied[body_id, :3] = np.asarray(force_world_n, dtype=float)

