"""Minimal protocol implemented later by MuJoCo and TRON2 SDK adapters."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RobotState:
    time_s: float
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    base_quaternion_wxyz: np.ndarray
    base_angular_velocity: np.ndarray


@dataclass(frozen=True)
class RobotCommand:
    joint_position_target: np.ndarray


class RobotInterface(Protocol):
    def read_state(self) -> RobotState: ...
    def write_command(self, command: RobotCommand) -> None: ...
    def emergency_stop(self) -> None: ...

