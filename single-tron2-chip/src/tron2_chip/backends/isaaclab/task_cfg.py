"""Dependency-light task contract mirrored by the future Isaac Lab managers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FixedBaseChipTaskCfg:
    num_envs: int = 4096
    simulation_dt_s: float = 0.002
    control_decimation: int = 10
    episode_length_s: float = 10.0
    history_steps: int = 10
    force_min_n: float = 0.0
    force_max_n: float = 15.0
    force_duration_min_s: float = 0.1
    force_duration_max_s: float = 1.0
    compliance_min_m_per_n: float = 0.001
    compliance_max_m_per_n: float = 0.004

    @property
    def control_dt_s(self):
        return self.simulation_dt_s * self.control_decimation

    def validate(self):
        if self.num_envs < 1 or self.simulation_dt_s <= 0.0 or self.control_decimation < 1:
            raise ValueError("invalid environment timing or count")
        if not 0.0 <= self.force_min_n <= self.force_max_n:
            raise ValueError("invalid force range")
        if not 0.0 < self.force_duration_min_s <= self.force_duration_max_s:
            raise ValueError("invalid force-duration range")
        if not 0.0 < self.compliance_min_m_per_n <= self.compliance_max_m_per_n:
            raise ValueError("invalid compliance range")


def require_isaaclab():
    try:
        import isaaclab  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Isaac Lab backend is unavailable. Use the pinned Ubuntu Isaac Lab "
            "environment; do not install it into croco_env."
        ) from error

