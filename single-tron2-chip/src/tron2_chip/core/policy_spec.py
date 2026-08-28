"""Versioned observation/action contract shared by all backends."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class PolicySpec:
    version: str
    joint_names: tuple[str, ...]
    proprioception_names: tuple[str, ...]
    goal_names: tuple[str, ...]
    action_names: tuple[str, ...]
    history_steps: int
    control_dt_s: float
    force_frame: str = "world"
    compliance_unit: str = "m/N"
    position_unit: str = "rad"
    velocity_unit: str = "rad/s"

    def __post_init__(self):
        if self.history_steps < 1 or self.control_dt_s <= 0.0:
            raise ValueError("history_steps and control_dt_s must be positive")
        if self.force_frame not in {"world", "base", "end_effector"}:
            raise ValueError("unsupported force frame: " + self.force_frame)
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint names must be unique")
        if len(self.action_names) != len(self.joint_names):
            raise ValueError("fixed-base action count must match controlled joints")

    @property
    def action_size(self) -> int:
        return len(self.action_names)

    @property
    def actor_observation_size(self) -> int:
        return self.history_steps * (
            len(self.proprioception_names) + self.action_size
        ) + len(self.goal_names)

    def to_dict(self):
        result = asdict(self)
        for name in (
            "joint_names", "proprioception_names", "goal_names", "action_names"
        ):
            result[name] = list(result[name])
        result["sha256"] = self.sha256
        return result

    @property
    def sha256(self) -> str:
        payload = asdict(self)
        canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        expected_hash = payload.pop("sha256", None)
        for name in (
            "joint_names", "proprioception_names", "goal_names", "action_names"
        ):
            payload[name] = tuple(payload[name])
        spec = cls(**payload)
        if expected_hash is not None and expected_hash != spec.sha256:
            raise ValueError("PolicySpec hash mismatch")
        return spec

    @classmethod
    def fixed_base_arm(cls, history_steps=10, control_dt_s=0.02):
        joints = tuple("arm{}_Joint".format(i) for i in range(1, 7))
        proprioception = (
            tuple("q_arm{}".format(i) for i in range(1, 7))
            + tuple("dq_arm{}".format(i) for i in range(1, 7))
            + ("ee_x", "ee_y", "ee_z", "ee_vx", "ee_vy", "ee_vz")
        )
        return cls(
            version="tron2-chip-fixed-arm-v1",
            joint_names=joints,
            proprioception_names=proprioception,
            goal_names=("goal_x", "goal_y", "goal_z", "compliance_x", "compliance_y", "compliance_z"),
            action_names=tuple("delta_q_arm{}".format(i) for i in range(1, 7)),
            history_steps=int(history_steps),
            control_dt_s=float(control_dt_s),
        )

