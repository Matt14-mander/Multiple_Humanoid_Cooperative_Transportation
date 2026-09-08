"""Versioned input/output contract for a PAINT-style intent estimator."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class IntentEstimatorSpec:
    version: str
    joint_names: tuple[str, ...]
    history_steps: int = 4
    control_dt_s: float = 0.02
    output_names: tuple[str, ...] = ("force_x", "force_y", "torque_yaw")
    wrench_frame: str = "base_yaw"
    wrench_reference: str = "payload_origin"
    feature_layout: str = "q_history,dq_history,previous_action_history"
    position_unit: str = "rad"
    velocity_unit: str = "rad/s"
    force_unit: str = "N"
    torque_unit: str = "N*m"

    def __post_init__(self):
        if not self.version:
            raise ValueError("version must not be empty")
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        if self.history_steps < 1 or self.control_dt_s <= 0.0:
            raise ValueError("history_steps and control_dt_s must be positive")
        if len(self.output_names) != 3:
            raise ValueError("the PAINT baseline requires [Fx, Fy, Mz]")
        if self.wrench_frame not in {"world", "base", "base_yaw", "payload"}:
            raise ValueError("unsupported wrench frame: " + self.wrench_frame)
        if self.feature_layout != "q_history,dq_history,previous_action_history":
            raise ValueError("unsupported feature layout")

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)

    @property
    def sample_size(self) -> int:
        return 3 * self.joint_count

    @property
    def input_size(self) -> int:
        return self.history_steps * self.sample_size

    @property
    def output_size(self) -> int:
        return len(self.output_names)

    @property
    def history_duration_s(self) -> float:
        return self.history_steps * self.control_dt_s

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            asdict(self), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        result = asdict(self)
        result["joint_names"] = list(self.joint_names)
        result["output_names"] = list(self.output_names)
        result["sha256"] = self.sha256
        return result

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        expected_hash = payload.pop("sha256", None)
        payload["joint_names"] = tuple(payload["joint_names"])
        payload["output_names"] = tuple(payload["output_names"])
        spec = cls(**payload)
        if expected_hash is not None and expected_hash != spec.sha256:
            raise ValueError("IntentEstimatorSpec hash mismatch")
        return spec

    @classmethod
    def airbot_planar(cls, history_steps=4, control_dt_s=0.02):
        return cls(
            version="tron2-airbot-paint-planar-v1",
            joint_names=tuple("arm{}_Joint".format(i) for i in range(1, 7)),
            history_steps=int(history_steps),
            control_dt_s=float(control_dt_s),
        )

