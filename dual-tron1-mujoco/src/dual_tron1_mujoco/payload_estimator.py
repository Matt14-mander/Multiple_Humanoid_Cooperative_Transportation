"""Object-level payload mass/center-of-mass identification.

The estimator deliberately treats the carried object as an independent rigid
body.  Fixed gripper/tool inertias belong in each robot model; the payload does
not.  A short, low-dynamic identification motion supplies world-frame contact
wrenches and object poses.  The estimator then identifies

    theta = [m, m*c_x, m*c_y, m*c_z]

with a robust sliding-window least-squares fit.  Once the estimate is
observable and consistent it is frozen for transport.  Frozen estimates are
monitored, but are never silently adapted during walking.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


CALIBRATING = "calibrating"
FROZEN = "frozen"
REIDENTIFICATION_REQUIRED = "reidentification_required"


def _vector(values: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=float)
    if result.shape != (3,):
        raise ValueError(name + " must contain three values")
    return result


def skew(vector: np.ndarray) -> np.ndarray:
    """Return the matrix ``[vector]x`` used for cross products."""
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def compose_object_wrench(
    contact_wrenches: np.ndarray,
    contact_points_world: np.ndarray,
    object_origin_world: np.ndarray,
) -> np.ndarray:
    """Compose contact wrenches into one wrench about the object origin."""
    wrenches = np.asarray(contact_wrenches, dtype=float)
    points = np.asarray(contact_points_world, dtype=float)
    origin = np.asarray(object_origin_world, dtype=float)
    if wrenches.ndim != 2 or wrenches.shape[1] != 6:
        raise ValueError("contact_wrenches must have shape (n, 6)")
    if points.shape != (wrenches.shape[0], 3):
        raise ValueError("contact_points_world must have shape (n, 3)")
    if origin.shape != (3,):
        raise ValueError("object_origin_world must contain three values")
    force = np.sum(wrenches[:, :3], axis=0)
    moment = np.sum(
        wrenches[:, 3:]
        + np.cross(points - origin, wrenches[:, :3]),
        axis=0,
    )
    return np.concatenate((force, moment))


def payload_regressor(
    rotation_world_from_body: np.ndarray,
    linear_acceleration_world: np.ndarray,
    gravity_world: np.ndarray,
) -> np.ndarray:
    """Build the low-dynamic regressor for ``[m, m*c_body]``.

    With ``u = a_origin - g`` and negligible angular-inertia terms,

        F = m*u
        tau_origin = (R*c) x (m*u) = -[u]x R (m*c).

    Multiple safe pitch/roll poses or translational excitation are required to
    make all center-of-mass components observable.
    """
    rotation = np.asarray(rotation_world_from_body, dtype=float)
    acceleration = np.asarray(linear_acceleration_world, dtype=float)
    gravity = np.asarray(gravity_world, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation_world_from_body must have shape (3, 3)")
    if acceleration.shape != (3,) or gravity.shape != (3,):
        raise ValueError("acceleration and gravity must contain three values")
    specific_force = acceleration - gravity
    regressor = np.zeros((6, 4))
    regressor[:3, 0] = specific_force
    regressor[3:, 1:] = -skew(specific_force) @ rotation
    return regressor


@dataclass(frozen=True)
class PayloadEstimatorConfig:
    window_size: int = 400
    minimum_samples: int = 20
    mass_bounds_kg: Tuple[float, float] = (0.1, 20.0)
    com_min_m: Tuple[float, float, float] = (-0.30, -0.20, -0.20)
    com_max_m: Tuple[float, float, float] = (0.30, 0.20, 0.20)
    ridge: float = 1e-8
    huber_delta: float = 0.5
    maximum_condition_number: float = 1e4
    maximum_residual_rms: float = 0.75
    innovation_threshold: float = 2.0
    innovation_consecutive_samples: int = 20


@dataclass(frozen=True)
class PayloadEstimate:
    mass_kg: float
    com_body_m: np.ndarray
    first_moment_kg_m: np.ndarray
    sample_count: int
    observable_rank: int
    condition_number: float
    residual_rms: float
    ready: bool
    state: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "mass_kg": self.mass_kg,
            "com_body_m": self.com_body_m.tolist(),
            "first_moment_kg_m": self.first_moment_kg_m.tolist(),
            "sample_count": self.sample_count,
            "observable_rank": self.observable_rank,
            "condition_number": self.condition_number,
            "residual_rms": self.residual_rms,
            "ready": self.ready,
            "state": self.state,
        }


@dataclass(frozen=True)
class _Sample:
    regressor: np.ndarray
    wrench: np.ndarray
    weight: float


class WindowedPayloadEstimator:
    """Robust, projected sliding-window estimator with explicit freezing."""

    def __init__(
        self,
        config: PayloadEstimatorConfig = PayloadEstimatorConfig(),
        prior_mass_kg: float = 1.0,
        prior_com_body_m: Sequence[float] = (0.0, 0.0, 0.0),
    ):
        if config.window_size < 4:
            raise ValueError("window_size must be at least four")
        if config.minimum_samples < 1:
            raise ValueError("minimum_samples must be positive")
        if config.mass_bounds_kg[0] <= 0.0:
            raise ValueError("minimum payload mass must be positive")
        if config.mass_bounds_kg[0] >= config.mass_bounds_kg[1]:
            raise ValueError("mass bounds must be increasing")
        self.config = config
        self.com_min = _vector(config.com_min_m, "com_min_m")
        self.com_max = _vector(config.com_max_m, "com_max_m")
        if np.any(self.com_min >= self.com_max):
            raise ValueError("COM bounds must be increasing")
        prior_com = _vector(prior_com_body_m, "prior_com_body_m")
        self.prior = np.concatenate(
            ([float(prior_mass_kg)], float(prior_mass_kg) * prior_com)
        )
        self.samples: List[_Sample] = []
        self.theta = self._project(self.prior)
        self.state = CALIBRATING
        self._innovation_count = 0
        self._estimate = self._make_estimate(0, np.inf, np.inf, False)

    def _project(self, theta: np.ndarray) -> np.ndarray:
        projected = np.asarray(theta, dtype=float).copy()
        mass = float(
            np.clip(projected[0], *self.config.mass_bounds_kg)
        )
        com = np.clip(projected[1:] / max(abs(projected[0]), 1e-9), self.com_min, self.com_max)
        return np.concatenate(([mass], mass * com))

    def _make_estimate(
        self,
        rank: int,
        condition_number: float,
        residual_rms: float,
        ready: bool,
    ) -> PayloadEstimate:
        mass = float(self.theta[0])
        first_moment = self.theta[1:].copy()
        return PayloadEstimate(
            mass_kg=mass,
            com_body_m=first_moment / mass,
            first_moment_kg_m=first_moment,
            sample_count=len(self.samples),
            observable_rank=rank,
            condition_number=float(condition_number),
            residual_rms=float(residual_rms),
            ready=bool(ready),
            state=self.state,
        )

    @property
    def estimate(self) -> PayloadEstimate:
        return self._estimate

    def start_calibration(self, clear_window: bool = True) -> None:
        if clear_window:
            self.samples = []
        self.state = CALIBRATING
        self._innovation_count = 0
        self._estimate = self._make_estimate(0, np.inf, np.inf, False)

    def add_sample(
        self,
        contact_wrenches: np.ndarray,
        contact_points_world: np.ndarray,
        object_origin_world: np.ndarray,
        rotation_world_from_body: np.ndarray,
        linear_acceleration_world: np.ndarray,
        gravity_world: np.ndarray,
        weight: float = 1.0,
    ) -> PayloadEstimate:
        """Add one identification sample while in the calibration state."""
        if self.state != CALIBRATING:
            return self.monitor(
                contact_wrenches,
                contact_points_world,
                object_origin_world,
                rotation_world_from_body,
                linear_acceleration_world,
                gravity_world,
            )
        if weight <= 0.0:
            raise ValueError("sample weight must be positive")
        regressor = payload_regressor(
            rotation_world_from_body,
            linear_acceleration_world,
            gravity_world,
        )
        wrench = compose_object_wrench(
            contact_wrenches,
            contact_points_world,
            object_origin_world,
        )
        self.samples.append(_Sample(regressor, wrench, float(weight)))
        if len(self.samples) > self.config.window_size:
            self.samples.pop(0)
        self._fit()
        return self._estimate

    def _fit(self) -> None:
        regressors = np.vstack([sample.regressor for sample in self.samples])
        wrenches = np.concatenate([sample.wrench for sample in self.samples])
        base_row_weights = np.repeat(
            [sample.weight for sample in self.samples], 6
        )
        singular_values = np.linalg.svd(
            np.sqrt(base_row_weights)[:, None] * regressors,
            compute_uv=False,
        )
        tolerance = max(regressors.shape) * np.finfo(float).eps * singular_values[0]
        rank = int(np.sum(singular_values > tolerance))
        condition_number = (
            float(singular_values[0] / singular_values[-1])
            if rank == 4 and singular_values[-1] > 0.0
            else np.inf
        )

        robust_sample_weights = np.ones(len(self.samples))
        theta = self.theta.copy()
        ridge_root = np.sqrt(max(self.config.ridge, 0.0))
        for _ in range(3):
            row_weights = base_row_weights * np.repeat(robust_sample_weights, 6)
            weighted_regressor = np.sqrt(row_weights)[:, None] * regressors
            weighted_wrench = np.sqrt(row_weights) * wrenches
            if ridge_root > 0.0:
                weighted_regressor = np.vstack(
                    (weighted_regressor, ridge_root * np.eye(4))
                )
                weighted_wrench = np.concatenate(
                    (weighted_wrench, ridge_root * self.prior)
                )
            theta = np.linalg.lstsq(
                weighted_regressor, weighted_wrench, rcond=None
            )[0]
            theta = self._project(theta)
            sample_residuals = np.array(
                [
                    np.linalg.norm(sample.regressor @ theta - sample.wrench)
                    for sample in self.samples
                ]
            )
            robust_sample_weights = np.minimum(
                1.0,
                self.config.huber_delta / np.maximum(sample_residuals, 1e-12),
            )

        self.theta = theta
        residual = regressors @ theta - wrenches
        residual_rms = float(np.sqrt(np.mean(np.square(residual))))
        ready = (
            len(self.samples) >= self.config.minimum_samples
            and rank == 4
            and condition_number <= self.config.maximum_condition_number
            and residual_rms <= self.config.maximum_residual_rms
        )
        self._estimate = self._make_estimate(
            rank, condition_number, residual_rms, ready
        )

    def freeze_if_ready(self) -> bool:
        if not self._estimate.ready:
            return False
        self.state = FROZEN
        self._innovation_count = 0
        self._estimate = self._make_estimate(
            self._estimate.observable_rank,
            self._estimate.condition_number,
            self._estimate.residual_rms,
            True,
        )
        return True

    def monitor(
        self,
        contact_wrenches: np.ndarray,
        contact_points_world: np.ndarray,
        object_origin_world: np.ndarray,
        rotation_world_from_body: np.ndarray,
        linear_acceleration_world: np.ndarray,
        gravity_world: np.ndarray,
    ) -> PayloadEstimate:
        """Check a frozen estimate without adapting it."""
        if self.state == CALIBRATING:
            return self._estimate
        regressor = payload_regressor(
            rotation_world_from_body,
            linear_acceleration_world,
            gravity_world,
        )
        wrench = compose_object_wrench(
            contact_wrenches,
            contact_points_world,
            object_origin_world,
        )
        innovation = float(np.linalg.norm(wrench - regressor @ self.theta))
        if innovation > self.config.innovation_threshold:
            self._innovation_count += 1
        else:
            self._innovation_count = 0
        if self._innovation_count >= self.config.innovation_consecutive_samples:
            self.state = REIDENTIFICATION_REQUIRED
        self._estimate = self._make_estimate(
            self._estimate.observable_rank,
            self._estimate.condition_number,
            self._estimate.residual_rms,
            self._estimate.ready,
        )
        return self._estimate
