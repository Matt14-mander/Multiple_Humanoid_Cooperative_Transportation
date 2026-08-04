"""Deterministic validation of payload mass/COM identification profiles."""

import argparse
import json
from typing import Dict

import numpy as np

from .payload_estimator import (
    PayloadEstimatorConfig,
    WindowedPayloadEstimator,
    payload_regressor,
    skew,
)


PROFILES = {
    "dumbbell": (2.0, np.array([0.0, 0.0, 0.0])),
    "offset_box": (6.0, np.array([0.08, -0.05, 0.03])),
}


def _rotation(roll: float, pitch: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    return np.array(
        [
            [cp, sp * sr, sp * cr],
            [0.0, cr, -sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _contact_wrenches(
    object_wrench: np.ndarray, contact_points: np.ndarray
) -> np.ndarray:
    grasp_matrix = np.zeros((6, 12))
    for index, point in enumerate(contact_points):
        column = 6 * index
        grasp_matrix[:3, column : column + 3] = np.eye(3)
        grasp_matrix[3:, column : column + 3] = skew(point)
        grasp_matrix[3:, column + 3 : column + 6] = np.eye(3)
    return (np.linalg.pinv(grasp_matrix) @ object_wrench).reshape(2, 6)


def run_profile(name: str, noise_std: float = 0.01) -> Dict[str, object]:
    if name not in PROFILES:
        raise ValueError("Unknown payload profile: " + name)
    true_mass, true_com = PROFILES[name]
    estimator = WindowedPayloadEstimator(
        PayloadEstimatorConfig(minimum_samples=24)
    )
    rng = np.random.default_rng(11)
    gravity = np.array([0.0, 0.0, -9.81])
    theta = np.concatenate(([true_mass], true_mass * true_com))
    handle_offsets = np.array([[0.0, -0.35, 0.0], [0.0, 0.35, 0.0]])
    for index in range(48):
        roll = np.deg2rad(-5.0 + 10.0 * (index % 6) / 5.0)
        pitch = np.deg2rad(-4.0 + 8.0 * ((index // 6) % 6) / 5.0)
        rotation = _rotation(roll, pitch)
        regressor = payload_regressor(rotation, np.zeros(3), gravity)
        contact_points = (rotation @ handle_offsets.T).T
        contacts = _contact_wrenches(regressor @ theta, contact_points)
        contacts += rng.normal(0.0, noise_std, size=contacts.shape)
        estimator.add_sample(
            contacts,
            contact_points,
            np.zeros(3),
            rotation,
            np.zeros(3),
            gravity,
        )
    frozen = estimator.freeze_if_ready()
    estimate = estimator.estimate
    return {
        "profile": name,
        "true_mass_kg": true_mass,
        "estimated_mass_kg": estimate.mass_kg,
        "mass_error_kg": estimate.mass_kg - true_mass,
        "true_com_body_m": true_com.tolist(),
        "estimated_com_body_m": estimate.com_body_m.tolist(),
        "com_error_norm_m": float(
            np.linalg.norm(estimate.com_body_m - true_com)
        ),
        "observable_rank": estimate.observable_rank,
        "condition_number": estimate.condition_number,
        "residual_rms": estimate.residual_rms,
        "frozen": frozen,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("all",) + tuple(PROFILES), default="all"
    )
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    names = PROFILES if args.profile == "all" else (args.profile,)
    results = {
        name: run_profile(name, noise_std=args.noise_std) for name in names
    }
    if args.json:
        print(json.dumps(results, indent=2))
        return
    print("profile       mass_true  mass_est  com_error  rank  cond    frozen")
    for name, result in results.items():
        print(
            "{:<12} {:9.3f} {:9.3f} {:9.4f}m {:4d} {:7.1f} {}".format(
                name,
                result["true_mass_kg"],
                result["estimated_mass_kg"],
                result["com_error_norm_m"],
                result["observable_rank"],
                result["condition_number"],
                "YES" if result["frozen"] else "NO",
            )
        )


if __name__ == "__main__":
    main()
