"""Physical-unit regression and operating-point metrics."""

from __future__ import annotations

import numpy as np

from .data import COMPONENTS


def regression_metrics(target: np.ndarray, prediction: np.ndarray, classes: np.ndarray,
                       force_threshold_n: float, torque_threshold_nm: float) -> dict:
    error = prediction - target
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    mae = np.mean(np.abs(error), axis=0)
    denom = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - np.sum(error ** 2, axis=0) / np.maximum(denom, 1e-12)
    sign_accuracy = {}
    thresholds = np.asarray([force_threshold_n] * 3 + [torque_threshold_nm] * 3)
    for i, name in enumerate(COMPONENTS):
        mask = np.abs(target[:, i]) > 1e-6
        sign_accuracy[name] = float(np.mean(np.sign(prediction[mask, i]) == np.sign(target[mask, i]))) if mask.any() else None
    zero = classes == 0
    false_positive = np.any(np.abs(prediction[zero]) > thresholds, axis=1)
    return {
        "samples": int(len(target)),
        "rmse": dict(zip(COMPONENTS, map(float, rmse))),
        "mae": dict(zip(COMPONENTS, map(float, mae))),
        "r2": dict(zip(COMPONENTS, map(float, r2))),
        "sign_accuracy": sign_accuracy,
        "focus": {name: {"rmse": float(rmse[i]), "mae": float(mae[i]), "r2": float(r2[i]),
                         "sign_accuracy": sign_accuracy[name]}
                  for i, name in ((0, "Fx"), (1, "Fy"), (5, "Mz"))},
        "zero_false_positive_rate": float(false_positive.mean()) if len(false_positive) else None,
        "zero_thresholds": {"force_n": force_threshold_n, "torque_nm": torque_threshold_nm},
    }

