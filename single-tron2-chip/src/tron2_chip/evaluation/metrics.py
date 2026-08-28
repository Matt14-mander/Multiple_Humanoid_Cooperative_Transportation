"""Metrics for a deployment-mode force pulse."""

from pathlib import Path

import numpy as np

from ..plot_csv import load_rollout_csv


AXES = ("x", "y", "z")


def compliance_metrics(csv_path: Path, commanded_compliance: float, axis: str,
                       steady_fraction=0.25, recovery_tolerance_m=0.005):
    if axis not in AXES:
        raise ValueError("axis must be x, y or z")
    data = load_rollout_csv(csv_path)
    time_s = data["time_s"]
    force = data["force_{}_n".format(axis)]
    displacement = data["ee_{}".format(axis)] - data["ref_{}".format(axis)]
    active_indices = np.flatnonzero(np.abs(force) > 1e-9)
    if not active_indices.size:
        raise ValueError("rollout has no active force samples on axis " + axis)
    steady_count = max(1, int(np.ceil(active_indices.size * float(steady_fraction))))
    steady_indices = active_indices[-steady_count:]
    steady_force = float(np.mean(force[steady_indices]))
    steady_displacement = float(np.mean(displacement[steady_indices]))
    effective_compliance = steady_displacement / steady_force
    relative_error = (
        abs(effective_compliance - commanded_compliance) / commanded_compliance
        if commanded_compliance > 0.0 else float("nan")
    )
    end_index = int(active_indices[-1])
    recovery_time = float("nan")
    for index in range(end_index + 1, len(time_s)):
        if np.all(np.abs(displacement[index:]) <= recovery_tolerance_m):
            recovery_time = float(time_s[index] - time_s[end_index])
            break
    result = {
        "axis": axis,
        "commanded_compliance_m_per_n": float(commanded_compliance),
        "steady_force_n": steady_force,
        "steady_displacement_m": steady_displacement,
        "effective_compliance_m_per_n": effective_compliance,
        "relative_error": relative_error,
        "peak_displacement_m": float(np.max(np.abs(displacement[active_indices]))),
        "final_error_m": float(abs(displacement[-1])),
        "recovery_time_s": recovery_time,
        "peak_control_fraction": float(np.max(data.get("max_control_fraction", np.array([np.nan])))),
    }
    return result

