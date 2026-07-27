#!/usr/bin/env python3
"""Small contact-force checks shared by Pinocchio validation scripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContactStatus:
    name: str
    active: bool
    normal_force: float
    tangential_force: float
    friction_ratio: float
    mode: str


def vector_norm(vector: np.ndarray) -> float:
    return float(np.sqrt(sum(value * value for value in vector)))


def classify_support_contacts(
    support_forces: np.ndarray,
    support_names: tuple[str, ...],
    active_mask: tuple[bool, ...],
    ground_normal: np.ndarray,
    friction_coefficient: float,
    tolerance: float = 1e-8,
) -> tuple[ContactStatus, ...]:
    """Classify active 3D point contacts as sticking, slipping, or lifted."""

    if support_forces.shape != (3 * len(support_names),):
        raise ValueError("support_forces must contain one 3D force per contact")
    if len(active_mask) != len(support_names):
        raise ValueError("active_mask and support_names must have equal length")
    if friction_coefficient < 0.0:
        raise ValueError("friction_coefficient must be nonnegative")
    normal = np.asarray(ground_normal, dtype=float)
    normal_length = vector_norm(normal)
    if normal_length <= tolerance:
        raise ValueError("ground_normal must be nonzero")
    normal = normal / normal_length

    statuses = []
    for index, name in enumerate(support_names):
        force = support_forces[3 * index : 3 * index + 3]
        normal_force = float(np.dot(force, normal))
        tangential = force - normal_force * normal
        tangential_force = vector_norm(tangential)
        denominator = max(normal_force, tolerance)
        friction_ratio = tangential_force / denominator
        if not active_mask[index]:
            mode = "inactive"
        elif normal_force < -tolerance:
            mode = "lift_off"
        elif tangential_force > friction_coefficient * max(normal_force, 0.0) + tolerance:
            mode = "slip"
        else:
            mode = "stick"
        statuses.append(
            ContactStatus(
                name=name,
                active=bool(active_mask[index]),
                normal_force=normal_force,
                tangential_force=tangential_force,
                friction_ratio=friction_ratio,
                mode=mode,
            )
        )
    return tuple(statuses)
