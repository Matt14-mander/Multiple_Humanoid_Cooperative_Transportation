"""
Wrench decomposition utilities.

Helper functions for working with 6D wrenches (force + moment vectors).
"""

import numpy as np
from typing import Tuple


def decompose_wrench(wrench: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose 6D wrench into force and moment components.

    Args:
        wrench: 6D wrench vector [fx, fy, fz, mx, my, mz]

    Returns:
        force: Force vector [fx, fy, fz]
        moment: Moment vector [mx, my, mz]
    """
    return wrench[:3], wrench[3:]


def wrench_magnitude(wrench: np.ndarray,
                    force_weight: float = 1.0,
                    moment_weight: float = 1.0) -> float:
    """
    Compute weighted magnitude of a wrench.

    Args:
        wrench: 6D wrench vector [fx, fy, fz, mx, my, mz]
        force_weight: Weight for force component
        moment_weight: Weight for moment component

    Returns:
        Weighted magnitude of the wrench
    """
    force, moment = decompose_wrench(wrench)
    force_mag = np.linalg.norm(force)
    moment_mag = np.linalg.norm(moment)

    return np.sqrt((force_weight * force_mag)**2 + (moment_weight * moment_mag)**2)


def normalize_wrench(wrench: np.ndarray,
                    max_force: float = 1.0,
                    max_moment: float = 1.0) -> np.ndarray:
    """
    Normalize wrench components independently.

    Args:
        wrench: 6D wrench vector
        max_force: Maximum force magnitude
        max_moment: Maximum moment magnitude

    Returns:
        Normalized wrench
    """
    force, moment = decompose_wrench(wrench)

    force_mag = np.linalg.norm(force)
    if force_mag > max_force:
        force = force * (max_force / force_mag)

    moment_mag = np.linalg.norm(moment)
    if moment_mag > max_moment:
        moment = moment * (max_moment / moment_mag)

    return np.concatenate([force, moment])


def transform_wrench(wrench: np.ndarray,
                    rotation: np.ndarray,
                    translation: np.ndarray) -> np.ndarray:
    """
    Transform a wrench from one frame to another.

    Args:
        wrench: 6D wrench in original frame
        rotation: 3×3 rotation matrix (original to new frame)
        translation: 3D translation vector (original to new frame)

    Returns:
        Transformed wrench in new frame
    """
    force, moment = decompose_wrench(wrench)

    # Transform force
    force_new = rotation @ force

    # Transform moment: τ_new = R @ τ + r × (R @ f)
    moment_new = rotation @ moment + np.cross(translation, force_new)

    return np.concatenate([force_new, moment_new])
