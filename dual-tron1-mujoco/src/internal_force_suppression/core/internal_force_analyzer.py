"""
Internal Force Analyzer - Decompose forces into effective and internal components.

Based on:
    Erhart, S., & Hirche, S. (2015).
    "Internal Force Analysis and Load Distribution for Cooperative Multi-Robot Manipulation."
    IEEE International Conference on Robotics and Automation (ICRA).

For dual-robot cooperative manipulation, the total force can be decomposed into:
    - Effective force: contributes to object motion
    - Internal force: closed-chain constraint forces (squeezing/pulling)

The goal is to minimize internal forces while maintaining stable grasp.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class ContactWrench:
    """Contact wrench at a grasp point."""
    force: np.ndarray  # [3] - force vector
    moment: np.ndarray  # [3] - moment vector
    contact_point: np.ndarray  # [3] - contact point in world frame

    @property
    def wrench(self) -> np.ndarray:
        """Return 6D wrench [force; moment]."""
        return np.concatenate([self.force, self.moment])


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Create skew-symmetric matrix from 3D vector.

    For vector v = [x, y, z], returns:
        [  0  -z   y ]
        [  z   0  -x ]
        [ -y   x   0 ]

    Used for cross product: skew(v) @ w = v × w
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


class InternalForceDecomposition:
    """
    Force decomposition for dual-robot cooperative manipulation.

    Theory (Erhart & Hirche 2015):
        For two robots grasping an object:
        - Grasp matrix G maps contact wrenches to object wrench
        - W_obj = G @ F_contact, where F_contact = [F1; F2]
        - Effective force: F_eff = G^+ @ W_obj (minimum norm solution)
        - Internal force: F_int = F_contact - F_eff (null space component)

        The internal force lies in the null space of G:
            G @ F_int = 0  (doesn't affect object motion)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize force decomposition.

        Args:
            config: Optional configuration dict
        """
        self.config = config or {}

    def compute_grasp_matrix(self,
                            grasp_points: List[np.ndarray],
                            object_com: np.ndarray) -> np.ndarray:
        """
        Compute grasp matrix G that maps contact wrenches to object wrench.

        For n contact points, G is 6×6n matrix:
            W_obj [6×1] = G [6×6n] @ F_contacts [6n×1]

        Each contact contributes a 6×6 block:
            [  I    0  ]
            [ [r]×  I  ]
        where [r]× is the skew-symmetric matrix of r = grasp_point - COM

        Args:
            grasp_points: List of contact points in world frame [3] each
            object_com: Object center of mass [3]

        Returns:
            G: Grasp matrix [6, 6*n_contacts]

        Reference: Erhart & Hirche (2015), Equation (3)
        """
        n_contacts = len(grasp_points)
        G = np.zeros((6, 6 * n_contacts))

        for i, point in enumerate(grasp_points):
            # Relative position from COM to contact point
            r = point - object_com

            # Grasp matrix block for this contact
            # Top block: force directly contributes to resultant force
            G[0:3, 6*i:6*i+3] = np.eye(3)

            # Bottom block: force contributes to moment via r × F
            G[3:6, 6*i:6*i+3] = skew_symmetric(r)

            # Moment directly contributes to resultant moment
            G[3:6, 6*i+3:6*i+6] = np.eye(3)

        return G

    def decompose_forces(self,
                        contact_wrenches: List[ContactWrench],
                        object_com: np.ndarray,
                        object_mass: Optional[float] = None) -> Dict[str, Any]:
        """
        Decompose contact forces into effective and internal components.

        Args:
            contact_wrenches: List of contact wrenches (one per robot)
            object_com: Object center of mass [3]
            object_mass: Object mass (optional, for normalization)

        Returns:
            Dict containing:
                - 'F_contact': Total contact wrench [6n]
                - 'F_effective': Effective force component [6n]
                - 'F_internal': Internal force component [6n]
                - 'W_object': Object wrench [6]
                - 'internal_magnitude': Scalar magnitude of internal force
                - 'effective_magnitude': Scalar magnitude of effective force
                - 'internal_ratio': ||F_internal|| / ||F_contact||

        Reference: Erhart & Hirche (2015), Section III.B
        """
        # Extract grasp points and concatenate wrenches
        grasp_points = [w.contact_point for w in contact_wrenches]
        F_contact = np.concatenate([w.wrench for w in contact_wrenches])

        # Compute grasp matrix
        G = self.compute_grasp_matrix(grasp_points, object_com)

        # Object wrench (resultant force/moment on object)
        W_obj = G @ F_contact

        # Effective force: minimum norm solution
        # F_eff = G^+ @ W_obj, where G^+ is the Moore-Penrose pseudoinverse
        G_pinv = np.linalg.pinv(G)
        F_effective = G_pinv @ W_obj

        # Internal force: null space component
        # F_int = F_contact - F_eff
        # This satisfies: G @ F_int = 0 (no effect on object)
        F_internal = F_contact - F_effective

        # Compute magnitudes
        internal_mag = np.linalg.norm(F_internal)
        effective_mag = np.linalg.norm(F_effective)
        contact_mag = np.linalg.norm(F_contact)

        # Internal force ratio (0 = no internal force, 1 = all internal)
        internal_ratio = internal_mag / contact_mag if contact_mag > 1e-6 else 0.0

        return {
            'F_contact': F_contact,
            'F_effective': F_effective,
            'F_internal': F_internal,
            'W_object': W_obj,
            'internal_magnitude': internal_mag,
            'effective_magnitude': effective_mag,
            'internal_ratio': internal_ratio,
            'grasp_matrix': G
        }

    def decompose_per_robot(self,
                           decomposition: Dict[str, Any],
                           n_robots: int = 2) -> List[Dict[str, np.ndarray]]:
        """
        Split decomposed forces back into per-robot components.

        Args:
            decomposition: Output from decompose_forces()
            n_robots: Number of robots (default: 2)

        Returns:
            List of dicts, one per robot, each containing:
                - 'F_effective': Effective wrench for this robot [6]
                - 'F_internal': Internal wrench for this robot [6]
        """
        F_effective = decomposition['F_effective']
        F_internal = decomposition['F_internal']

        per_robot = []
        for i in range(n_robots):
            per_robot.append({
                'F_effective': F_effective[6*i:6*i+6],
                'F_internal': F_internal[6*i:6*i+6]
            })

        return per_robot


class InternalForceAnalyzer:
    """
    High-level analyzer for internal force analysis in dual-robot systems.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize analyzer.

        Args:
            config: Configuration dict with keys:
                - max_safe_internal_force: Maximum safe internal force magnitude (N)
                - warning_threshold: Warning threshold (N)
                - decomposition_method: "null_space_projection" (default)
        """
        self.config = config
        self.max_safe_force = config.get('max_safe_internal_force', 50.0)
        self.warning_threshold = config.get('warning_threshold', 35.0)

        self.decomposition = InternalForceDecomposition(config)

        # History for monitoring
        self.internal_force_history = []
        self.max_history_length = 1000

    def analyze(self,
               robot1_wrench: ContactWrench,
               robot2_wrench: ContactWrench,
               object_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze internal forces in dual-robot system.

        Args:
            robot1_wrench: Contact wrench from robot 1
            robot2_wrench: Contact wrench from robot 2
            object_state: Dict containing:
                - 'com': Object center of mass [3]
                - 'mass': Object mass (optional)

        Returns:
            Dict containing:
                - All outputs from decompose_forces()
                - 'per_robot': List of per-robot force components
                - 'safety_status': "safe", "warning", or "danger"
                - 'safety_ratio': Internal force / max_safe_force (0-1+)
        """
        # Decompose forces
        result = self.decomposition.decompose_forces(
            contact_wrenches=[robot1_wrench, robot2_wrench],
            object_com=object_state['com'],
            object_mass=object_state.get('mass', None)
        )

        # Per-robot decomposition
        result['per_robot'] = self.decomposition.decompose_per_robot(result, n_robots=2)

        # Safety analysis
        internal_mag = result['internal_magnitude']
        safety_ratio = internal_mag / self.max_safe_force

        if internal_mag >= self.max_safe_force:
            safety_status = "danger"
        elif internal_mag >= self.warning_threshold:
            safety_status = "warning"
        else:
            safety_status = "safe"

        result['safety_status'] = safety_status
        result['safety_ratio'] = safety_ratio

        # Update history
        self.internal_force_history.append(internal_mag)
        if len(self.internal_force_history) > self.max_history_length:
            self.internal_force_history.pop(0)

        return result

    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistical summary of internal force history.

        Returns:
            Dict with mean, max, std of internal forces
        """
        if not self.internal_force_history:
            return {'mean': 0.0, 'max': 0.0, 'std': 0.0}

        history = np.array(self.internal_force_history)
        return {
            'mean': float(np.mean(history)),
            'max': float(np.max(history)),
            'std': float(np.std(history)),
            'current': float(history[-1]) if len(history) > 0 else 0.0
        }

    def reset_history(self):
        """Reset internal force history."""
        self.internal_force_history = []
