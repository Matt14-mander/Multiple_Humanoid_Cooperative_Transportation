"""
Residual Admittance Controller - Compliant control on top of base motion policy.

Based on:
    Hogan, N. (1985). "Impedance Control: An Approach to Manipulation."
    Seraji, H., & Colbaugh, R. (1997). "Force Tracking in Impedance Control."

Admittance control implements a dynamic relationship between force and motion:
    M*ẍ + B*ẋ + K*x = F_ext

where:
    - M: desired inertia
    - B: desired damping
    - K: desired stiffness
    - F_ext: external force (internal force in our case)
    - x: admittance displacement (compliance)

The residual admittance controller computes a corrective motion that makes
the robot compliant to internal forces, thereby reducing them.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AdmittanceParameters:
    """Parameters for admittance dynamics."""
    M: np.ndarray  # Desired inertia [6×6] or [6]
    B: np.ndarray  # Desired damping [6×6] or [6]
    K: np.ndarray  # Desired stiffness [6×6] or [6]

    def __post_init__(self):
        """Ensure diagonal matrices for simplicity."""
        self.M = np.atleast_1d(self.M)
        self.B = np.atleast_1d(self.B)
        self.K = np.atleast_1d(self.K)

        # Convert to diagonal matrices if vectors
        if self.M.ndim == 1:
            self.M = np.diag(self.M)
        if self.B.ndim == 1:
            self.B = np.diag(self.B)
        if self.K.ndim == 1:
            self.K = np.diag(self.K)

    @staticmethod
    def critical_damping(M: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Compute critical damping: B = 2*sqrt(M*K)

        Args:
            M: Inertia (diagonal)
            K: Stiffness (diagonal)

        Returns:
            B: Critical damping (diagonal)
        """
        M_diag = np.diag(M) if M.ndim == 2 else M
        K_diag = np.diag(K) if K.ndim == 2 else K
        return 2 * np.sqrt(M_diag * K_diag)


class ResidualAdmittanceController:
    """
    Residual admittance controller for internal force suppression.

    The controller computes a residual action that is added to the base
    motion policy output. The residual makes the robot compliant to
    internal forces, allowing it to "give way" and reduce squeezing/pulling.

    Theory:
        1. Internal force acts as external disturbance: F_int
        2. Admittance dynamics: M*a + B*v + K*x = F_int
        3. Solve for admittance acceleration and integrate
        4. Map Cartesian admittance motion to joint space residual
        5. Add residual to base action with gain scheduling
    """

    def __init__(self, config: Dict[str, Any], robot_index: int = 0):
        """
        Initialize residual admittance controller.

        Args:
            config: Configuration dict with keys:
                - desired_inertia: [6] or [6×6] inertia matrix
                - desired_damping: [6] or [6×6] damping matrix (or "critical")
                - desired_stiffness: [6] or [6×6] stiffness matrix
                - residual_gain: Scalar gain for residual action (0-1)
                - max_residual_magnitude: Maximum allowed residual magnitude
                - enable_gain_scheduling: Use adaptive gain based on internal force
        """
        self.config = config
        self.robot_index = int(robot_index)
        if self.robot_index < 0:
            raise ValueError("robot_index must be non-negative")

        # Parse admittance parameters
        M = np.array(config['desired_inertia'])
        K = np.array(config['desired_stiffness'])

        # Damping: use critical damping if specified
        damping_config = config.get('desired_damping')
        if isinstance(damping_config, str) and damping_config == "critical":
            B = AdmittanceParameters.critical_damping(M, K)
        else:
            B = np.array(damping_config)

        self.params = AdmittanceParameters(M=M, B=B, K=K)

        # Control parameters
        self.residual_gain = config.get('residual_gain', 0.3)
        self.max_residual_mag = config.get('max_residual_magnitude', 0.1)
        self.enable_gain_scheduling = config.get('enable_gain_scheduling', False)

        # Admittance state (Cartesian space: [x, y, z, rx, ry, rz])
        self.x_admittance = np.zeros(6)  # Admittance displacement
        self.v_admittance = np.zeros(6)  # Admittance velocity
        self.a_admittance = np.zeros(6)  # Admittance acceleration

        # Limits
        self.max_displacement = config.get('max_admittance_displacement', 0.1)  # meters/radians
        self.max_velocity = config.get('max_admittance_velocity', 0.5)  # m/s, rad/s

    def reset(self):
        """Reset admittance state."""
        self.x_admittance = np.zeros(6)
        self.v_admittance = np.zeros(6)
        self.a_admittance = np.zeros(6)

    def compute_admittance_dynamics(self,
                                   F_internal: np.ndarray,
                                   dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute admittance dynamics response to internal force.

        Admittance equation:
            M*a + B*v + K*x = F_internal

        Solving for acceleration:
            a = M^(-1) * (F_internal - B*v - K*x)

        Args:
            F_internal: Internal force wrench [6] (force + moment)
            dt: Time step (seconds)

        Returns:
            x, v, a: Admittance displacement, velocity, acceleration

        Reference: Hogan (1985), Equation (2)
        """
        # Solve for admittance acceleration
        # a = M^(-1) * (F_internal - B*v - K*x)
        self.a_admittance = np.linalg.solve(
            self.params.M,
            F_internal - self.params.B @ self.v_admittance - self.params.K @ self.x_admittance
        )

        # Semi-implicit Euler integration (more stable than explicit)
        self.v_admittance += self.a_admittance * dt
        self.x_admittance += self.v_admittance * dt

        # Apply velocity limits
        v_mag = np.linalg.norm(self.v_admittance)
        if v_mag > self.max_velocity:
            self.v_admittance *= self.max_velocity / v_mag

        # Apply displacement limits
        x_mag = np.linalg.norm(self.x_admittance)
        if x_mag > self.max_displacement:
            self.x_admittance *= self.max_displacement / x_mag

        return self.x_admittance, self.v_admittance, self.a_admittance

    def cartesian_to_joint_residual(self,
                                   x_cart: np.ndarray,
                                   v_cart: np.ndarray,
                                   robot_state: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Map Cartesian admittance motion to joint space residual action.

        For now, we use a simple proportional mapping:
            residual = K_p * x_cart + K_d * v_cart

        TODO: For more accurate mapping, use Jacobian:
            residual = J^(-1) * (K_p * x_cart + K_d * v_cart)

        Args:
            x_cart: Cartesian admittance displacement [6]
            v_cart: Cartesian admittance velocity [6]
            robot_state: Optional robot state for Jacobian computation

        Returns:
            residual: Joint space residual action
        """
        # Simple proportional-derivative Cartesian residual.
        K_p = 1.0  # Position gain
        K_d = 0.1  # Velocity gain
        residual_cart = K_p * x_cart + K_d * v_cart

        if robot_state is None or 'jacobian' not in robot_state:
            return residual_cart

        jacobian = np.asarray(robot_state['jacobian'], dtype=float)
        if jacobian.ndim != 2 or jacobian.shape[0] != 6:
            raise ValueError("robot_state['jacobian'] must have shape (6, n)")

        residual_joint = np.linalg.pinv(jacobian) @ residual_cart
        action_indices = robot_state.get('action_indices')
        if action_indices is None:
            return residual_joint

        action_indices = np.asarray(action_indices, dtype=int)
        if action_indices.shape != (residual_joint.size,):
            raise ValueError(
                "robot_state['action_indices'] must match Jacobian columns"
            )
        action_size = int(robot_state.get('action_size', action_indices.max() + 1))
        if np.any(action_indices < 0) or np.any(action_indices >= action_size):
            raise ValueError("robot_state['action_indices'] are out of range")
        residual_action = np.zeros(action_size)
        residual_action[action_indices] = residual_joint
        return residual_action

    def compute_adaptive_gain(self, internal_force_magnitude: float) -> float:
        """
        Compute adaptive residual gain based on internal force magnitude.

        Strategy:
            - Low internal force: low gain (don't interfere with base policy)
            - High internal force: high gain (actively suppress)

        Uses sigmoid function for smooth transition.

        Args:
            internal_force_magnitude: Magnitude of internal force (N)

        Returns:
            Adaptive gain (0-1)
        """
        if not self.enable_gain_scheduling:
            return self.residual_gain

        # Sigmoid gain scheduling
        F_nominal = 20.0  # Nominal internal force threshold (N)
        slope = 0.2  # Sigmoid slope

        # Sigmoid: g(F) = g_min + (g_max - g_min) / (1 + exp(-slope*(F - F_nom)))
        g_min = 0.1 * self.residual_gain
        g_max = self.residual_gain

        gain = g_min + (g_max - g_min) / (
            1 + np.exp(-slope * (internal_force_magnitude - F_nominal))
        )

        return gain

    def compute_residual_action(self,
                               base_action: np.ndarray,
                               internal_force_info: Dict[str, Any],
                               dt: float,
                               robot_state: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute residual action to be added to base policy output.

        Args:
            base_action: Base action from pre-trained policy
            internal_force_info: Output from InternalForceAnalyzer containing:
                - 'F_internal': Internal force wrench [6n]
                - 'per_robot': Per-robot decomposition
                - 'internal_magnitude': Scalar magnitude
            dt: Time step (seconds)
            robot_state: Optional robot state for advanced mapping

        Returns:
            corrected_action: Base action + residual
            diagnostics: Dict with diagnostic information
        """
        # Extract the internal wrench assigned to this controller's robot.
        per_robot = internal_force_info.get('per_robot', [])
        if per_robot:
            if self.robot_index >= len(per_robot):
                raise IndexError("robot_index exceeds per_robot decomposition")
            F_internal = per_robot[self.robot_index]['F_internal']
        else:
            F_internal_total = internal_force_info['F_internal']
            n_robots = len(F_internal_total) // 6
            if self.robot_index >= n_robots:
                raise IndexError("robot_index exceeds F_internal decomposition")
            start = 6 * self.robot_index
            F_internal = F_internal_total[start:start + 6]

        # Compute admittance dynamics
        x_adm, v_adm, a_adm = self.compute_admittance_dynamics(F_internal, dt)

        # Map to joint space residual
        mapping_state = dict(robot_state or {})
        mapping_state.setdefault('action_size', np.asarray(base_action).size)
        residual_action = self.cartesian_to_joint_residual(
            x_adm, v_adm, mapping_state
        )

        # Apply adaptive gain
        internal_mag = internal_force_info['internal_magnitude']
        adaptive_gain = self.compute_adaptive_gain(internal_mag)

        # Scale residual
        residual_action *= adaptive_gain

        # Apply magnitude limit
        residual_mag = np.linalg.norm(residual_action)
        if residual_mag > self.max_residual_mag:
            residual_action *= self.max_residual_mag / residual_mag

        base_action = np.asarray(base_action, dtype=float)
        if residual_action.shape != base_action.shape:
            raise ValueError(
                "Residual action shape {} does not match base action shape {}; "
                "provide robot_state['jacobian'] and action_indices for mapping"
                .format(residual_action.shape, base_action.shape)
            )
        corrected_action = base_action + residual_action

        # Diagnostics
        diagnostics = {
            'residual_action': residual_action,
            'residual_magnitude': residual_mag,
            'adaptive_gain': adaptive_gain,
            'admittance_displacement': x_adm,
            'admittance_velocity': v_adm,
            'admittance_acceleration': a_adm,
            'F_internal': F_internal
        }

        return corrected_action, diagnostics

    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Get current admittance state.

        Returns:
            Dict with x, v, a admittance states
        """
        return {
            'x': self.x_admittance.copy(),
            'v': self.v_admittance.copy(),
            'a': self.a_admittance.copy()
        }

    def set_parameters(self, **kwargs):
        """
        Update controller parameters at runtime.

        Args:
            **kwargs: Parameter updates (e.g., residual_gain=0.5)
        """
        for key, value in kwargs.items():
            if key == 'residual_gain':
                self.residual_gain = value
            elif key == 'max_residual_magnitude':
                self.max_residual_mag = value
            elif key == 'enable_gain_scheduling':
                self.enable_gain_scheduling = value
            elif key in ['desired_inertia', 'desired_damping', 'desired_stiffness']:
                # Update admittance parameters
                if key == 'desired_inertia':
                    self.params.M = np.diag(np.atleast_1d(value))
                elif key == 'desired_damping':
                    self.params.B = np.diag(np.atleast_1d(value))
                elif key == 'desired_stiffness':
                    self.params.K = np.diag(np.atleast_1d(value))
