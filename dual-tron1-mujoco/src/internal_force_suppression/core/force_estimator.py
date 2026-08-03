"""
Force Estimator - Sensorless force estimation using momentum observer.

Based on:
    De Luca, A., & Mattone, R. (2005).
    "Sensorless robot collision detection and hybrid force/motion control."
    IEEE International Conference on Robotics and Automation (ICRA).

The generalized momentum observer estimates external forces/torques without
direct force sensors by monitoring the difference between expected and actual
robot dynamics.
"""

import numpy as np
from typing import Dict, Optional, Any
import pinocchio as pin


class GeneralizedMomentumObserver:
    """
    Generalized Momentum Observer for external torque estimation.

    Theory:
        The generalized momentum is defined as: p = M(q) * v
        where M(q) is the mass matrix, q is joint position, v is joint velocity.

        The dynamics with external torque:
            M(q)*dv/dt + C(q,v)*v + g(q) = τ_cmd + τ_ext

        Rearranging:
            τ_ext = M(q)*dv/dt + C(q,v)*v + g(q) - τ_cmd
                  = dp/dt + h(q,v) - τ_cmd

        Using a momentum residual r with observer gain K:
            beta(q,v) = g(q) - C(q,v)^T*v
            r = K * [integral(τ_measured - beta - r)dt - (p - p_0)]

        With this module's torque sign convention, the residual follows
        dr/dt = K*(τ_ext - r), so its steady-state value is τ_ext.

    Reference: De Luca & Mattone (2005), Equations (8)-(10)
    """

    def __init__(self,
                 robot_model: pin.Model,
                 observer_gain: float = 100.0,
                 cutoff_frequency: Optional[float] = None):
        """
        Initialize momentum observer.

        Args:
            robot_model: Pinocchio robot model
            observer_gain: Observer gain K (higher = faster convergence, more noise)
                          Typical range: 50-200 Hz
            cutoff_frequency: Optional low-pass filter cutoff (Hz)
        """
        self.model = robot_model
        self.data = robot_model.createData()
        self.K = observer_gain
        self.cutoff_freq = cutoff_frequency

        # State
        self.nv = robot_model.nv
        self.r = np.zeros(self.nv)  # Momentum residual
        self.p_0 = np.zeros(self.nv)  # Initial momentum
        self.momentum_integral = np.zeros(self.nv)
        self.is_initialized = False

        # Low-pass filter state (if enabled)
        self.tau_ext_filtered = np.zeros(self.nv)

    def reset(self):
        """Reset observer state."""
        self.r = np.zeros(self.nv)
        self.p_0 = np.zeros(self.nv)
        self.momentum_integral = np.zeros(self.nv)
        self.tau_ext_filtered = np.zeros(self.nv)
        self.is_initialized = False

    def initialize(self, q: np.ndarray, v: np.ndarray):
        """
        Initialize observer with current state.

        Args:
            q: Joint positions [nq]
            v: Joint velocities [nv]
        """
        # Compute initial momentum
        pin.computeAllTerms(self.model, self.data, q, v)
        M = self.data.M
        self.p_0 = M @ v
        self.momentum_integral = np.zeros(self.nv)
        self.is_initialized = True

    def estimate_external_torque(self,
                                 q: np.ndarray,
                                 v: np.ndarray,
                                 tau_measured: np.ndarray,
                                 dt: float) -> np.ndarray:
        """
        Estimate external joint torques.

        Args:
            q: Joint positions [nv]
            v: Joint velocities [nv]
            tau_measured: Measured/commanded joint torques [nv]
            dt: Time step (seconds)

        Returns:
            tau_ext: Estimated external torques [nv]
        """
        if not self.is_initialized:
            self.initialize(q, v)
            return np.zeros(self.nv)

        # Compute robot dynamics using Pinocchio
        # This computes M, C, g terms
        pin.computeAllTerms(self.model, self.data, q, v)

        M = self.data.M  # Mass matrix
        C = self.data.C  # Coriolis matrix
        g = self.data.g  # Gravity generalized torque
        beta = g - C.T @ v

        # Integrate the observer residual. This produces the first-order
        # response r_dot = K * (tau_ext - r).
        self.momentum_integral += (tau_measured - beta - self.r) * dt

        # Compute current momentum
        p = M @ v

        # Recover the external-torque residual from momentum balance.
        self.r = self.K * (
            self.momentum_integral - (p - self.p_0)
        )
        tau_ext = self.r.copy()

        # Optional low-pass filtering
        if self.cutoff_freq is not None:
            alpha = dt * 2 * np.pi * self.cutoff_freq / (1 + dt * 2 * np.pi * self.cutoff_freq)
            self.tau_ext_filtered = (1 - alpha) * self.tau_ext_filtered + alpha * tau_ext
            tau_ext = self.tau_ext_filtered

        return tau_ext

    def joint_torque_to_cartesian_wrench(self,
                                        tau_ext: np.ndarray,
                                        q: np.ndarray,
                                        frame_name: str) -> np.ndarray:
        """
        Convert joint space external torque to Cartesian wrench.

        Uses: F_ext = (J^T)^+ * τ_ext
        where J is the Jacobian of the specified frame.

        Args:
            tau_ext: External joint torques [nv]
            q: Joint positions [nv]
            frame_name: Name of the frame (e.g., end-effector)

        Returns:
            wrench: Cartesian wrench [fx, fy, fz, mx, my, mz]
        """
        # Get frame ID
        frame_id = self.model.getFrameId(frame_name)

        # Compute Jacobian
        pin.framesForwardKinematics(self.model, self.data, q)
        J = pin.computeFrameJacobian(self.model, self.data, q, frame_id,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        # Solve: J^T * F = τ_ext  =>  F = (J^T)^+ * τ_ext
        wrench = np.linalg.lstsq(J.T, tau_ext, rcond=None)[0]

        return wrench


class ImplicitForceEstimator:
    """
    High-level interface for implicit force estimation.

    Wraps the momentum observer and provides a clean API for the IFSM system.
    """

    def __init__(self,
                 robot_model: pin.Model,
                 config: Dict[str, Any]):
        """
        Initialize force estimator.

        Args:
            robot_model: Pinocchio robot model
            config: Configuration dict with keys:
                - estimator_type: "momentum_observer" (future: "torque_based", etc.)
                - observer_gain: float (default: 100.0)
                - cutoff_frequency: float or None (default: None)
                - end_effector_frame: str (frame name for Cartesian wrench)
        """
        self.model = robot_model
        self.config = config

        estimator_type = config.get('estimator_type', 'momentum_observer')
        observer_gain = config.get('observer_gain', 100.0)
        cutoff_freq = config.get('cutoff_frequency', None)
        self.ee_frame = config.get('end_effector_frame', 'hand')

        if estimator_type == 'momentum_observer':
            self.estimator = GeneralizedMomentumObserver(
                robot_model=robot_model,
                observer_gain=observer_gain,
                cutoff_frequency=cutoff_freq
            )
        else:
            raise ValueError(f"Unknown estimator type: {estimator_type}")

    def reset(self):
        """Reset estimator state."""
        self.estimator.reset()

    def estimate_contact_wrench(self, robot_state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Estimate contact wrench from robot state.

        Args:
            robot_state: Dict containing:
                - 'q': joint positions [nv]
                - 'v': joint velocities [nv]
                - 'tau': joint torques [nv]
                - 'dt': time step (seconds)

        Returns:
            Dict containing:
                - 'wrench': Cartesian wrench [6] (force + moment)
                - 'force': Force vector [3]
                - 'moment': Moment vector [3]
                - 'tau_ext': Joint space external torque [nv]
        """
        q = robot_state['q']
        v = robot_state['v']
        tau = robot_state['tau']
        dt = robot_state['dt']

        # Estimate joint space external torque
        tau_ext = self.estimator.estimate_external_torque(q, v, tau, dt)

        # Convert to Cartesian wrench
        wrench = self.estimator.joint_torque_to_cartesian_wrench(
            tau_ext, q, self.ee_frame
        )

        return {
            'wrench': wrench,
            'force': wrench[:3],
            'moment': wrench[3:],
            'tau_ext': tau_ext
        }
