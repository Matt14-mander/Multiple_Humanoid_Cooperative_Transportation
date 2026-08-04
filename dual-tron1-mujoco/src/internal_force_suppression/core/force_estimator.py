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


FREE_SPACE = "free_space"
GRASP = "grasp"
CARRY = "carry"
RELEASE = "release"
CONTACT_PHASES = (FREE_SPACE, GRASP, CARRY, RELEASE)


class ObserverBiasCompensator:
    """Learn slow joint-residual bias only while external contact is absent.

    The estimate is frozen during grasp and carry so a sustained payload force
    cannot be absorbed as model bias.  Release explicitly re-opens learning.
    """

    def __init__(self, size: int, time_constant_s: float = 0.10):
        if time_constant_s <= 0.0:
            raise ValueError("bias time constant must be positive")
        self.size = int(size)
        self.time_constant_s = float(time_constant_s)
        self.bias = np.zeros(self.size)
        self.phase = FREE_SPACE
        self.sample_count = 0

    @property
    def learning_enabled(self) -> bool:
        return self.phase in {FREE_SPACE, RELEASE}

    def reset(self) -> None:
        self.bias.fill(0.0)
        self.phase = FREE_SPACE
        self.sample_count = 0

    def set_phase(self, phase: str) -> None:
        phase = str(phase).lower()
        if phase not in CONTACT_PHASES:
            raise ValueError("unknown observer contact phase: " + phase)
        self.phase = phase

    def compensate(
        self, residual: np.ndarray, dt: float, phase: Optional[str] = None
    ) -> np.ndarray:
        value = np.asarray(residual, dtype=float)
        if value.shape != (self.size,):
            raise ValueError("observer residual has an unexpected shape")
        if dt <= 0.0:
            raise ValueError("observer timestep must be positive")
        if phase is not None:
            self.set_phase(phase)
        if self.learning_enabled:
            alpha = 1.0 - np.exp(-float(dt) / self.time_constant_s)
            self.bias += alpha * (value - self.bias)
            self.sample_count += 1
        return value - self.bias


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
                 cutoff_frequency: Optional[float] = None,
                 momentum_matrix_scale: float = 1.0,
                 bias_compensation_time_constant_s: Optional[float] = None):
        """
        Initialize momentum observer.

        Args:
            robot_model: Pinocchio robot model
            observer_gain: Observer gain K (higher = faster convergence, more noise)
                          Typical range: 50-200 Hz
            cutoff_frequency: Optional low-pass filter cutoff (Hz)
            momentum_matrix_scale: Scale applied only to the mass matrix used
                in generalized momentum. This is primarily useful for
                isolating mass-matrix errors in validation; it does not alter
                gravity or Coriolis terms.
        """
        if momentum_matrix_scale <= 0.0:
            raise ValueError("momentum_matrix_scale must be positive")
        self.model = robot_model
        self.data = robot_model.createData()
        self.K = observer_gain
        self.cutoff_freq = cutoff_frequency
        self.momentum_matrix_scale = momentum_matrix_scale

        # State
        self.nv = robot_model.nv
        self.r = np.zeros(self.nv)  # Momentum residual
        self.p_0 = np.zeros(self.nv)  # Initial momentum
        self.momentum_integral = np.zeros(self.nv)
        self.is_initialized = False

        # Low-pass filter state (if enabled)
        self.tau_ext_filtered = np.zeros(self.nv)
        self.raw_tau_ext = np.zeros(self.nv)
        self.bias_compensator = (
            ObserverBiasCompensator(
                self.nv, bias_compensation_time_constant_s
            )
            if bias_compensation_time_constant_s is not None
            else None
        )

    def reset(self):
        """Reset observer state."""
        self.r = np.zeros(self.nv)
        self.p_0 = np.zeros(self.nv)
        self.momentum_integral = np.zeros(self.nv)
        self.tau_ext_filtered = np.zeros(self.nv)
        self.raw_tau_ext = np.zeros(self.nv)
        if self.bias_compensator is not None:
            self.bias_compensator.reset()
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
        M = self.momentum_matrix_scale * self.data.M
        self.p_0 = M @ v
        self.momentum_integral = np.zeros(self.nv)
        self.is_initialized = True

    def estimate_external_torque(self,
                                 q: np.ndarray,
                                 v: np.ndarray,
                                 tau_measured: np.ndarray,
                                 dt: float,
                                 contact_phase: Optional[str] = None) -> np.ndarray:
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

        M = self.momentum_matrix_scale * self.data.M  # Momentum matrix
        C = self.data.C  # Coriolis matrix
        g = self.data.g  # Gravity generalized torque
        beta = g - C.T @ v

        # From p_dot = tau_measured + tau_ext - beta, integrate the known
        # model terms and feed the residual back with the sign required for
        # r_dot = K * (tau_ext - r).
        self.momentum_integral += (tau_measured - beta + self.r) * dt

        # Compute current momentum
        p = M @ v

        # Recover the external-torque residual from momentum balance.
        self.r = self.K * ((p - self.p_0) - self.momentum_integral)
        tau_ext = self.r.copy()

        # Optional low-pass filtering
        if self.cutoff_freq is not None:
            alpha = dt * 2 * np.pi * self.cutoff_freq / (1 + dt * 2 * np.pi * self.cutoff_freq)
            self.tau_ext_filtered = (1 - alpha) * self.tau_ext_filtered + alpha * tau_ext
            tau_ext = self.tau_ext_filtered

        self.raw_tau_ext = tau_ext.copy()
        if self.bias_compensator is not None:
            tau_ext = self.bias_compensator.compensate(
                tau_ext, dt, phase=contact_phase
            )

        return tau_ext

    @property
    def estimated_bias(self) -> np.ndarray:
        if self.bias_compensator is None:
            return np.zeros(self.nv)
        return self.bias_compensator.bias.copy()

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
        bias_time_constant = (
            config.get('bias_time_constant_s', 0.10)
            if config.get('bias_compensation_enabled', False)
            else None
        )
        self.ee_frame = config.get('end_effector_frame', 'hand')

        if estimator_type == 'momentum_observer':
            self.estimator = GeneralizedMomentumObserver(
                robot_model=robot_model,
                observer_gain=observer_gain,
                cutoff_frequency=cutoff_freq,
                bias_compensation_time_constant_s=bias_time_constant,
            )
        else:
            raise ValueError(f"Unknown estimator type: {estimator_type}")

    def reset(self):
        """Reset estimator state."""
        self.estimator.reset()

    def set_contact_phase(self, phase: str) -> None:
        """Switch bias learning according to the task contact state."""
        if self.estimator.bias_compensator is None:
            return
        self.estimator.bias_compensator.set_phase(phase)

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
        tau_ext = self.estimator.estimate_external_torque(
            q,
            v,
            tau,
            dt,
            contact_phase=robot_state.get('contact_phase'),
        )

        # Convert to Cartesian wrench
        wrench = self.estimator.joint_torque_to_cartesian_wrench(
            tau_ext, q, self.ee_frame
        )

        return {
            'wrench': wrench,
            'force': wrench[:3],
            'moment': wrench[3:],
            'tau_ext': tau_ext,
            'tau_ext_raw': self.estimator.raw_tau_ext.copy(),
            'tau_bias': self.estimator.estimated_bias,
        }
