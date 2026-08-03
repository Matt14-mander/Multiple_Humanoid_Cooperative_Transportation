"""
Safety Monitor - Monitor internal forces and trigger safety responses.

Monitors internal force magnitude and rate of change to detect
potentially dangerous situations.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from collections import deque
import time


class SafetyMonitor:
    """
    Safety monitoring system for internal force suppression.

    Monitors:
        - Internal force magnitude
        - Force rate of change
        - Sustained high forces

    Actions:
        - Warning: Log warning message
        - Gradual stop: Reduce velocity gradually
        - Emergency stop: Immediate halt
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize safety monitor.

        Args:
            config: Configuration dict with keys:
                - enable: Enable safety monitoring
                - emergency_stop_threshold: Force threshold for emergency stop (N)
                - gradual_stop_threshold: Force threshold for gradual stop (N)
                - force_rate_limit: Maximum force rate of change (N/s)
                - enable_emergency_stop: Actually trigger emergency stop
        """
        self.config = config
        self.enabled = config.get('enable', True)
        self.emergency_threshold = config.get('emergency_stop_threshold', 100.0)
        self.gradual_threshold = config.get('gradual_stop_threshold', 70.0)
        self.force_rate_limit = config.get('force_rate_limit', 200.0)
        self.enable_emergency = config.get('enable_emergency_stop', True)

        # State tracking
        self.last_force = 0.0
        self.last_time = time.time()
        self.force_history = deque(maxlen=100)
        self.violation_count = 0

        # Status
        self.current_status = "normal"  # "normal", "warning", "gradual_stop", "emergency"
        self.last_violation_time = None

    def reset(self):
        """Reset monitor state."""
        self.last_force = 0.0
        self.last_time = time.time()
        self.force_history.clear()
        self.violation_count = 0
        self.current_status = "normal"
        self.last_violation_time = None

    def check(self, force_info: Dict[str, Any], dt: Optional[float] = None) -> bool:
        """
        Check if current forces are safe.

        Args:
            force_info: Dict from InternalForceAnalyzer containing:
                - 'internal_magnitude': Scalar internal force magnitude
                - 'safety_status': "safe", "warning", or "danger"
            dt: Optional time step (if not provided, computed from system time)

        Returns:
            is_safe: True if safe to continue, False if action needed
        """
        if not self.enabled:
            return True

        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_time

        internal_force = force_info['internal_magnitude']

        # Check 1: Force magnitude
        if internal_force >= self.emergency_threshold:
            self.current_status = "emergency"
            self.violation_count += 1
            self.last_violation_time = current_time
            return False

        if internal_force >= self.gradual_threshold:
            self.current_status = "gradual_stop"
            self.violation_count += 1
            self.last_violation_time = current_time
            return False

        # Check 2: Force rate of change
        if dt > 0 and self.last_force > 0:
            force_rate = abs(internal_force - self.last_force) / dt
            if force_rate > self.force_rate_limit:
                self.current_status = "warning"
                self.violation_count += 1
                self.last_violation_time = current_time
                # Don't fail, just warn
                print(f"[SafetyMonitor] WARNING: High force rate: {force_rate:.1f} N/s")

        # Check 3: Sustained high force
        self.force_history.append(internal_force)
        if len(self.force_history) >= 50:  # Check last 0.1s at 500Hz
            avg_force = np.mean(list(self.force_history)[-50:])
            if avg_force > self.gradual_threshold * 0.8:
                self.current_status = "warning"
                print(f"[SafetyMonitor] WARNING: Sustained high force: {avg_force:.1f} N")

        # Update state
        self.last_force = internal_force
        self.last_time = current_time

        # If we passed all checks
        if self.current_status in ["emergency", "gradual_stop"]:
            return False

        self.current_status = "normal"
        return True

    def get_safe_action(self,
                       base_action: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Get safe action based on current status.

        Args:
            base_action: Original action (optional)

        Returns:
            safe_action: Modified action, or None for emergency stop
        """
        if not self.enable_emergency:
            # Safety monitoring disabled, return base action
            return base_action

        if self.current_status == "emergency":
            # Emergency stop: return zero action
            print(f"[SafetyMonitor] EMERGENCY STOP! Force: {self.last_force:.1f} N")
            if base_action is not None:
                return np.zeros_like(base_action)
            return None

        elif self.current_status == "gradual_stop":
            # Gradual stop: scale down action
            scale = 0.3  # Reduce to 30% of original
            print(f"[SafetyMonitor] Gradual stop active. Force: {self.last_force:.1f} N")
            if base_action is not None:
                return base_action * scale
            return None

        else:
            # Normal or warning: return base action
            return base_action

    def get_status(self) -> Dict[str, Any]:
        """
        Get current safety status.

        Returns:
            Dict with status information
        """
        return {
            'status': self.current_status,
            'last_force': self.last_force,
            'violation_count': self.violation_count,
            'last_violation_time': self.last_violation_time,
            'enabled': self.enabled
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistical summary of force history.

        Returns:
            Dict with statistics
        """
        if not self.force_history:
            return {
                'mean': 0.0,
                'max': 0.0,
                'std': 0.0,
                'current': 0.0
            }

        history = np.array(list(self.force_history))
        return {
            'mean': float(np.mean(history)),
            'max': float(np.max(history)),
            'std': float(np.std(history)),
            'current': float(history[-1]) if len(history) > 0 else 0.0,
            'samples': len(history)
        }
