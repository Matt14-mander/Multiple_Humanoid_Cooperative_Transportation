"""
Integrated Controller - Combines base motion policy with internal force suppression.

This is the main interface for using the IFSM system.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import time

from .core.force_estimator import FREE_SPACE, ImplicitForceEstimator
from .core.internal_force_analyzer import InternalForceAnalyzer, ContactWrench
from .core.admittance_controller import ResidualAdmittanceController
from .utils.safety_monitor import SafetyMonitor
from .config.ifsm_config import IFSMConfig


class DualRobotCooperativeController:
    """
    Integrated controller for dual-robot cooperative manipulation.

    Combines:
        - Base motion policy (pre-trained)
        - Internal force suppression (IFSM)

    Usage:
        controller = DualRobotCooperativeController(motion_policy, robot_models, config)

        for step in simulation:
            observation = get_observation()
            actions = controller.step(observation, dt)
            apply_actions(actions)
    """

    def __init__(self,
                 motion_policy,
                 robot1_model,
                 robot2_model,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize integrated controller.

        Args:
            motion_policy: Pre-trained motion policy with .predict(obs) method
            robot1_model: Pinocchio model for robot 1
            robot2_model: Pinocchio model for robot 2
            config: Configuration dict or IFSMConfig (uses default if None)
        """
        self.motion_policy = motion_policy

        # Load configuration
        if config is None:
            self.config = IFSMConfig()
        elif isinstance(config, dict):
            self.config = IFSMConfig(**config)
        else:
            self.config = config

        # Initialize force estimators
        self.force_estimator_1 = ImplicitForceEstimator(
            robot_model=robot1_model,
            config=self.config['force_estimator']
        )
        self.force_estimator_2 = ImplicitForceEstimator(
            robot_model=robot2_model,
            config=self.config['force_estimator']
        )

        # Initialize internal force analyzer
        self.force_analyzer = InternalForceAnalyzer(
            config=self.config['force_analyzer']
        )

        # Initialize admittance controllers
        self.admittance_controller_1 = ResidualAdmittanceController(
            config=self.config['admittance_robot1'],
            robot_index=0,
        )
        self.admittance_controller_2 = ResidualAdmittanceController(
            config=self.config['admittance_robot2'],
            robot_index=1,
        )

        # Initialize safety monitor
        self.safety_monitor = SafetyMonitor(
            config=self.config['safety']
        )

        # Statistics
        self.step_count = 0
        self.total_time = 0.0
        self.enable_profiling = self.config['system'].get('enable_profiling', False)
        self.contact_phase = FREE_SPACE

    def set_contact_phase(self, phase: str) -> None:
        """Apply one shared grasp-state transition to both observers."""
        self.force_estimator_1.set_contact_phase(phase)
        self.force_estimator_2.set_contact_phase(phase)
        self.contact_phase = phase

    def reset(self):
        """Reset all components."""
        self.force_estimator_1.reset()
        self.force_estimator_2.reset()
        self.admittance_controller_1.reset()
        self.admittance_controller_2.reset()
        self.safety_monitor.reset()
        self.force_analyzer.reset_history()
        self.step_count = 0
        self.total_time = 0.0
        self.contact_phase = FREE_SPACE
        self.set_contact_phase(FREE_SPACE)

    def step(self,
            observation: Dict[str, Any],
            dt: float) -> Dict[str, Any]:
        """
        Execute one control step.

        Args:
            observation: Observation dict containing:
                - 'robot1_state': {'q', 'v', 'tau'}
                - 'robot2_state': {'q', 'v', 'tau'}
                - 'object_state': {'com', 'mass', ...}
                - (optional) 'robot1_contact_point': [3]
                - (optional) 'robot2_contact_point': [3]
            dt: Time step (seconds)

        Returns:
            Dict containing:
                - 'robot1_action': Action for robot 1
                - 'robot2_action': Action for robot 2
                - 'diagnostics': Diagnostic information
        """
        start_time = time.time() if self.enable_profiling else None

        # 1. Get base actions from motion policy
        base_action_1, base_action_2 = self.motion_policy.predict(observation)

        # 2. Estimate contact forces
        robot1_state = dict(observation['robot1_state'])
        robot2_state = dict(observation['robot2_state'])

        robot1_state['dt'] = dt
        robot2_state['dt'] = dt
        self.set_contact_phase(
            observation.get('contact_phase', self.contact_phase)
        )
        robot1_state['contact_phase'] = self.contact_phase
        robot2_state['contact_phase'] = self.contact_phase

        wrench_1_dict = self.force_estimator_1.estimate_contact_wrench(robot1_state)
        wrench_2_dict = self.force_estimator_2.estimate_contact_wrench(robot2_state)

        # Create ContactWrench objects
        contact_point_1 = observation.get('robot1_contact_point',
                                         observation['object_state']['com'])
        contact_point_2 = observation.get('robot2_contact_point',
                                         observation['object_state']['com'])

        wrench_1 = ContactWrench(
            force=wrench_1_dict['force'],
            moment=wrench_1_dict['moment'],
            contact_point=contact_point_1
        )
        wrench_2 = ContactWrench(
            force=wrench_2_dict['force'],
            moment=wrench_2_dict['moment'],
            contact_point=contact_point_2
        )

        # 3. Analyze internal forces
        force_info = self.force_analyzer.analyze(
            robot1_wrench=wrench_1,
            robot2_wrench=wrench_2,
            object_state=observation['object_state']
        )

        # 4. Safety check
        is_safe = self.safety_monitor.check(force_info, dt)

        if not is_safe:
            # Get safe actions
            safe_action_1 = self.safety_monitor.get_safe_action(base_action_1)
            safe_action_2 = self.safety_monitor.get_safe_action(base_action_2)

            return {
                'robot1_action': safe_action_1,
                'robot2_action': safe_action_2,
                'diagnostics': {
                    'safety_triggered': True,
                    'safety_status': self.safety_monitor.get_status(),
                    'internal_force_info': force_info
                }
            }

        # 5. Compute residual admittance actions
        action_1, residual_info_1 = self.admittance_controller_1.compute_residual_action(
            base_action=base_action_1,
            internal_force_info=force_info,
            dt=dt,
            robot_state=robot1_state
        )

        action_2, residual_info_2 = self.admittance_controller_2.compute_residual_action(
            base_action=base_action_2,
            internal_force_info=force_info,
            dt=dt,
            robot_state=robot2_state
        )

        # 6. Compile diagnostics
        diagnostics = {
            'safety_triggered': False,
            'contact_phase': self.contact_phase,
            'internal_force': {
                'magnitude': force_info['internal_magnitude'],
                'ratio': force_info['internal_ratio'],
                'safety_status': force_info['safety_status'],
                'safety_ratio': force_info['safety_ratio']
            },
            'robot1': {
                'base_action': base_action_1,
                'residual': residual_info_1,
                'wrench': wrench_1_dict
            },
            'robot2': {
                'base_action': base_action_2,
                'residual': residual_info_2,
                'wrench': wrench_2_dict
            },
            'force_decomposition': force_info
        }

        # Profiling
        if self.enable_profiling:
            elapsed = time.time() - start_time
            self.total_time += elapsed
            self.step_count += 1
            diagnostics['timing'] = {
                'step_time': elapsed,
                'avg_time': self.total_time / self.step_count
            }

        return {
            'robot1_action': action_1,
            'robot2_action': action_2,
            'diagnostics': diagnostics
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistical summary of controller performance.

        Returns:
            Dict with statistics from all components
        """
        return {
            'force_analyzer': self.force_analyzer.get_statistics(),
            'safety_monitor': self.safety_monitor.get_statistics(),
            'admittance_1': self.admittance_controller_1.get_state(),
            'admittance_2': self.admittance_controller_2.get_state(),
            'step_count': self.step_count,
            'total_time': self.total_time,
            'avg_step_time': self.total_time / self.step_count if self.step_count > 0 else 0.0
        }

    def update_config(self, config_updates: Dict[str, Any]):
        """
        Update configuration at runtime.

        Args:
            config_updates: Dict of config updates (can be nested)
        """
        from .config.ifsm_config import merge_configs
        self.config.config = merge_configs(self.config.config, config_updates)

        # Update components that support runtime updates
        if 'admittance_robot1' in config_updates:
            for key, value in config_updates['admittance_robot1'].items():
                self.admittance_controller_1.set_parameters(**{key: value})

        if 'admittance_robot2' in config_updates:
            for key, value in config_updates['admittance_robot2'].items():
                self.admittance_controller_2.set_parameters(**{key: value})
