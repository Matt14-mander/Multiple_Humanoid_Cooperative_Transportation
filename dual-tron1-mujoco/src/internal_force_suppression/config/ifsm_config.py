"""
Configuration loader for Internal Force Suppression Module.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load IFSM configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_default_config() -> Dict[str, Any]:
    """
    Get default IFSM configuration.

    Returns:
        Default configuration dictionary
    """
    # Find default config file
    current_dir = Path(__file__).parent.parent.parent.parent
    default_config_path = current_dir / "configs" / "internal_force_suppression" / "default_ifsm.yaml"

    if default_config_path.exists():
        return load_config(str(default_config_path))
    else:
        # Fallback to hardcoded defaults
        return {
            'force_estimator': {
                'estimator_type': 'momentum_observer',
                'observer_gain': 100.0,
                'cutoff_frequency': None,
                'end_effector_frame': 'hand'
            },
            'force_analyzer': {
                'max_safe_internal_force': 50.0,
                'warning_threshold': 35.0,
                'decomposition_method': 'null_space_projection'
            },
            'admittance_robot1': {
                'desired_inertia': [10.0, 10.0, 10.0, 1.0, 1.0, 1.0],
                'desired_damping': [50.0, 50.0, 50.0, 5.0, 5.0, 5.0],
                'desired_stiffness': [100.0, 100.0, 100.0, 10.0, 10.0, 10.0],
                'residual_gain': 0.3,
                'max_residual_magnitude': 0.1,
                'enable_gain_scheduling': False,
                'max_admittance_displacement': 0.1,
                'max_admittance_velocity': 0.5
            },
            'admittance_robot2': {
                'desired_inertia': [10.0, 10.0, 10.0, 1.0, 1.0, 1.0],
                'desired_damping': [50.0, 50.0, 50.0, 5.0, 5.0, 5.0],
                'desired_stiffness': [100.0, 100.0, 100.0, 10.0, 10.0, 10.0],
                'residual_gain': 0.3,
                'max_residual_magnitude': 0.1,
                'enable_gain_scheduling': False,
                'max_admittance_displacement': 0.1,
                'max_admittance_velocity': 0.5
            },
            'safety': {
                'enable': True,
                'emergency_stop_threshold': 100.0,
                'gradual_stop_threshold': 70.0,
                'force_rate_limit': 200.0,
                'enable_emergency_stop': True
            },
            'system': {
                'control_frequency': 500,
                'log_level': 2,
                'enable_profiling': False
            },
            'mujoco_adapter': {
                'residual_gain': 0.5,
                'wrench_cutoff_frequency_hz': 20.0,
                'max_correction_force_n': 5.0,
                'max_correction_torque_nm': 1.0
            }
        }


def merge_configs(base_config: Dict[str, Any],
                 override_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge override config into base config.

    Args:
        base_config: Base configuration
        override_config: Override configuration (optional)

    Returns:
        Merged configuration
    """
    if override_config is None:
        return base_config

    merged = base_config.copy()

    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


class IFSMConfig:
    """
    IFSM configuration manager.
    """

    def __init__(self, config_path: Optional[str] = None, **overrides):
        """
        Initialize configuration.

        Args:
            config_path: Optional path to config file (uses default if None)
            **overrides: Keyword arguments to override config values
        """
        if config_path is not None:
            self.config = load_config(config_path)
        else:
            self.config = get_default_config()

        # Apply overrides
        if overrides:
            self.config = merge_configs(self.config, overrides)

    def __getitem__(self, key: str) -> Any:
        """Get config value by key."""
        return self.config[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        return self.config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.config.copy()

    def save(self, path: str):
        """Save configuration to file."""
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
