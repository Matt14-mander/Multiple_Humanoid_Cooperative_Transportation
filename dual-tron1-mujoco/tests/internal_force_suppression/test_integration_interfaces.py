import numpy as np
import pytest

from internal_force_suppression.core.admittance_controller import (
    ResidualAdmittanceController,
)


def _config():
    return {
        "desired_inertia": np.ones(6),
        "desired_damping": np.zeros(6),
        "desired_stiffness": np.zeros(6),
        "residual_gain": 1.0,
        "max_residual_magnitude": 10.0,
        "enable_gain_scheduling": False,
        "max_admittance_displacement": 10.0,
        "max_admittance_velocity": 10.0,
    }


def _force_info():
    first = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    second = -first
    return {
        "F_internal": np.concatenate((first, second)),
        "internal_magnitude": float(np.linalg.norm((first, second))),
        "per_robot": [
            {"F_internal": first},
            {"F_internal": second},
        ],
    }


def test_integrated_controller_module_imports():
    from internal_force_suppression.integrated_controller import (
        DualRobotCooperativeController,
    )

    assert DualRobotCooperativeController is not None


def test_second_controller_uses_second_robot_internal_wrench():
    controller = ResidualAdmittanceController(_config(), robot_index=1)

    _, diagnostics = controller.compute_residual_action(
        base_action=np.zeros(6),
        internal_force_info=_force_info(),
        dt=0.01,
    )

    assert np.allclose(
        diagnostics["F_internal"],
        _force_info()["per_robot"][1]["F_internal"],
    )
    assert np.all(diagnostics["residual_action"][:3] < 0.0)


def test_cartesian_residual_maps_into_selected_action_channels():
    controller = ResidualAdmittanceController(_config(), robot_index=0)
    action_indices = np.array([0, 3, 6, 9, 10, 11])

    action, diagnostics = controller.compute_residual_action(
        base_action=np.zeros(14),
        internal_force_info=_force_info(),
        dt=0.01,
        robot_state={
            "jacobian": np.eye(6),
            "action_indices": action_indices,
        },
    )

    inactive = np.setdiff1d(np.arange(14), action_indices)
    assert action.shape == (14,)
    assert np.allclose(action[inactive], 0.0)
    assert np.allclose(action[action_indices], diagnostics["residual_action"][action_indices])


def test_non_six_dimensional_action_requires_mapping():
    controller = ResidualAdmittanceController(_config(), robot_index=0)

    with pytest.raises(ValueError, match="provide robot_state"):
        controller.compute_residual_action(
            base_action=np.zeros(14),
            internal_force_info=_force_info(),
            dt=0.01,
        )
