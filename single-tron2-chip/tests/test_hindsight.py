import numpy as np
import pytest

from tron2_chip.hindsight import HistoryBuffer, compliance_matrix, hindsight_goal
from tron2_chip.perturbation import ForcePulse


def test_scalar_hindsight_goal_matches_chip_equation():
    reference = np.array([0.4, -0.1, 1.2])
    force = np.array([10.0, -2.0, 0.0])
    result = hindsight_goal(reference, force, 0.002)
    np.testing.assert_allclose(result, reference - 0.002 * force)


def test_axis_specific_compliance_only_moves_enabled_axis():
    reference = np.ones(3)
    result = hindsight_goal(reference, [10.0, 10.0, 10.0], [0.002, 0.0, 0.0])
    np.testing.assert_allclose(result, [0.98, 1.0, 1.0])


def test_invalid_compliance_is_rejected():
    with pytest.raises(ValueError):
        compliance_matrix([-0.1, 0.0, 0.0])


def test_history_is_initialized_and_oldest_to_newest():
    history = HistoryBuffer(3, (2,))
    history.reset([1.0, 2.0])
    history.append([3.0, 4.0])
    np.testing.assert_allclose(history.array(), [[1, 2], [1, 2], [3, 4]])


def test_force_pulse_has_half_open_time_interval():
    pulse = ForcePulse(np.array([5.0, 0.0, 0.0]), 0.5, 0.2)
    np.testing.assert_allclose(pulse.at(0.499), 0.0)
    np.testing.assert_allclose(pulse.at(0.5), [5.0, 0.0, 0.0])
    np.testing.assert_allclose(pulse.at(0.699), [5.0, 0.0, 0.0])
    np.testing.assert_allclose(pulse.at(0.7), 0.0)

