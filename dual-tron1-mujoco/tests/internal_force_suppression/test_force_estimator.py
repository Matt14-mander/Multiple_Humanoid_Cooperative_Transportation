"""
Unit tests for Force Estimator.
"""

import pytest
import numpy as np
import pinocchio as pin
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from internal_force_suppression.core.force_estimator import (
    GeneralizedMomentumObserver,
    ImplicitForceEstimator
)


@pytest.fixture
def simple_robot():
    """Create a simple 2-DOF robot for testing."""
    model = pin.Model()
    parent_id = 0

    # Joint 1
    joint1 = pin.JointModelRY()
    placement1 = pin.SE3.Identity()
    inertia1 = pin.Inertia.FromSphere(1.0, 0.1)
    joint1_id = model.addJoint(parent_id, joint1, placement1, "joint1")
    model.appendBodyToJoint(joint1_id, inertia1, pin.SE3.Identity())

    # Joint 2
    joint2 = pin.JointModelRY()
    placement2 = pin.SE3(np.eye(3), np.array([0, 0, 0.5]))
    inertia2 = pin.Inertia.FromSphere(0.5, 0.05)
    joint2_id = model.addJoint(joint1_id, joint2, placement2, "joint2")
    model.appendBodyToJoint(joint2_id, inertia2, pin.SE3.Identity())

    # End-effector frame
    ee_placement = pin.SE3(np.eye(3), np.array([0, 0, 0.5]))
    model.addFrame(pin.Frame("hand", joint2_id, ee_placement, pin.FrameType.OP_FRAME))

    return model


class TestGeneralizedMomentumObserver:
    """Test cases for momentum observer."""

    def test_initialization(self, simple_robot):
        """Test observer initialization."""
        observer = GeneralizedMomentumObserver(simple_robot, observer_gain=100.0)

        assert observer.nv == simple_robot.nv
        assert observer.K == 100.0
        assert not observer.is_initialized

    def test_reset(self, simple_robot):
        """Test observer reset."""
        observer = GeneralizedMomentumObserver(simple_robot)

        # Initialize and run
        q = np.zeros(simple_robot.nv)
        v = np.ones(simple_robot.nv)
        observer.initialize(q, v)
        assert observer.is_initialized

        # Reset
        observer.reset()
        assert not observer.is_initialized
        assert np.allclose(observer.r, 0.0)

    def test_zero_external_force(self, simple_robot):
        """Test that zero external force is estimated correctly."""
        observer = GeneralizedMomentumObserver(simple_robot, observer_gain=100.0)

        q = np.zeros(simple_robot.nv)
        v = np.zeros(simple_robot.nv)

        # Compute dynamics
        data = simple_robot.createData()
        pin.computeAllTerms(simple_robot, data, q, v)
        tau_measured = data.g  # Only gravity, no external force

        # Run observer for several steps
        dt = 0.001
        for _ in range(100):
            tau_ext = observer.estimate_external_torque(q, v, tau_measured, dt)

        # Should converge to near zero
        assert np.linalg.norm(tau_ext) < 0.1

    def test_constant_external_force(self, simple_robot):
        """Test estimation of constant external force."""
        observer = GeneralizedMomentumObserver(simple_robot, observer_gain=150.0)

        q = np.zeros(simple_robot.nv)
        v = np.zeros(simple_robot.nv)
        tau_ext_true = np.array([5.0, -3.0])

        # Compute dynamics
        data = simple_robot.createData()
        pin.computeAllTerms(simple_robot, data, q, v)
        tau_measured = data.g + tau_ext_true

        # Run observer
        dt = 0.001
        for _ in range(500):  # Let it converge
            tau_ext_est = observer.estimate_external_torque(q, v, tau_measured, dt)

        # Should be close to true value
        error = np.linalg.norm(tau_ext_est - tau_ext_true)
        assert error < 0.5


class TestImplicitForceEstimator:
    """Test cases for high-level force estimator."""

    def test_initialization(self, simple_robot):
        """Test estimator initialization."""
        config = {
            'estimator_type': 'momentum_observer',
            'observer_gain': 100.0,
            'end_effector_frame': 'hand'
        }

        estimator = ImplicitForceEstimator(simple_robot, config)
        assert estimator.ee_frame == 'hand'

    def test_estimate_contact_wrench(self, simple_robot):
        """Test contact wrench estimation."""
        config = {
            'estimator_type': 'momentum_observer',
            'observer_gain': 100.0,
            'end_effector_frame': 'hand'
        }

        estimator = ImplicitForceEstimator(simple_robot, config)

        robot_state = {
            'q': np.zeros(simple_robot.nv),
            'v': np.zeros(simple_robot.nv),
            'tau': np.array([1.0, 0.5]),
            'dt': 0.001
        }

        result = estimator.estimate_contact_wrench(robot_state)

        assert 'wrench' in result
        assert 'force' in result
        assert 'moment' in result
        assert 'tau_ext' in result

        assert len(result['wrench']) == 6
        assert len(result['force']) == 3
        assert len(result['moment']) == 3

    def test_invalid_estimator_type(self, simple_robot):
        """Test that invalid estimator type raises error."""
        config = {
            'estimator_type': 'invalid_type',
            'observer_gain': 100.0,
            'end_effector_frame': 'hand'
        }

        with pytest.raises(ValueError):
            ImplicitForceEstimator(simple_robot, config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
