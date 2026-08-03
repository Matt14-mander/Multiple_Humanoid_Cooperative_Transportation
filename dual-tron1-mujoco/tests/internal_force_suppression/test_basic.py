"""
Simplified unit tests for Force Estimator (without Pinocchio dependency).

These tests verify the core logic without requiring full robot models.
For full integration tests with Pinocchio, see test_force_estimator_full.py
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestForceEstimatorBasics:
    """Test basic force estimation concepts without full robot models."""

    def test_module_imports(self):
        """Test that modules can be imported."""
        try:
            from internal_force_suppression.core import force_estimator
            from internal_force_suppression.core import internal_force_analyzer
            from internal_force_suppression.core import admittance_controller
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import modules: {e}")

    def test_config_loading(self):
        """Test configuration loading."""
        from internal_force_suppression.config import IFSMConfig

        # Test default config
        config = IFSMConfig()
        assert config is not None
        assert 'force_estimator' in config.config
        assert 'force_analyzer' in config.config
        assert 'admittance_robot1' in config.config

    def test_config_values(self):
        """Test configuration values."""
        from internal_force_suppression.config import IFSMConfig

        config = IFSMConfig()

        # Check force estimator config
        assert config['force_estimator']['observer_gain'] == 100.0
        assert config['force_estimator']['estimator_type'] == 'momentum_observer'

        # Check force analyzer config
        assert config['force_analyzer']['max_safe_internal_force'] == 50.0

        # Check admittance config
        assert config['admittance_robot1']['residual_gain'] == 0.3


class TestWrenchDecomposition:
    """Test wrench decomposition utilities."""

    def test_decompose_wrench(self):
        """Test wrench decomposition."""
        from internal_force_suppression.utils.wrench_decomposition import decompose_wrench

        wrench = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        force, moment = decompose_wrench(wrench)

        assert np.allclose(force, [1.0, 2.0, 3.0])
        assert np.allclose(moment, [4.0, 5.0, 6.0])

    def test_wrench_magnitude(self):
        """Test wrench magnitude calculation."""
        from internal_force_suppression.utils.wrench_decomposition import wrench_magnitude

        wrench = np.array([3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
        mag = wrench_magnitude(wrench)

        assert np.isclose(mag, 5.0)  # 3-4-5 triangle

    def test_normalize_wrench(self):
        """Test wrench normalization."""
        from internal_force_suppression.utils.wrench_decomposition import normalize_wrench

        wrench = np.array([10.0, 0.0, 0.0, 0.0, 5.0, 0.0])
        normalized = normalize_wrench(wrench, max_force=5.0, max_moment=2.0)

        # Force should be scaled down to 5.0
        assert np.isclose(np.linalg.norm(normalized[:3]), 5.0)
        # Moment should be scaled down to 2.0
        assert np.isclose(np.linalg.norm(normalized[3:]), 2.0)


class TestInternalForceDecomposition:
    """Test internal force decomposition (mathematical components)."""

    def test_skew_symmetric(self):
        """Test skew-symmetric matrix generation."""
        from internal_force_suppression.core.internal_force_analyzer import skew_symmetric

        v = np.array([1.0, 2.0, 3.0])
        skew = skew_symmetric(v)

        # Check anti-symmetry
        assert np.allclose(skew, -skew.T)

        # Check specific values
        assert skew[0, 1] == -3.0
        assert skew[0, 2] == 2.0
        assert skew[1, 2] == -1.0

    def test_contact_wrench_dataclass(self):
        """Test ContactWrench dataclass."""
        from internal_force_suppression.core.internal_force_analyzer import ContactWrench

        force = np.array([1.0, 2.0, 3.0])
        moment = np.array([4.0, 5.0, 6.0])
        contact_point = np.array([0.5, 0.0, 1.0])

        wrench = ContactWrench(
            force=force,
            moment=moment,
            contact_point=contact_point
        )

        assert np.allclose(wrench.force, force)
        assert np.allclose(wrench.moment, moment)
        assert np.allclose(wrench.contact_point, contact_point)
        assert len(wrench.wrench) == 6


class TestAdmittanceParameters:
    """Test admittance parameter handling."""

    def test_admittance_parameters_creation(self):
        """Test AdmittanceParameters creation."""
        from internal_force_suppression.core.admittance_controller import AdmittanceParameters

        M = np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
        B = np.array([50.0, 50.0, 50.0, 5.0, 5.0, 5.0])
        K = np.array([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])

        params = AdmittanceParameters(M=M, B=B, K=K)

        # Should convert to diagonal matrices
        assert params.M.shape == (6, 6)
        assert params.B.shape == (6, 6)
        assert params.K.shape == (6, 6)

    def test_critical_damping_calculation(self):
        """Test critical damping calculation."""
        from internal_force_suppression.core.admittance_controller import AdmittanceParameters

        M = np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
        K = np.array([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])

        B_critical = AdmittanceParameters.critical_damping(M, K)

        # Critical damping: B = 2*sqrt(M*K)
        expected = 2 * np.sqrt(M * K)
        assert np.allclose(B_critical, expected)


class TestSafetyMonitor:
    """Test safety monitoring system."""

    def test_safety_monitor_initialization(self):
        """Test SafetyMonitor initialization."""
        from internal_force_suppression.utils.safety_monitor import SafetyMonitor

        config = {
            'enable': True,
            'emergency_stop_threshold': 100.0,
            'gradual_stop_threshold': 70.0,
            'force_rate_limit': 200.0,
            'enable_emergency_stop': True
        }

        monitor = SafetyMonitor(config)

        assert monitor.enabled == True
        assert monitor.emergency_threshold == 100.0
        assert monitor.current_status == "normal"

    def test_safety_monitor_check_safe(self):
        """Test safety monitoring with safe forces."""
        from internal_force_suppression.utils.safety_monitor import SafetyMonitor

        config = {
            'enable': True,
            'emergency_stop_threshold': 100.0,
            'gradual_stop_threshold': 70.0,
            'force_rate_limit': 200.0,
            'enable_emergency_stop': True
        }

        monitor = SafetyMonitor(config)

        force_info = {
            'internal_magnitude': 30.0,
            'safety_status': 'safe'
        }

        is_safe = monitor.check(force_info, dt=0.002)
        assert is_safe == True
        assert monitor.current_status == "normal"

    def test_safety_monitor_check_danger(self):
        """Test safety monitoring with dangerous forces."""
        from internal_force_suppression.utils.safety_monitor import SafetyMonitor

        config = {
            'enable': True,
            'emergency_stop_threshold': 100.0,
            'gradual_stop_threshold': 70.0,
            'force_rate_limit': 200.0,
            'enable_emergency_stop': True
        }

        monitor = SafetyMonitor(config)

        force_info = {
            'internal_magnitude': 150.0,
            'safety_status': 'danger'
        }

        is_safe = monitor.check(force_info, dt=0.002)
        assert is_safe == False
        assert monitor.current_status == "emergency"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
