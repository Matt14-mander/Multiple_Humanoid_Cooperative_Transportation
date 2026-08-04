import numpy as np
import pytest

from dual_tron1_mujoco.ifsm_disturbance_test import (
    _grasp_bias_wrenches,
    run_case,
)


def test_fixed_base_grasp_bias_reduces_internal_force_without_instability():
    common = {
        "duration_s": 1.2,
        "disturbance_start_s": 0.3,
        "measurement_start_s": 0.6,
    }
    off = run_case("grasp_bias", enable_ifsm=False, **common)
    on = run_case("grasp_bias", enable_ifsm=True, **common)

    assert on["internal_rms"] <= 0.75 * off["internal_rms"]
    assert on["internal_peak"] <= 0.80 * off["internal_peak"]
    assert on["payload_error_peak_m"] <= 0.02
    assert on["payload_tilt_peak_rad"] <= 0.10
    assert on["grasp_gap_peak_m"] <= 0.015
    assert on["arm_saturation_peak"] <= 0.05
    assert on["correction_peak"] > 0.0


def test_grasp_bias_has_zero_resultant_force_and_moment():
    wrenches = _grasp_bias_wrenches(force_n=8.0)
    resultant_force = np.sum(wrenches[:, :3], axis=0)
    grasp_offsets = (
        np.array([0.0, -0.354, 0.0]),
        np.array([0.0, 0.354, 0.0]),
    )
    resultant_moment = sum(
        (
            np.cross(offset, wrench[:3]) + wrench[3:]
            for offset, wrench in zip(grasp_offsets, wrenches)
        ),
        start=np.zeros(3),
    )

    assert np.allclose(resultant_force, 0.0)
    assert np.allclose(resultant_moment, 0.0)


@pytest.mark.parametrize("scenario", ["asymmetric_payload", "single_arm"])
def test_fixed_base_physical_disturbances_stay_within_pose_limits(scenario):
    result = run_case(
        scenario,
        enable_ifsm=True,
        duration_s=2.0,
        disturbance_start_s=0.5,
        measurement_start_s=0.8,
    )

    assert result["payload_error_peak_m"] <= 0.018
    assert result["payload_tilt_peak_rad"] <= 0.08
    assert result["grasp_gap_peak_m"] <= 0.015
    assert result["arm_saturation_peak"] <= 0.05
    assert result["correction_peak"] > 0.0
