from pathlib import Path

import numpy as np

from dual_tron1_mujoco.airbot_observer_robustness import (
    run_robustness_case,
    run_robustness_suite,
)


def test_split_model_errors_distinguish_static_mass_matrix_error(tmp_path: Path):
    matrix_static = run_robustness_case(
        "mass_matrix_10pct_low_zero", model_path=tmp_path / "matrix.xml"
    )
    matrix_motion = run_robustness_case(
        "mass_matrix_10pct_low_motion", model_path=tmp_path / "motion.xml"
    )
    mass_static = run_robustness_case(
        "mass_10pct_low_zero", model_path=tmp_path / "mass.xml"
    )
    com_static = run_robustness_case(
        "com_offset_10mm_x_zero", model_path=tmp_path / "com.xml"
    )
    gravity_static = run_robustness_case(
        "gravity_5pct_low_zero", model_path=tmp_path / "gravity.xml"
    )

    assert matrix_static["force_rmse_n"] < 0.05
    assert matrix_motion["force_rmse_n"] > matrix_static["force_rmse_n"]
    assert mass_static["force_rmse_n"] > 0.20
    assert com_static["force_rmse_n"] > 0.05
    assert gravity_static["force_rmse_n"] > 0.20


def test_airbot_robustness_suite_tracks_passes_and_known_gaps(tmp_path: Path):
    results = run_robustness_suite(model_path=tmp_path / "airbot.xml")

    expected_passes = {
        "state_noise",
        "unmodeled_friction",
        "delay_5ms_jitter_10pct",
        "near_singular_pose",
        "impulse_40ms_motion",
    }
    expected_gaps = {
        "torque_calibration",
        "inertia_10pct_low",
        "inertia_10pct_low_zero",
        "unmodeled_tool_200g",
        "unmodeled_tool_200g_zero",
        "drivetrain_loss",
        "combined_adverse",
        "mass_10pct_low_motion",
        "mass_10pct_low_zero",
        "com_offset_10mm_x_motion",
        "com_offset_10mm_x_zero",
        "gravity_5pct_low_motion",
        "gravity_5pct_low_zero",
    }

    expected_split_passes = {
        "mass_matrix_10pct_low_motion",
        "mass_matrix_10pct_low_zero",
    }

    assert all(results[name]["accepted"] for name in expected_passes)
    assert all(not results[name]["accepted"] for name in expected_gaps)
    assert all(results[name]["accepted"] for name in expected_split_passes)
    assert all(
        np.isfinite(result["force_rmse_n"]) for result in results.values()
    )
    assert results["state_noise"]["force_rmse_n"] < 0.20
    assert results["impulse_40ms_motion"][
        "post_disturbance_settling_time_s"
    ] < 0.10
    assert results["near_singular_pose"]["jacobian_condition_peak"] > 300.0
    assert results["unmodeled_tool_200g_zero"]["force_rmse_n"] > 1.0


def test_static_bias_and_tool_model_compensations_remove_false_force(
    tmp_path: Path,
):
    mass_raw = run_robustness_case(
        "mass_10pct_low_zero", model_path=tmp_path / "mass_raw.xml"
    )
    mass_fixed = run_robustness_case(
        "mass_10pct_low_zero_compensated",
        model_path=tmp_path / "mass_fixed.xml",
    )
    tool_raw = run_robustness_case(
        "unmodeled_tool_200g_zero", model_path=tmp_path / "tool_raw.xml"
    )
    tool_fixed = run_robustness_case(
        "modeled_tool_200g_zero", model_path=tmp_path / "tool_fixed.xml"
    )

    assert mass_fixed["force_rmse_n"] < 0.20
    assert mass_fixed["force_rmse_n"] < 0.25 * mass_raw["force_rmse_n"]
    assert tool_fixed["force_rmse_n"] < 0.20
    assert tool_fixed["force_rmse_n"] < 0.25 * tool_raw["force_rmse_n"]

    mass_contact = run_robustness_case(
        "mass_10pct_low_static_compensated",
        model_path=tmp_path / "mass_contact.xml",
    )
    tool_contact = run_robustness_case(
        "modeled_tool_200g_static",
        model_path=tmp_path / "tool_contact.xml",
    )
    assert mass_contact["force_rmse_n"] < 0.20
    assert tool_contact["force_rmse_n"] < 0.20
    assert np.linalg.norm(mass_contact["estimated_wrench_mean"][:3]) > 1.0
    assert np.linalg.norm(tool_contact["estimated_wrench_mean"][:3]) > 1.0
