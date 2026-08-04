"""Repeatable AIRBOT momentum-observer robustness benchmark."""

import argparse
import json
from pathlib import Path
from typing import Dict

from .airbot_observer_validation import (
    ObserverPerturbation,
    run_validation_case,
)
from .paths import AIRBOT_OBSERVER_MODEL


ROBUSTNESS_CASES = {
    "baseline_motion": {
        "scenario": "constant_wrench_motion",
    },
    "state_noise": {
        "scenario": "constant_wrench_motion",
        "perturbation": ObserverPerturbation(
            position_noise_std_rad=0.0005,
            velocity_noise_std_rad_s=0.01,
            cutoff_frequency_hz=20.0,
        ),
    },
    "torque_calibration": {
        "scenario": "constant_wrench_motion",
        "perturbation": ObserverPerturbation(
            torque_noise_std_nm=0.03,
            torque_scale=0.98,
            cutoff_frequency_hz=20.0,
        ),
    },
    "inertia_10pct_low": {
        "scenario": "constant_wrench_motion",
        "perturbation": ObserverPerturbation(
            observer_inertia_scale=0.90,
            cutoff_frequency_hz=20.0,
        ),
    },
    "inertia_10pct_low_zero": {
        "scenario": "zero_force",
        "perturbation": ObserverPerturbation(
            observer_inertia_scale=0.90,
            cutoff_frequency_hz=20.0,
        ),
    },
    "mass_matrix_10pct_low_motion": {
        "scenario": "constant_wrench_motion",
        "perturbation": ObserverPerturbation(
            momentum_matrix_scale=0.90,
            cutoff_frequency_hz=20.0,
        ),
    },
    "mass_matrix_10pct_low_zero": {
        "scenario": "zero_force",
        "perturbation": ObserverPerturbation(
            momentum_matrix_scale=0.90,
            cutoff_frequency_hz=20.0,
        ),
    },
    "mass_10pct_low_motion": {
        "scenario": "constant_wrench_motion",
        "perturbation": ObserverPerturbation(
            observer_mass_scale=0.90,
            cutoff_frequency_hz=20.0,
        ),
    },
    "mass_10pct_low_zero": {
        "scenario": "zero_force",
        "perturbation": ObserverPerturbation(
            observer_mass_scale=0.90,
            cutoff_frequency_hz=20.0,
        ),
    },
    "mass_10pct_low_zero_compensated": {
        "scenario": "zero_force",
        "perturbation": ObserverPerturbation(
            observer_mass_scale=0.90,
            cutoff_frequency_hz=20.0,
            enable_bias_compensation=True,
        ),
    },
    "mass_10pct_low_static_compensated": {
        "scenario": "constant_wrench_static",
        "perturbation": ObserverPerturbation(
            observer_mass_scale=0.90,
            cutoff_frequency_hz=20.0,
            enable_bias_compensation=True,
        ),
    },
    "com_offset_10mm_x_motion": {
        "scenario": "constant_wrench_motion",
        "perturbation": ObserverPerturbation(
            observer_com_offset_m=(0.01, 0.0, 0.0),
            cutoff_frequency_hz=20.0,
        ),
    },
    "com_offset_10mm_x_zero": {
        "scenario": "zero_force",
        "perturbation": ObserverPerturbation(
            observer_com_offset_m=(0.01, 0.0, 0.0),
            cutoff_frequency_hz=20.0,
        ),
    },
    "gravity_5pct_low_motion": {
        "scenario": "constant_wrench_motion",
        "perturbation": ObserverPerturbation(
            observer_gravity_scale=0.95,
            cutoff_frequency_hz=20.0,
        ),
    },
    "gravity_5pct_low_zero": {
        "scenario": "zero_force",
        "perturbation": ObserverPerturbation(
            observer_gravity_scale=0.95,
            cutoff_frequency_hz=20.0,
        ),
    },
    "unmodeled_friction": {
        "scenario": "constant_wrench_motion",
        "perturbation": ObserverPerturbation(
            joint_damping_nm_s_rad=0.04,
            joint_frictionloss_nm=0.03,
            cutoff_frequency_hz=20.0,
        ),
    },
    "unmodeled_tool_200g": {
        "scenario": "constant_wrench_motion",
        "perturbation": ObserverPerturbation(
            unmodeled_tool_mass_kg=0.20,
            cutoff_frequency_hz=20.0,
        ),
    },
    "unmodeled_tool_200g_zero": {
        "scenario": "zero_force",
        "perturbation": ObserverPerturbation(
            unmodeled_tool_mass_kg=0.20,
            cutoff_frequency_hz=20.0,
        ),
    },
    "modeled_tool_200g_zero": {
        "scenario": "zero_force",
        "perturbation": ObserverPerturbation(
            unmodeled_tool_mass_kg=0.20,
            observer_tool_mass_kg=0.20,
            cutoff_frequency_hz=20.0,
        ),
    },
    "modeled_tool_200g_static": {
        "scenario": "constant_wrench_static",
        "perturbation": ObserverPerturbation(
            unmodeled_tool_mass_kg=0.20,
            observer_tool_mass_kg=0.20,
            cutoff_frequency_hz=20.0,
        ),
    },
    "drivetrain_loss": {
        "scenario": "constant_wrench_motion",
        "perturbation": ObserverPerturbation(
            actuator_efficiency=0.97,
            actuator_deadzone_nm=0.01,
            cutoff_frequency_hz=20.0,
        ),
    },
    "delay_5ms_jitter_10pct": {
        "scenario": "constant_wrench_motion",
        "perturbation": ObserverPerturbation(
            sample_delay_s=0.005,
            dt_jitter_fraction=0.10,
            cutoff_frequency_hz=20.0,
        ),
    },
    "extended_pose": {
        "scenario": "constant_wrench_static",
        "pose_name": "extended",
    },
    "folded_pose": {
        "scenario": "constant_wrench_static",
        "pose_name": "folded",
    },
    "near_singular_pose": {
        "scenario": "constant_wrench_static",
        "pose_name": "near_singular",
    },
    "impulse_40ms_motion": {
        "scenario": "constant_wrench_motion",
        "duration_s": 1.0,
        "measurement_start_s": 0.35,
        "force_duration_s": 0.04,
    },
    "combined_adverse": {
        "scenario": "constant_wrench_motion",
        "perturbation": ObserverPerturbation(
            position_noise_std_rad=0.0005,
            velocity_noise_std_rad_s=0.01,
            torque_noise_std_nm=0.03,
            torque_scale=0.98,
            observer_inertia_scale=0.95,
            joint_damping_nm_s_rad=0.03,
            joint_frictionloss_nm=0.02,
            unmodeled_tool_mass_kg=0.10,
            actuator_efficiency=0.98,
            actuator_deadzone_nm=0.005,
            sample_delay_s=0.003,
            dt_jitter_fraction=0.05,
            cutoff_frequency_hz=15.0,
            random_seed=17,
        ),
    },
}

MODEL_ERROR_CASES = (
    "mass_matrix_10pct_low_motion",
    "mass_matrix_10pct_low_zero",
    "mass_10pct_low_motion",
    "mass_10pct_low_zero",
    "com_offset_10mm_x_motion",
    "com_offset_10mm_x_zero",
    "gravity_5pct_low_motion",
    "gravity_5pct_low_zero",
)


def evaluate_robustness_result(result: Dict[str, object]) -> Dict[str, object]:
    """Apply provisional stage-one acceptance limits to one result."""
    failures = []
    scenario = result["scenario"]
    if scenario == "zero_force":
        if result["force_rmse_n"] > 0.20:
            failures.append("false_force_rmse_n")
        if result["moment_rmse_nm"] > 0.05:
            failures.append("false_moment_rmse_nm")
    elif result["force_duration_s"] is not None:
        if result["active_force_rmse_n"] > 0.60:
            failures.append("active_force_rmse_n")
        if result["force_peak_relative_error_percent"] > 15.0:
            failures.append("force_peak_relative_error_percent")
        settling = result["post_disturbance_settling_time_s"]
        if settling is None or settling > 0.10:
            failures.append("post_disturbance_settling_time_s")
    else:
        if result["force_rmse_n"] > 0.50:
            failures.append("force_rmse_n")
        if result["moment_rmse_nm"] > 0.10:
            failures.append("moment_rmse_nm")
        if result["force_direction_error_deg"] > 10.0:
            failures.append("force_direction_error_deg")
        response = result["response_time_90_s"]
        if response is None or response > 0.15:
            failures.append("response_time_90_s")
    if result["actuator_saturation_fraction"] > 0.01:
        failures.append("actuator_saturation_fraction")
    return {"accepted": not failures, "failed_metrics": failures}


def run_robustness_case(
    name: str,
    model_path: Path = AIRBOT_OBSERVER_MODEL,
) -> Dict[str, object]:
    if name not in ROBUSTNESS_CASES:
        raise ValueError("Unknown AIRBOT robustness case: " + name)
    arguments = dict(ROBUSTNESS_CASES[name])
    arguments["model_path"] = model_path
    result = run_validation_case(**arguments)
    result["robustness_case"] = name
    result.update(evaluate_robustness_result(result))
    return result


def run_robustness_suite(
    model_path: Path = AIRBOT_OBSERVER_MODEL,
) -> Dict[str, Dict[str, object]]:
    return {
        name: run_robustness_case(name, model_path=model_path)
        for name in ROBUSTNESS_CASES
    }


def run_model_error_suite(
    model_path: Path = AIRBOT_OBSERVER_MODEL,
) -> Dict[str, Dict[str, object]]:
    """Run only the isolated mass-matrix, mass, COM and gravity cases."""
    return {
        name: run_robustness_case(name, model_path=model_path)
        for name in MODEL_ERROR_CASES
    }


def _print_results(results: Dict[str, Dict[str, object]]) -> None:
    print(
        "case                         tau_rmse force_rmse moment_rmse "
        "direction  response  jac_cond  result"
    )
    for name, result in results.items():
        response = result["response_time_90_s"]
        response_text = "   n/a" if response is None else "{:6.3f}".format(response)
        print(
            "{:<28} {:8.4f} {:10.4f} {:11.4f} {:8.2f}deg {}s {:8.1f}  {}".format(
                name,
                result["joint_torque_rmse_nm"],
                result["force_rmse_n"],
                result["moment_rmse_nm"],
                result["force_direction_error_deg"],
                response_text,
                result["jacobian_condition_peak"],
                "PASS" if result["accepted"] else "FAIL",
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=AIRBOT_OBSERVER_MODEL)
    parser.add_argument(
        "--case",
        choices=("all", "model-errors") + tuple(ROBUSTNESS_CASES),
        default="all",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.case == "all":
        results = run_robustness_suite(model_path=args.model)
    elif args.case == "model-errors":
        results = run_model_error_suite(model_path=args.model)
    else:
        results = {
            args.case: run_robustness_case(args.case, model_path=args.model)
        }
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_results(results)


if __name__ == "__main__":
    main()
