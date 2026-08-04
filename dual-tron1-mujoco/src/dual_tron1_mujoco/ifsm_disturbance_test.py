"""Fixed-base OFF/ON disturbance benchmark for MuJoCo IFSM integration."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Optional

import mujoco
import numpy as np

from internal_force_suppression.config.ifsm_config import IFSMConfig

from .carry_controller import CooperativeCarryHoldController
from .configuration import load_config
from .control import JointHoldController
from .internal_force_controller import MujocoInternalForceController
from .model_loader import load_model
from .paths import CARRY_HOLD_CONFIG, CARRY_HOLD_MODEL


SCENARIOS = ("grasp_bias", "asymmetric_payload", "single_arm")


def _body_id(model, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise KeyError("MuJoCo body not found: " + name)
    return body_id


def _grasp_bias_wrenches(force_n: float = 8.0) -> np.ndarray:
    """Return a pure squeezing null-space wrench for y-axis handles."""
    wrenches = np.zeros((2, 6))
    wrenches[0, 1] = force_n
    wrenches[1, 1] = -force_n
    return wrenches


def _apply_external_disturbance(model, data, scenario: str) -> np.ndarray:
    """Apply one step of the selected physical disturbance."""
    data.qfrc_applied.fill(0.0)
    commanded_wrench = np.zeros((2, 6))
    if scenario == "grasp_bias":
        return _grasp_bias_wrenches()

    if scenario == "asymmetric_payload":
        payload_id = _body_id(model, "payload_body")
        force = np.array([0.0, 0.0, -8.0])
        torque = np.zeros(3)
        application_point = data.xpos[payload_id] + np.array([0.0, -0.25, 0.0])
        mujoco.mj_applyFT(
            model,
            data,
            force,
            torque,
            application_point,
            payload_id,
            data.qfrc_applied,
        )
        return commanded_wrench

    if scenario == "single_arm":
        body_id = _body_id(model, "r1_link6")
        mujoco.mj_applyFT(
            model,
            data,
            np.array([0.0, 6.0, 0.0]),
            np.zeros(3),
            data.xpos[body_id],
            body_id,
            data.qfrc_applied,
        )
        return commanded_wrench

    raise ValueError("Unknown disturbance scenario: " + scenario)


def run_case(
    scenario: str,
    enable_ifsm: bool,
    config_path: Path = CARRY_HOLD_CONFIG,
    model_path: Path = CARRY_HOLD_MODEL,
    duration_s: float = 2.0,
    disturbance_start_s: float = 0.5,
    measurement_start_s: float = 0.8,
    residual_gain: float = None,
    carry_impedance_override: Optional[Mapping[str, object]] = None,
) -> Dict[str, float]:
    if scenario not in SCENARIOS:
        raise ValueError("Unknown disturbance scenario: " + scenario)
    if measurement_start_s < disturbance_start_s:
        raise ValueError("measurement_start_s must not precede disturbance_start_s")

    model = load_model(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    config = load_config(config_path)
    controllers = [
        JointHoldController(model, data, prefix, config["control"])
        for prefix in ("r1_", "r2_")
    ]
    carry_config = dict(config["control"])
    carry_config.update(config["carry_impedance"])
    if carry_impedance_override:
        carry_config.update(carry_impedance_override)
    carry = CooperativeCarryHoldController(model, data, carry_config)
    ifsm_config = IFSMConfig().to_dict()
    ifsm_config["safety"]["enable"] = False
    if residual_gain is not None:
        if not 0.0 <= residual_gain <= 1.0:
            raise ValueError("residual_gain must be between 0 and 1")
        ifsm_config["admittance_robot1"]["residual_gain"] = residual_gain
        ifsm_config["admittance_robot2"]["residual_gain"] = residual_gain
        ifsm_config["mujoco_adapter"]["residual_gain"] = residual_gain
    ifsm = MujocoInternalForceController(model, ifsm_config)

    payload_id = _body_id(model, "payload_body")
    grasp_pairs = [
        (_body_id(model, "r1_link6"), _body_id(model, "payload_grasp_left")),
        (_body_id(model, "r2_link6"), _body_id(model, "payload_grasp_right")),
    ]
    payload_origin = data.xpos[payload_id].copy()
    dt = float(model.opt.timestep)
    internal_samples = []
    payload_errors = []
    payload_tilts = []
    grasp_gaps = []
    correction_norms = []

    while data.time < duration_s:
        for controller in controllers:
            controller.update(data)
        correction, diagnostics = ifsm.update(data, dt)
        applied_correction = correction if enable_ifsm else np.zeros((2, 6))
        disturbance_wrench = np.zeros((2, 6))
        if data.time >= disturbance_start_s:
            disturbance_wrench = _apply_external_disturbance(
                model, data, scenario
            )
        else:
            data.qfrc_applied.fill(0.0)
        carry.update(data, disturbance_wrench + applied_correction)
        mujoco.mj_step(model, data)

        if data.time >= measurement_start_s:
            internal_samples.append(float(diagnostics["internal_magnitude"]))
            payload_errors.append(
                float(np.linalg.norm(data.xpos[payload_id] - payload_origin))
            )
            z_axis = data.xmat[payload_id].reshape(3, 3)[:, 2]
            payload_tilts.append(
                math.acos(float(np.clip(z_axis[2], -1.0, 1.0)))
            )
            grasp_gaps.append(
                max(
                    float(np.linalg.norm(data.xpos[robot] - data.xpos[payload]))
                    for robot, payload in grasp_pairs
                )
            )
            correction_norms.append(
                float(np.max(np.linalg.norm(applied_correction, axis=1)))
            )

    internal = np.asarray(internal_samples)
    return {
        "scenario": scenario,
        "ifsm_enabled": bool(enable_ifsm),
        "internal_peak": float(np.max(internal)),
        "internal_rms": float(np.sqrt(np.mean(internal**2))),
        "internal_final": float(internal[-1]),
        "payload_error_peak_m": float(np.max(payload_errors)),
        "payload_tilt_peak_rad": float(np.max(payload_tilts)),
        "grasp_gap_peak_m": float(np.max(grasp_gaps)),
        "arm_saturation_peak": float(np.max(carry.saturation_fractions)),
        "correction_peak": float(np.max(correction_norms)),
    }


def run_comparison(**kwargs) -> Dict[str, Dict[str, Dict[str, float]]]:
    results = {}
    for scenario in SCENARIOS:
        off = run_case(scenario, enable_ifsm=False, **kwargs)
        on = run_case(scenario, enable_ifsm=True, **kwargs)
        peak_reduction = 100.0 * (
            1.0 - on["internal_peak"] / max(off["internal_peak"], 1e-12)
        )
        rms_reduction = 100.0 * (
            1.0 - on["internal_rms"] / max(off["internal_rms"], 1e-12)
        )
        results[scenario] = {
            "off": off,
            "on": on,
            "comparison": {
                "internal_peak_reduction_percent": peak_reduction,
                "internal_rms_reduction_percent": rms_reduction,
            },
        }
    return results


def _print_results(results) -> None:
    print(
        "scenario              peak_off  peak_on  peak_red   "
        "rms_off   rms_on   rms_red  payload_off/on  tilt_off/on"
    )
    for scenario, result in results.items():
        off, on, comparison = (
            result["off"],
            result["on"],
            result["comparison"],
        )
        print(
            "{:<21} {:8.3f} {:8.3f} {:7.1f}% "
            "{:8.3f} {:8.3f} {:7.1f}% {:7.4f}/{:7.4f} "
            "{:7.4f}/{:7.4f}".format(
                scenario,
                off["internal_peak"],
                on["internal_peak"],
                comparison["internal_peak_reduction_percent"],
                off["internal_rms"],
                on["internal_rms"],
                comparison["internal_rms_reduction_percent"],
                off["payload_error_peak_m"],
                on["payload_error_peak_m"],
                off["payload_tilt_peak_rad"],
                on["payload_tilt_peak_rad"],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CARRY_HOLD_CONFIG)
    parser.add_argument("--model", type=Path, default=CARRY_HOLD_MODEL)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--disturbance-start", type=float, default=0.5)
    parser.add_argument("--measurement-start", type=float, default=0.8)
    parser.add_argument("--residual-gain", type=float)
    parser.add_argument("--translation-stiffness", type=float, nargs=3)
    parser.add_argument("--translation-damping", type=float, nargs=3)
    parser.add_argument("--rotation-stiffness", type=float, nargs=3)
    parser.add_argument("--rotation-damping", type=float, nargs=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    impedance_override = {
        key: value
        for key, value in (
            ("translation_stiffness", args.translation_stiffness),
            ("translation_damping", args.translation_damping),
            ("rotation_stiffness", args.rotation_stiffness),
            ("rotation_damping", args.rotation_damping),
        )
        if value is not None
    }
    results = run_comparison(
        config_path=args.config,
        model_path=args.model,
        duration_s=args.duration,
        disturbance_start_s=args.disturbance_start,
        measurement_start_s=args.measurement_start,
        residual_gain=args.residual_gain,
        carry_impedance_override=impedance_override,
    )
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_results(results)


if __name__ == "__main__":
    main()
