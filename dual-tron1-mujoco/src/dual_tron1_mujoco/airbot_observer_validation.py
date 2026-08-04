"""AIRBOT fixed-base momentum-observer validation against MuJoCo truth."""

import argparse
import copy
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import mujoco
import numpy as np
import pinocchio as pin

from internal_force_suppression.core.force_estimator import (
    GeneralizedMomentumObserver,
)

from .build_scene import _add_inertial, _indent, _numbers, _origin_pose
from .model_loader import load_model
from .paths import AIRBOT_OBSERVER_MODEL, ARM_URDF


AIRBOT_JOINTS = ("J1", "J2", "J3", "J4", "J5", "J6")
AIRBOT_ROOT_LINK = "airbot_arm"
AIRBOT_EE_FRAME = "link6"
SCENARIOS = ("zero_force", "constant_wrench_static", "constant_wrench_motion")
AIRBOT_POSES = {
    "nominal": np.array([0.0, -1.10, 1.20, 0.0, 0.30, 0.0]),
    "extended": np.array([0.20, -0.50, 0.25, 0.30, 0.55, -0.20]),
    "folded": np.array([-0.40, -2.00, 2.40, 0.50, -0.70, 0.30]),
    # Deterministically sampled pose with a Jacobian condition number near 200.
    "near_singular": np.array(
        [1.162667, -1.013606, 2.392115, -1.455346, -0.753101, 2.013708]
    ),
}


@dataclass(frozen=True)
class ObserverPerturbation:
    """Repeatable plant and measurement imperfections for robustness tests."""

    position_noise_std_rad: float = 0.0
    velocity_noise_std_rad_s: float = 0.0
    torque_noise_std_nm: float = 0.0
    torque_scale: float = 1.0
    momentum_matrix_scale: float = 1.0
    observer_inertia_scale: float = 1.0
    observer_mass_scale: float = 1.0
    observer_com_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    observer_gravity_scale: float = 1.0
    joint_damping_nm_s_rad: float = 0.0
    joint_frictionloss_nm: float = 0.0
    unmodeled_tool_mass_kg: float = 0.0
    observer_tool_mass_kg: float = 0.0
    enable_bias_compensation: bool = False
    bias_time_constant_s: float = 0.08
    actuator_efficiency: float = 1.0
    actuator_deadzone_nm: float = 0.0
    sample_delay_s: float = 0.0
    dt_jitter_fraction: float = 0.0
    cutoff_frequency_hz: Optional[float] = None
    random_seed: int = 7


def build_airbot_pinocchio_model(
    inertia_scale: float = 1.0,
    mass_scale: float = 1.0,
    com_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    gravity_scale: float = 1.0,
    tool_mass_kg: float = 0.0,
    tool_com_offset_m: Tuple[float, float, float] = (0.0, 0.0, -0.08),
    tool_size_m: Tuple[float, float, float] = (0.05, 0.05, 0.08),
) -> pin.Model:
    """Build the fixed-base AIRBOT model from the same deployed URDF chain."""
    source = ET.parse(str(ARM_URDF)).getroot()
    links = {item.get("name"): item for item in source.findall("link")}
    joints = {item.get("name"): item for item in source.findall("joint")}
    selected_links = [AIRBOT_ROOT_LINK]
    for joint_name in AIRBOT_JOINTS:
        selected_links.append(joints[joint_name].find("child").get("link"))

    robot = ET.Element("robot", {"name": "airbot_observer_validation"})
    for link_name in selected_links:
        link = copy.deepcopy(links[link_name])
        for geometry in list(link.findall("visual")) + list(
            link.findall("collision")
        ):
            link.remove(geometry)
        robot.append(link)
    for joint_name in AIRBOT_JOINTS:
        robot.append(copy.deepcopy(joints[joint_name]))
    model = pin.buildModelFromXML(ET.tostring(robot, encoding="unicode"))
    if inertia_scale <= 0.0:
        raise ValueError("inertia_scale must be positive")
    if mass_scale <= 0.0:
        raise ValueError("mass_scale must be positive")
    if gravity_scale <= 0.0:
        raise ValueError("gravity_scale must be positive")
    if tool_mass_kg < 0.0:
        raise ValueError("tool_mass_kg must be nonnegative")
    com_offset = np.asarray(com_offset_m, dtype=float)
    if com_offset.shape != (3,):
        raise ValueError("com_offset_m must contain three values")
    for joint_id in range(1, model.njoints):
        inertia = model.inertias[joint_id]
        model.inertias[joint_id] = pin.Inertia(
            inertia.mass * inertia_scale * mass_scale,
            inertia.lever + com_offset,
            inertia.inertia * inertia_scale,
        )
    model.gravity.linear = gravity_scale * model.gravity.linear
    if tool_mass_kg > 0.0:
        tool_com = np.asarray(tool_com_offset_m, dtype=float)
        tool_size = np.asarray(tool_size_m, dtype=float)
        if tool_com.shape != (3,) or tool_size.shape != (3,):
            raise ValueError("tool COM and size must contain three values")
        if np.any(tool_size <= 0.0):
            raise ValueError("tool dimensions must be positive")
        x, y, z = tool_size
        tool_inertia = tool_mass_kg / 12.0 * np.diag(
            [y * y + z * z, x * x + z * z, x * x + y * y]
        )
        terminal_joint = model.getJointId(AIRBOT_JOINTS[-1])
        model.inertias[terminal_joint] = (
            model.inertias[terminal_joint]
            + pin.Inertia(tool_mass_kg, tool_com, tool_inertia)
        )
    return model


def build_airbot_observer_model(
    output_path: Path,
    joint_damping_nm_s_rad: float = 0.0,
    joint_frictionloss_nm: float = 0.0,
    unmodeled_tool_mass_kg: float = 0.0,
) -> Path:
    """Build a dynamics-only fixed-base AIRBOT MJCF from the deployed URDF."""
    urdf_root = ET.parse(str(ARM_URDF)).getroot()
    links = {item.get("name"): item for item in urdf_root.findall("link")}
    joints = {item.get("name"): item for item in urdf_root.findall("joint")}

    root = ET.Element("mujoco", {"model": "airbot_observer_validation"})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": "0.001",
            "gravity": "0 0 -9.81",
            "integrator": "RK4",
            "solver": "Newton",
            "iterations": "50",
            "tolerance": "1e-10",
        },
    )
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "joint",
        {
            "damping": str(joint_damping_nm_s_rad),
            "frictionloss": str(joint_frictionloss_nm),
            "armature": "0",
        },
    )
    ET.SubElement(default, "geom", {"contype": "0", "conaffinity": "0"})

    worldbody = ET.SubElement(root, "worldbody")
    parent_body = ET.SubElement(worldbody, "body", {"name": AIRBOT_ROOT_LINK})
    _add_inertial(parent_body, links[AIRBOT_ROOT_LINK])
    ET.SubElement(
        parent_body,
        "geom",
        {"type": "cylinder", "size": "0.07 0.055", "rgba": "0.3 0.3 0.35 1"},
    )

    for joint_name in AIRBOT_JOINTS:
        source_joint = joints[joint_name]
        child_name = source_joint.find("child").get("link")
        child_body = ET.SubElement(
            parent_body,
            "body",
            {"name": child_name, **_origin_pose(source_joint.find("origin"))},
        )
        _add_inertial(child_body, links[child_name])
        axis = source_joint.find("axis")
        limit = source_joint.find("limit")
        ET.SubElement(
            child_body,
            "joint",
            {
                "name": joint_name,
                "type": "hinge",
                "axis": _numbers(
                    axis.get("xyz") if axis is not None else None,
                    "0 0 1",
                ),
                "limited": "true",
                "range": "{} {}".format(
                    limit.get("lower"), limit.get("upper")
                ),
            },
        )
        ET.SubElement(
            child_body,
            "geom",
            {
                "type": "sphere",
                "size": "0.035",
                "mass": "0",
                "rgba": "0.25 0.55 0.8 1",
            },
        )
        parent_body = child_body

    ET.SubElement(
        parent_body,
        "site",
        {"name": "ee_site", "size": "0.025", "rgba": "1 0.2 0.2 1"},
    )
    if unmodeled_tool_mass_kg > 0.0:
        tool = ET.SubElement(
            parent_body,
            "body",
            {"name": "unmodeled_tool", "pos": "0 0 -0.08"},
        )
        ET.SubElement(
            tool,
            "geom",
            {
                "type": "box",
                "size": "0.025 0.025 0.04",
                "mass": str(unmodeled_tool_mass_kg),
                "rgba": "0.8 0.5 0.2 1",
            },
        )

    actuator = ET.SubElement(root, "actuator")
    for joint_name in AIRBOT_JOINTS:
        effort = joints[joint_name].find("limit").get("effort")
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": joint_name,
                "joint": joint_name,
                "gear": "1",
                "ctrllimited": "true",
                "ctrlrange": "-{} {}".format(effort, effort),
            },
        )

    _indent(root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(
        str(output_path), encoding="utf-8", xml_declaration=True
    )
    return output_path


def _ids(model, object_type, names: Iterable[str]) -> np.ndarray:
    result = np.array(
        [mujoco.mj_name2id(model, object_type, name) for name in names],
        dtype=int,
    )
    if np.any(result < 0):
        raise KeyError("MuJoCo names not found: {}".format(tuple(names)))
    return result


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def run_validation_case(
    scenario: str,
    model_path: Path = AIRBOT_OBSERVER_MODEL,
    duration_s: float = 1.5,
    force_start_s: float = 0.4,
    measurement_start_s: float = 0.7,
    observer_gain: float = 100.0,
    pose_name: str = "nominal",
    perturbation: Optional[ObserverPerturbation] = None,
    force_duration_s: Optional[float] = None,
) -> Dict[str, object]:
    """Run one validation case and compare estimates with MuJoCo truth."""
    if scenario not in SCENARIOS:
        raise ValueError("Unknown AIRBOT observer scenario: " + scenario)
    if pose_name not in AIRBOT_POSES:
        raise ValueError("Unknown AIRBOT pose: " + pose_name)
    if measurement_start_s >= duration_s:
        raise ValueError("measurement_start_s must precede duration_s")
    perturbation = perturbation or ObserverPerturbation()
    model_path = build_airbot_observer_model(
        model_path,
        joint_damping_nm_s_rad=perturbation.joint_damping_nm_s_rad,
        joint_frictionloss_nm=perturbation.joint_frictionloss_nm,
        unmodeled_tool_mass_kg=perturbation.unmodeled_tool_mass_kg,
    )
    mj_model = load_model(model_path)
    mj_data = mujoco.MjData(mj_model)
    control_model = build_airbot_pinocchio_model(
        tool_mass_kg=perturbation.observer_tool_mass_kg
    )
    control_data = control_model.createData()
    pin_model = build_airbot_pinocchio_model(
        inertia_scale=perturbation.observer_inertia_scale,
        mass_scale=perturbation.observer_mass_scale,
        com_offset_m=perturbation.observer_com_offset_m,
        gravity_scale=perturbation.observer_gravity_scale,
        tool_mass_kg=perturbation.observer_tool_mass_kg,
    )
    pin_data = pin_model.createData()

    joint_ids = _ids(
        mj_model, mujoco.mjtObj.mjOBJ_JOINT, AIRBOT_JOINTS
    )
    actuator_ids = _ids(
        mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, AIRBOT_JOINTS
    )
    qpos_indices = mj_model.jnt_qposadr[joint_ids]
    dof_indices = mj_model.jnt_dofadr[joint_ids]
    if pin_model.nq != 6 or pin_model.nv != 6:
        raise ValueError("AIRBOT Pinocchio model must have six fixed-base DOFs")
    if tuple(pin_model.names[1:]) != AIRBOT_JOINTS:
        raise ValueError("MuJoCo and Pinocchio AIRBOT joint order differs")

    ee_body_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, AIRBOT_EE_FRAME
    )
    ee_site_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
    )
    q_home = AIRBOT_POSES[pose_name].copy()
    amplitude = np.array([0.20, 0.15, 0.16, 0.10, 0.10, 0.12])
    frequency = 0.35
    omega = 2.0 * np.pi * frequency
    mj_data.qpos[qpos_indices] = q_home
    mujoco.mj_forward(mj_model, mj_data)

    observer = GeneralizedMomentumObserver(
        pin_model,
        observer_gain=observer_gain,
        cutoff_frequency=perturbation.cutoff_frequency_hz,
        momentum_matrix_scale=perturbation.momentum_matrix_scale,
        bias_compensation_time_constant_s=(
            perturbation.bias_time_constant_s
            if perturbation.enable_bias_compensation
            else None
        ),
    )
    rng = np.random.default_rng(perturbation.random_seed)
    dt = float(mj_model.opt.timestep)
    kp = np.array([80.0, 80.0, 70.0, 8.0, 7.0, 5.0])
    kd = np.array([5.0, 5.0, 4.5, 0.7, 0.6, 0.5])
    effort = np.array([18.0, 18.0, 18.0, 3.0, 3.0, 3.0])
    applied_wrench = np.array([2.0, -1.5, 1.0, 0.08, -0.06, 0.05])
    if scenario == "zero_force":
        applied_wrench.fill(0.0)

    tau_estimates = []
    tau_truths = []
    wrench_estimates = []
    wrench_truths = []
    condition_numbers = []
    saturation_steps = 0
    saturation_counts = np.zeros(6, dtype=int)
    max_unclipped_torque = np.zeros(6)
    tracking_errors = []
    total_steps = 0
    response_time_90_s = None
    post_disturbance_settling_time_s = None
    max_active_truth_torque_norm = 0.0
    measurement_history = []
    delay_steps = int(round(perturbation.sample_delay_s / dt))

    while mj_data.time < duration_s:
        t = float(mj_data.time)
        q = mj_data.qpos[qpos_indices].copy()
        v = mj_data.qvel[dof_indices].copy()
        if scenario == "constant_wrench_motion":
            phase = omega * t
            q_ref = q_home + amplitude * (1.0 - np.cos(phase))
            v_ref = amplitude * omega * np.sin(phase)
            a_ref = amplitude * omega**2 * np.cos(phase)
        else:
            q_ref = q_home
            v_ref = np.zeros(6)
            a_ref = np.zeros(6)
        feedforward = pin.rnea(control_model, control_data, q, v, a_ref)
        tau_unclipped = feedforward + kp * (q_ref - q) + kd * (v_ref - v)
        tau_command = np.clip(tau_unclipped, -effort, effort)
        saturated = np.abs(tau_unclipped) > effort
        saturation_steps += int(np.any(saturated))
        saturation_counts += saturated
        max_unclipped_torque = np.maximum(
            max_unclipped_torque, np.abs(tau_unclipped)
        )
        total_steps += 1
        delivered_torque = perturbation.actuator_efficiency * np.sign(
            tau_command
        ) * np.maximum(
            np.abs(tau_command) - perturbation.actuator_deadzone_nm,
            0.0,
        )
        mj_data.ctrl[actuator_ids] = delivered_torque

        mj_data.qfrc_applied.fill(0.0)
        wrench_active = t >= force_start_s and scenario != "zero_force"
        if force_duration_s is not None:
            wrench_active = wrench_active and t < force_start_s + force_duration_s
        if wrench_active:
            mujoco.mj_applyFT(
                mj_model,
                mj_data,
                applied_wrench[:3],
                applied_wrench[3:],
                mj_data.site_xpos[ee_site_id],
                ee_body_id,
                mj_data.qfrc_applied,
            )
        tau_truth = mj_data.qfrc_applied[dof_indices].copy()
        wrench_truth = applied_wrench.copy() if wrench_active else np.zeros(6)

        mujoco.mj_step(mj_model, mj_data)
        q_next = mj_data.qpos[qpos_indices].copy()
        v_next = mj_data.qvel[dof_indices].copy()
        measured_q = q_next + rng.normal(
            0.0, perturbation.position_noise_std_rad, size=6
        )
        measured_v = v_next + rng.normal(
            0.0, perturbation.velocity_noise_std_rad_s, size=6
        )
        measured_tau = perturbation.torque_scale * tau_command + rng.normal(
            0.0, perturbation.torque_noise_std_nm, size=6
        )
        measurement_history.append((measured_q, measured_v, measured_tau))
        delayed_index = max(0, len(measurement_history) - delay_steps - 1)
        observer_q, observer_v, observer_tau = measurement_history[delayed_index]
        observer_dt = dt * (
            1.0
            + rng.uniform(
                -perturbation.dt_jitter_fraction,
                perturbation.dt_jitter_fraction,
            )
        )
        contact_phase = "carry" if wrench_active else "free_space"
        tau_estimate = observer.estimate_external_torque(
            observer_q,
            observer_v,
            observer_tau,
            observer_dt,
            contact_phase=contact_phase,
        )
        wrench_estimate = observer.joint_torque_to_cartesian_wrench(
            tau_estimate, observer_q, AIRBOT_EE_FRAME
        )
        if wrench_active and response_time_90_s is None:
            truth_norm = np.linalg.norm(tau_truth)
            max_active_truth_torque_norm = max(
                max_active_truth_torque_norm, truth_norm
            )
            if truth_norm > 0.0 and np.linalg.norm(
                tau_estimate - tau_truth
            ) <= 0.1 * truth_norm:
                response_time_90_s = max(
                    0.0, float(mj_data.time) - force_start_s
                )
        elif wrench_active:
            max_active_truth_torque_norm = max(
                max_active_truth_torque_norm, np.linalg.norm(tau_truth)
            )
        if (
            force_duration_s is not None
            and t >= force_start_s + force_duration_s
            and post_disturbance_settling_time_s is None
            and max_active_truth_torque_norm > 0.0
            and np.linalg.norm(tau_estimate)
            <= 0.1 * max_active_truth_torque_norm
        ):
            post_disturbance_settling_time_s = (
                t - force_start_s - force_duration_s
            )

        if mj_data.time >= measurement_start_s:
            tracking_errors.append(float(np.linalg.norm(q_ref - q_next)))
            tau_estimates.append(tau_estimate)
            tau_truths.append(tau_truth)
            wrench_estimates.append(wrench_estimate)
            wrench_truths.append(wrench_truth)
            frame_id = pin_model.getFrameId(AIRBOT_EE_FRAME)
            jacobian = pin.computeFrameJacobian(
                pin_model,
                pin_data,
                q_next,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            condition_numbers.append(float(np.linalg.cond(jacobian)))

    tau_estimates = np.asarray(tau_estimates)
    tau_truths = np.asarray(tau_truths)
    wrench_estimates = np.asarray(wrench_estimates)
    wrench_truths = np.asarray(wrench_truths)
    tau_error = tau_estimates - tau_truths
    wrench_error = wrench_estimates - wrench_truths
    active_mask = np.linalg.norm(wrench_truths[:, :3], axis=1) > 1e-12
    direction_samples = active_mask if np.any(active_mask) else np.ones(
        len(wrench_truths), dtype=bool
    )
    estimated_force_mean = np.mean(
        wrench_estimates[direction_samples, :3], axis=0
    )
    truth_force_mean = np.mean(wrench_truths[direction_samples, :3], axis=0)
    direction_error_deg = 0.0
    if np.linalg.norm(truth_force_mean) > 0.0:
        cosine = np.dot(estimated_force_mean, truth_force_mean) / (
            np.linalg.norm(estimated_force_mean) * np.linalg.norm(truth_force_mean)
        )
        direction_error_deg = float(
            np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        )
    active_force_rmse = 0.0
    force_peak_relative_error_percent = 0.0
    if np.any(active_mask):
        active_force_rmse = _rms(wrench_error[active_mask, :3])
        estimated_peak = float(
            np.max(np.linalg.norm(wrench_estimates[active_mask, :3], axis=1))
        )
        truth_peak = float(
            np.max(np.linalg.norm(wrench_truths[active_mask, :3], axis=1))
        )
        force_peak_relative_error_percent = 100.0 * abs(
            estimated_peak - truth_peak
        ) / truth_peak
    inactive_mask = ~active_mask
    inactive_force_rms = (
        _rms(wrench_estimates[inactive_mask, :3])
        if np.any(inactive_mask)
        else 0.0
    )
    return {
        "scenario": scenario,
        "force_duration_s": force_duration_s,
        "pose": pose_name,
        "perturbation": asdict(perturbation),
        "sample_count": int(len(tau_estimates)),
        "joint_torque_rmse_nm": _rms(tau_error),
        "joint_torque_bias_norm_nm": float(
            np.linalg.norm(np.mean(tau_error, axis=0))
        ),
        "force_rmse_n": _rms(wrench_error[:, :3]),
        "active_force_rmse_n": active_force_rmse,
        "inactive_force_rms_n": inactive_force_rms,
        "force_peak_relative_error_percent": force_peak_relative_error_percent,
        "force_bias_norm_n": float(
            np.linalg.norm(np.mean(wrench_error[:, :3], axis=0))
        ),
        "moment_rmse_nm": _rms(wrench_error[:, 3:]),
        "moment_bias_norm_nm": float(
            np.linalg.norm(np.mean(wrench_error[:, 3:], axis=0))
        ),
        "jacobian_condition_peak": float(np.max(condition_numbers)),
        "actuator_saturation_fraction": saturation_steps / max(total_steps, 1),
        "actuator_saturation_fractions": (
            saturation_counts / max(total_steps, 1)
        ).tolist(),
        "max_unclipped_torque_nm": max_unclipped_torque.tolist(),
        "tracking_error_peak_rad": float(np.max(tracking_errors)),
        "response_time_90_s": response_time_90_s,
        "post_disturbance_settling_time_s": post_disturbance_settling_time_s,
        "force_direction_error_deg": direction_error_deg,
        "estimated_wrench_mean": np.mean(wrench_estimates, axis=0).tolist(),
        "truth_wrench_mean": np.mean(wrench_truths, axis=0).tolist(),
        "estimated_joint_bias_nm": observer.estimated_bias.tolist(),
        "bias_compensation_enabled": bool(
            perturbation.enable_bias_compensation
        ),
    }


def run_validation_suite(**kwargs) -> Dict[str, Dict[str, object]]:
    return {scenario: run_validation_case(scenario, **kwargs) for scenario in SCENARIOS}


def _print_results(results: Dict[str, Dict[str, object]]) -> None:
    print(
        "scenario                  tau_rmse  force_rmse  moment_rmse  "
        "jac_cond  saturation"
    )
    for scenario, result in results.items():
        print(
            "{:<25} {:8.4f} {:10.4f} {:12.4f} {:9.1f} {:9.2f}%".format(
                scenario,
                result["joint_torque_rmse_nm"],
                result["force_rmse_n"],
                result["moment_rmse_nm"],
                result["jacobian_condition_peak"],
                100.0 * result["actuator_saturation_fraction"],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=AIRBOT_OBSERVER_MODEL)
    parser.add_argument("--scenario", choices=("all",) + SCENARIOS, default="all")
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--force-start", type=float, default=0.4)
    parser.add_argument("--measurement-start", type=float, default=0.7)
    parser.add_argument("--observer-gain", type=float, default=100.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    common = {
        "model_path": args.model,
        "duration_s": args.duration,
        "force_start_s": args.force_start,
        "measurement_start_s": args.measurement_start,
        "observer_gain": args.observer_gain,
    }
    if args.scenario == "all":
        results = run_validation_suite(**common)
    else:
        results = {args.scenario: run_validation_case(args.scenario, **common)}
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_results(results)


if __name__ == "__main__":
    main()
