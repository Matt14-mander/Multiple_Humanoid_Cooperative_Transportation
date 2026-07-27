#!/usr/bin/env python3
"""Validate alternating leg stance/swing phases with contact switching."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from animate_leg_supported_cooperative_carry import (
    joint_indices,
    leg_joint_names,
    solve_leg_position_ik,
    support_position_error,
)
from animate_cooperative_carry import place_free_flyer
from check_coupled_dynamics import elimination_rank, vector_norm
from check_models import parser_path
from coupled_dynamics import CoupledDynamicsModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-type",
        choices=("WF_TRON1A", "SF_TRON1A"),
        default="WF_TRON1A",
    )
    parser.add_argument("--steps-per-phase", type=int, default=8)
    parser.add_argument("--step-length", type=float, default=0.08)
    parser.add_argument("--step-height", type=float, default=0.06)
    return parser.parse_args()


def stack_configuration(q1, q2, qp):
    return np.concatenate((q1, q2, qp))


def load_models(pin, root: Path, robot_type: str):
    robot_urdf = (
        root
        / "robot_description"
        / "pointfoot"
        / robot_type
        / "urdf"
        / "robot_with_arm.urdf"
    )
    payload_urdf = root / "payload" / "payload_with_handles.urdf"
    robot_model = pin.buildModelFromUrdf(
        parser_path(robot_urdf), pin.JointModelFreeFlyer()
    )
    payload_model = pin.buildModelFromUrdf(
        parser_path(payload_urdf), pin.JointModelFreeFlyer()
    )
    return robot_model, payload_model


def main() -> None:
    args = parse_args()
    if args.steps_per_phase < 2:
        raise SystemExit("steps-per-phase must be at least 2")
    if args.step_length <= 0.0 or args.step_height <= 0.0:
        raise SystemExit("step-length and step-height must be positive")

    try:
        import pinocchio as pin
    except ImportError as exc:
        raise SystemExit("Pinocchio is not installed in this environment.") from exc

    root = Path(__file__).resolve().parents[1]
    robot_model, payload_model = load_models(pin, root, args.robot_type)
    coupled = CoupledDynamicsModel(
        pin,
        robot_model,
        payload_model,
        support_mode="fixed_3d_position",
    )
    q_robot_1 = place_free_flyer(
        pin.neutral(robot_model), -0.7, 0.0, 0.8, 0.0
    )
    q_robot_2 = place_free_flyer(
        pin.neutral(robot_model), 0.7, 0.0, 0.8, math.pi
    )
    q_payload = place_free_flyer(
        pin.neutral(payload_model), 0.0, 0.0, 1.04, 0.0
    )
    robot_data_1 = robot_model.createData()
    robot_data_2 = robot_model.createData()
    pin.forwardKinematics(robot_model, robot_data_1, q_robot_1)
    pin.updateFramePlacements(robot_model, robot_data_1)
    pin.forwardKinematics(robot_model, robot_data_2, q_robot_2)
    pin.updateFramePlacements(robot_model, robot_data_2)
    stance_targets_1 = [
        robot_data_1.oMf[frame_id].translation.copy()
        for frame_id in coupled.support_frame_ids
    ]
    stance_targets_2 = [
        robot_data_2.oMf[frame_id].translation.copy()
        for frame_id in coupled.support_frame_ids
    ]
    initial_leg_names = leg_joint_names(args.robot_type)
    leg_q_indices, leg_v_indices = joint_indices(
        robot_model, initial_leg_names
    )
    initial_leg_q = np.concatenate(
        (q_robot_1[leg_q_indices], q_robot_2[leg_q_indices])
    )

    phases = (
        ("both stance", None, (True, True, True, True)),
        ("robot 1 left swing", (0, 0), (False, True, True, True)),
        ("both stance after robot 1 touchdown", None, (True, True, True, True)),
        ("robot 2 right swing", (1, 1), (True, True, True, False)),
        ("both stance after robot 2 touchdown", None, (True, True, True, True)),
    )
    max_stance_error = 0.0
    max_swing_error = 0.0
    max_leg_motion = 0.0
    minimum_rank = coupled.contact_wrench_dim
    maximum_dynamics_error = 0.0
    maximum_acceleration_error = 0.0
    phase_rows = []

    for phase_index, (phase_name, swing_spec, active_mask) in enumerate(phases):
        if phase_index == 2:
            stance_targets_1[0] = stance_targets_1[0] + np.array(
                (args.step_length, 0.0, 0.0)
            )
        if phase_index == 4:
            stance_targets_2[1] = stance_targets_2[1] + np.array(
                (args.step_length, 0.0, 0.0)
            )
        for local_step in range(args.steps_per_phase):
            alpha = local_step / (args.steps_per_phase - 1)
            targets_1 = [target.copy() for target in stance_targets_1]
            targets_2 = [target.copy() for target in stance_targets_2]
            if swing_spec is not None:
                swing_alpha = alpha
                offset = np.array(
                    (
                        args.step_length * swing_alpha,
                        0.0,
                        args.step_height * math.sin(math.pi * swing_alpha),
                    )
                )
                if swing_spec[0] == 0:
                    targets_1[swing_spec[1]] = (
                        stance_targets_1[swing_spec[1]] + offset
                    )
                else:
                    targets_2[swing_spec[1]] = (
                        stance_targets_2[swing_spec[1]] + offset
                    )

            q_robot_1, _ = solve_leg_position_ik(
                pin,
                robot_model,
                robot_data_1,
                q_robot_1,
                coupled.support_frame_ids,
                tuple(targets_1),
                leg_q_indices,
                leg_v_indices,
            )
            q_robot_2, _ = solve_leg_position_ik(
                pin,
                robot_model,
                robot_data_2,
                q_robot_2,
                coupled.support_frame_ids,
                tuple(targets_2),
                leg_q_indices,
                leg_v_indices,
            )

            error_1 = support_position_error(
                pin,
                robot_model,
                robot_data_1,
                q_robot_1,
                coupled.support_frame_ids,
                tuple(targets_1),
            )
            error_2 = support_position_error(
                pin,
                robot_model,
                robot_data_2,
                q_robot_2,
                coupled.support_frame_ids,
                tuple(targets_2),
            )
            target_errors = [
                vector_norm(error_1[:3]),
                vector_norm(error_1[3:]),
                vector_norm(error_2[:3]),
                vector_norm(error_2[3:]),
            ]
            for index, error in enumerate(target_errors):
                if active_mask[index]:
                    max_stance_error = max(max_stance_error, error)
                else:
                    max_swing_error = max(max_swing_error, error)

            current_leg_q = np.concatenate(
                (q_robot_1[leg_q_indices], q_robot_2[leg_q_indices])
            )
            max_leg_motion = max(
                max_leg_motion,
                vector_norm(current_leg_q - initial_leg_q),
            )
            coupled.set_support_active_mask(active_mask)
            q = stack_configuration(q_robot_1, q_robot_2, q_payload)
            terms = coupled.evaluate(q, np.zeros(coupled.nv))
            acceleration, wrench = coupled.solve_constrained_dynamics(
                terms, np.zeros(coupled.nu)
            )
            dynamics_error = coupled.dynamics_residual(
                terms, acceleration, np.zeros(coupled.nu), wrench
            )
            acceleration_error = coupled.acceleration_constraint_residual(
                terms, acceleration
            )
            rank = elimination_rank(terms.contact_jacobian)
            minimum_rank = min(minimum_rank, rank)
            maximum_dynamics_error = max(
                maximum_dynamics_error, vector_norm(dynamics_error)
            )
            maximum_acceleration_error = max(
                maximum_acceleration_error, vector_norm(acceleration_error)
            )
        phase_rows.append((phase_name, active_mask, terms.contact_jacobian.shape[0]))

    print(f"robot_type: {args.robot_type}")
    print(f"leg joints: {initial_leg_names}")
    print("phase contact rows:")
    for phase_name, active_mask, rows in phase_rows:
        print(f"  {phase_name}: mask={active_mask}, rows={rows}")
    print(f"maximum active stance position error: {max_stance_error:.6g} m")
    print(f"maximum swing target error: {max_swing_error:.6g} m")
    print(f"maximum leg joint motion norm: {max_leg_motion:.6g}")
    print(f"minimum contact Jacobian rank: {minimum_rank}")
    print(f"maximum KKT dynamics residual: {maximum_dynamics_error:.6g}")
    print(
        "maximum KKT acceleration residual: "
        f"{maximum_acceleration_error:.6g}"
    )

    if max_stance_error > 1e-5 or max_swing_error > 1e-5:
        raise RuntimeError("Leg stance/swing IK error is too large")
    if minimum_rank < 18:
        raise RuntimeError("A leg contact mode became rank deficient")
    if maximum_dynamics_error > 1e-7:
        raise RuntimeError("Leg gait KKT dynamics residual is too large")
    if maximum_acceleration_error > 1e-7:
        raise RuntimeError("Leg gait KKT acceleration residual is too large")
    print("leg stance/swing contact-switching check passed")


if __name__ == "__main__":
    main()
