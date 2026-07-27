#!/usr/bin/env python3
"""Check constrained inverse dynamics and URDF actuator effort limits."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from check_coupled_dynamics import (
    elimination_rank,
    place_free_flyer,
    stack_configuration,
    vector_norm,
)
from check_models import parser_path
from coupled_dynamics import CoupledDynamicsModel, matmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-type",
        choices=("WF_TRON1A", "SF_TRON1A"),
        default="WF_TRON1A",
    )
    parser.add_argument(
        "--support-mode",
        choices=("fixed_3d_position", "rolling_no_slip"),
        default="fixed_3d_position",
    )
    parser.add_argument(
        "--payload-height", type=float, default=1.04
    )
    return parser.parse_args()


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
    if args.support_mode == "rolling_no_slip" and args.robot_type != "WF_TRON1A":
        raise SystemExit("rolling_no_slip is only available for WF_TRON1A")

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
        support_mode=args.support_mode,
    )

    q1 = place_free_flyer(
        pin.neutral(robot_model), -0.7, 0.0, 0.8, 0.0
    )
    q2 = place_free_flyer(
        pin.neutral(robot_model), 0.7, 0.0, 0.8, math.pi
    )
    qp = place_free_flyer(
        pin.neutral(payload_model), 0.0, 0.0, args.payload_height, 0.0
    )
    q = stack_configuration(q1, q2, qp)
    v = np.zeros(coupled.nv)
    acceleration = np.zeros(coupled.nv)
    terms = coupled.evaluate(q, v)

    actuator_contact_map = np.hstack(
        (terms.actuation_matrix, terms.contact_jacobian.T)
    )
    map_rank = elimination_rank(actuator_contact_map)
    actuated_torque, contact_wrench = (
        coupled.solve_minimum_norm_inverse_dynamics(terms, acceleration)
    )
    support_forces = np.concatenate(
        tuple(
            np.array(
                [
                    sum(
                        terms.support_force_bases[index][row, column]
                        * contact_wrench[3 * index + column]
                        for column in range(3)
                    )
                    for row in range(3)
                ]
            )
            for index in range(len(terms.support_force_bases))
        )
    )
    residual = coupled.dynamics_residual(
        terms,
        acceleration,
        actuated_torque,
        contact_wrench,
    )

    effort_limits = []
    effort_ratios = []
    for velocity_index in coupled.robot_actuated_velocity_indices:
        limit = float(robot_model.effortLimit[velocity_index])
        effort_limits.append(limit)
        torque_value = float(
            actuated_torque[
                coupled.robot_actuated_velocity_indices.index(velocity_index)
            ]
        )
        if limit > 0.0 and math.isfinite(limit):
            effort_ratios.append(abs(torque_value) / limit)
    max_effort_ratio = max(effort_ratios) if effort_ratios else 0.0

    print(f"robot_type: {args.robot_type}")
    print(f"support mode: {args.support_mode}")
    print(f"prescribed acceleration norm: {vector_norm(acceleration):.6g}")
    print(f"actuator/contact map shape: {actuator_contact_map.shape}")
    print(f"actuator/contact map rank: {map_rank}")
    print(f"robot_1 torques: {actuated_torque[:14]}")
    print(f"robot_2 torques: {actuated_torque[14:]}")
    print(f"support forces (world frame): {support_forces}")
    print(f"grasp wrenches: {contact_wrench[12:]}")
    print(f"effort limits: {effort_limits}")
    print(f"maximum effort ratio: {max_effort_ratio:.6g}")
    print(f"inverse dynamics residual norm: {vector_norm(residual):.6g}")

    if map_rank < coupled.nv:
        raise RuntimeError("Actuator/contact map cannot span all generalized forces")
    if vector_norm(residual) > 1e-7:
        raise RuntimeError("Inverse dynamics residual is too large")
    print("constrained inverse dynamics check passed")


if __name__ == "__main__":
    main()
