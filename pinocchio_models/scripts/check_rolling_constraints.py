#!/usr/bin/env python3
"""Validate WF wheel rolling/no-slip contacts in the coupled model."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from check_coupled_dynamics import (
    elimination_rank,
    place_free_flyer,
    stack_configuration,
    stack_velocity,
    vector_norm,
)
from check_models import parser_path
from coupled_dynamics import CoupledDynamicsModel, matvec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finite-difference-step", type=float, default=1e-6)
    parser.add_argument("--ground-height", type=float, default=0.0)
    return parser.parse_args()


def load_models(pin, root: Path):
    robot_urdf = (
        root
        / "robot_description"
        / "pointfoot"
        / "WF_TRON1A"
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


def align_base_to_ground(pin, model, q, frame_ids, radius, ground_height):
    """Set the free-flyer height so both wheel contact points share the ground."""

    data = model.createData()
    q[2] = 0.0
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    center_heights = [data.oMf[frame_id].translation[2] for frame_id in frame_ids]
    q[2] = ground_height + radius - sum(center_heights) / len(center_heights)
    return q


def main() -> None:
    args = parse_args()
    if args.finite_difference_step <= 0.0:
        raise SystemExit("finite-difference-step must be positive")

    try:
        import pinocchio as pin
    except ImportError as exc:
        raise SystemExit("Pinocchio is not installed in this environment.") from exc

    root = Path(__file__).resolve().parents[1]
    robot_model, payload_model = load_models(pin, root)
    coupled = CoupledDynamicsModel(
        pin,
        robot_model,
        payload_model,
        support_mode="rolling_no_slip",
    )

    q1 = place_free_flyer(
        pin.neutral(robot_model), -0.7, 0.0, 0.8, 0.0
    )
    q2 = place_free_flyer(
        pin.neutral(robot_model), 0.7, 0.0, 0.8, math.pi
    )
    q1 = align_base_to_ground(
        pin,
        robot_model,
        q1,
        coupled.support_frame_ids,
        coupled.support_contact_radius,
        args.ground_height,
    )
    q2 = align_base_to_ground(
        pin,
        robot_model,
        q2,
        coupled.support_frame_ids,
        coupled.support_contact_radius,
        args.ground_height,
    )
    qp = place_free_flyer(
        pin.neutral(payload_model), 0.0, 0.0, 1.04, 0.0
    )
    q = stack_configuration(q1, q2, qp)

    v1 = np.zeros(robot_model.nv)
    v2 = np.zeros(robot_model.nv)
    vp = np.zeros(payload_model.nv)
    for index in range(robot_model.nv):
        v1[index] = 0.002 * math.sin(index + 0.7)
        v2[index] = 0.002 * math.cos(index + 0.4)
    for index in range(payload_model.nv):
        vp[index] = 0.002 * math.sin(index + 0.2)
    v = stack_velocity(v1, v2, vp)

    terms = coupled.evaluate(q, v)
    contact_positions = (
        coupled.support_contact_position(
            coupled.robot_data_1, coupled.support_frame_ids[0]
        ),
        coupled.support_contact_position(
            coupled.robot_data_1, coupled.support_frame_ids[1]
        ),
        coupled.support_contact_position(
            coupled.robot_data_2, coupled.support_frame_ids[0]
        ),
        coupled.support_contact_position(
            coupled.robot_data_2, coupled.support_frame_ids[1]
        ),
    )

    q1_plus = pin.integrate(robot_model, q1, args.finite_difference_step * v1)
    q2_plus = pin.integrate(robot_model, q2, args.finite_difference_step * v2)
    qp_plus = pin.integrate(payload_model, qp, args.finite_difference_step * vp)
    q_plus = stack_configuration(q1_plus, q2_plus, qp_plus)
    terms_plus = coupled.evaluate(q_plus, v)
    finite_difference_jdot = (
        terms_plus.contact_jacobian - terms.contact_jacobian
    ) / args.finite_difference_step
    jdot_error = vector_norm(
        terms.contact_jacobian_dot_velocity
        - matvec(finite_difference_jdot, v)
    )

    zero_torque = np.zeros(coupled.nu)
    solved_acceleration, solved_contact_wrench = (
        coupled.solve_constrained_dynamics(terms, zero_torque)
    )
    solved_support_forces = np.concatenate(
        tuple(
            matvec(
                terms.support_force_bases[index],
                solved_contact_wrench[3 * index : 3 * index + 3],
            )
            for index in range(len(terms.support_force_bases))
        )
    )
    dynamics_residual = coupled.dynamics_residual(
        terms,
        solved_acceleration,
        zero_torque,
        solved_contact_wrench,
    )
    acceleration_residual = coupled.acceleration_constraint_residual(
        terms, solved_acceleration
    )
    rank = elimination_rank(terms.contact_jacobian)
    ground_error = max(
        abs(position[2] - args.ground_height) for position in contact_positions
    )

    print("support mode: rolling_no_slip")
    print(f"wheel radius: {coupled.support_contact_radius:.6g} m")
    print(f"wheel contact positions: {contact_positions}")
    print(f"ground height error: {ground_error:.6g} m")
    print(f"Jc shape: {terms.contact_jacobian.shape}")
    print(f"contact Jacobian rank: {rank}")
    print(f"Jdot*v finite-difference error: {jdot_error:.6g}")
    print(f"solved support forces (world frame): {solved_support_forces}")
    print(f"solved grasp wrenches: {solved_contact_wrench[12:]}")
    print(f"KKT dynamics residual norm: {vector_norm(dynamics_residual):.6g}")
    print(
        "KKT acceleration-constraint residual norm: "
        f"{vector_norm(acceleration_residual):.6g}"
    )

    if ground_error > 1e-8:
        raise RuntimeError("Wheel contact points are not on the configured ground")
    if rank < coupled.contact_wrench_dim:
        raise RuntimeError("Rolling/grasp contact Jacobian is rank deficient")
    if jdot_error > 5e-4:
        raise RuntimeError("Rolling Jdot*v finite-difference check failed")
    if vector_norm(dynamics_residual) > 1e-7:
        raise RuntimeError("Rolling KKT dynamics residual is too large")
    if vector_norm(acceleration_residual) > 1e-7:
        raise RuntimeError("Rolling KKT acceleration residual is too large")
    print("rolling constraint check passed")


if __name__ == "__main__":
    main()
