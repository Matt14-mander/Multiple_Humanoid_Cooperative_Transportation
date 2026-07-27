#!/usr/bin/env python3
"""Validate the stacked two-robot rigid-grasp dynamics construction."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from check_models import parser_path
from coupled_dynamics import CoupledDynamicsModel, matvec


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
    parser.add_argument("--finite-difference-step", type=float, default=1e-6)
    return parser.parse_args()


def place_free_flyer(q, x: float, y: float, z: float, yaw: float):
    q[:3] = [x, y, z]
    q[3:7] = [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
    return q


def vector_norm(vector: np.ndarray) -> float:
    return float(np.sqrt(sum(value * value for value in vector)))


def elimination_rank(matrix: np.ndarray, tolerance: float = 1e-8) -> int:
    work = np.array(matrix, dtype=float, copy=True)
    rows, columns = work.shape
    scale = max(float(np.max(np.abs(work))), 1.0)
    rank = 0
    for column in range(columns):
        if rank >= rows:
            break
        pivot_row = rank + int(np.argmax(np.abs(work[rank:, column])))
        if abs(work[pivot_row, column]) <= tolerance * scale:
            continue
        if pivot_row != rank:
            work[[rank, pivot_row]] = work[[pivot_row, rank]]
        work[rank] /= work[rank, column]
        for row in range(rank + 1, rows):
            factor = work[row, column]
            if abs(factor) > tolerance:
                work[row] -= factor * work[rank]
        rank += 1
    return rank


def stack_configuration(q1, q2, qp):
    return np.concatenate((q1, q2, qp))


def stack_velocity(v1, v2, vp):
    return np.concatenate((v1, v2, vp))


def main() -> None:
    try:
        import pinocchio as pin
    except ImportError as exc:
        raise SystemExit("Pinocchio is not installed in this environment.") from exc

    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    robot_urdf = (
        root
        / "robot_description"
        / "pointfoot"
        / args.robot_type
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
    coupled = CoupledDynamicsModel(
        pin,
        robot_model,
        payload_model,
        support_mode=args.support_mode,
    )

    q1 = place_free_flyer(pin.neutral(robot_model), -0.7, 0.0, 0.8, 0.0)
    q2 = place_free_flyer(
        pin.neutral(robot_model), 0.7, 0.0, 0.8, math.pi
    )
    qp = place_free_flyer(
        pin.neutral(payload_model), 0.0, 0.0, 1.04, 0.0
    )
    q = stack_configuration(q1, q2, qp)

    v1 = np.zeros(robot_model.nv)
    v2 = np.zeros(robot_model.nv)
    vp = np.zeros(payload_model.nv)
    for index in range(robot_model.nv):
        v1[index] = 0.003 * math.sin(index + 1.0)
        v2[index] = 0.003 * math.cos(index + 1.0)
    for index in range(payload_model.nv):
        vp[index] = 0.003 * math.sin(index + 0.5)
    v = stack_velocity(v1, v2, vp)

    terms = coupled.evaluate(q, v)
    tau = np.zeros(coupled.nu)
    lambda_contact = np.zeros(coupled.contact_wrench_dim)
    acceleration = np.zeros(coupled.nv)
    kkt_matrix, kkt_rhs = coupled.assemble_kkt(terms, tau)
    solved_acceleration, solved_contact_wrench = (
        coupled.solve_constrained_dynamics(terms, tau)
    )

    q1_plus = pin.integrate(robot_model, q1, args.finite_difference_step * v1)
    q2_plus = pin.integrate(robot_model, q2, args.finite_difference_step * v2)
    qp_plus = pin.integrate(payload_model, qp, args.finite_difference_step * vp)
    q_plus = stack_configuration(q1_plus, q2_plus, qp_plus)
    terms_plus = coupled.evaluate(q_plus, v)
    finite_difference_jdot = (
        terms_plus.contact_jacobian - terms.contact_jacobian
    ) / args.finite_difference_step
    finite_difference_jdot_v = matvec(finite_difference_jdot, v)

    mass_symmetry_error = float(
        np.max(np.abs(terms.mass_matrix - terms.mass_matrix.T))
    )
    jdot_error = vector_norm(
        terms.contact_jacobian_dot_velocity - finite_difference_jdot_v
    )
    kinematic_rank = elimination_rank(terms.contact_jacobian)
    dynamics_residual = coupled.dynamics_residual(
        terms, acceleration, tau, lambda_contact
    )
    acceleration_residual = coupled.acceleration_constraint_residual(
        terms, acceleration
    )
    solved_dynamics_residual = coupled.dynamics_residual(
        terms, solved_acceleration, tau, solved_contact_wrench
    )
    solved_acceleration_residual = coupled.acceleration_constraint_residual(
        terms, solved_acceleration
    )

    print(f"robot_type: {args.robot_type}")
    print("payload_model: payload_with_handles")
    print(f"support mode: {coupled.support_mode}")
    print(f"stacked configuration dimension nq: {coupled.nq}")
    print(f"stacked velocity dimension nv: {coupled.nv}")
    print(f"actuated torque dimension nu: {coupled.nu}")
    print(f"support frames: {coupled.support_frame_names}")
    print(f"support constraint dimension: {coupled.support_constraint_dim}")
    print(f"grasp constraint dimension: {coupled.grasp_constraint_dim}")
    print(f"contact wrench dimension: {coupled.contact_wrench_dim}")
    print(f"M shape: {terms.mass_matrix.shape}")
    print(f"h shape: {terms.nonlinear_effects.shape}")
    print(f"B shape: {terms.actuation_matrix.shape}")
    print(f"Jc shape: {terms.contact_jacobian.shape}")
    print(f"KKT shape: {kkt_matrix.shape}, rhs shape: {kkt_rhs.shape}")
    print(f"contact Jacobian rank: {kinematic_rank}")
    print(f"contact wrench order: {coupled.contact_names}")
    print(f"mass matrix symmetry error: {mass_symmetry_error:.6g}")
    print(f"Jdot*v finite-difference error: {jdot_error:.6g}")
    print(f"zero-input dynamics residual norm: {vector_norm(dynamics_residual):.6g}")
    print(
        f"zero-acceleration constraint residual norm: "
        f"{vector_norm(acceleration_residual):.6g}"
    )
    print(f"solved acceleration norm: {vector_norm(solved_acceleration):.6g}")
    print(f"solved support wrench: {solved_contact_wrench[:coupled.support_constraint_dim]}")
    print(f"solved grasp wrench: {solved_contact_wrench[coupled.support_constraint_dim:]}")
    print(
        f"solved dynamics residual norm: "
        f"{vector_norm(solved_dynamics_residual):.6g}"
    )
    print(
        f"solved acceleration constraint residual norm: "
        f"{vector_norm(solved_acceleration_residual):.6g}"
    )

    if terms.mass_matrix.shape != (coupled.nv, coupled.nv):
        raise RuntimeError("Unexpected mass matrix shape")
    if terms.contact_jacobian.shape != (
        coupled.contact_wrench_dim,
        coupled.nv,
    ):
        raise RuntimeError("Unexpected contact Jacobian shape")
    if kinematic_rank < coupled.contact_wrench_dim:
        raise RuntimeError("The support/grasp constraints are rank deficient")
    if mass_symmetry_error > 1e-10:
        raise RuntimeError("The stacked mass matrix is not symmetric")
    if jdot_error > 5e-4:
        raise RuntimeError("Jdot*v finite-difference check failed")
    if kkt_matrix.shape != (
        coupled.nv + coupled.contact_wrench_dim,
        coupled.nv + coupled.contact_wrench_dim,
    ):
        raise RuntimeError("Unexpected KKT system shape")
    if vector_norm(solved_dynamics_residual) > 1e-7:
        raise RuntimeError("KKT dynamics solve did not satisfy the dynamics")
    if vector_norm(solved_acceleration_residual) > 1e-7:
        raise RuntimeError(
            "KKT dynamics solve did not satisfy the acceleration constraints"
        )
    print("coupled dynamics construction check passed")


if __name__ == "__main__":
    main()
