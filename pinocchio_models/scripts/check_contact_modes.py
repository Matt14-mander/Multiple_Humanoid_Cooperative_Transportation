#!/usr/bin/env python3
"""Validate friction, unilateral-contact diagnostics, and contact switching."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from check_coupled_dynamics import elimination_rank, place_free_flyer, stack_configuration, vector_norm
from check_models import parser_path
from check_rolling_constraints import align_base_to_ground
from contact_validation import classify_support_contacts
from coupled_dynamics import CoupledDynamicsModel, matvec


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


def physical_support_forces(coupled, terms, contact_wrench):
    """Convert rolling multiplier coordinates to world-frame forces."""

    full_forces = np.zeros(12)
    support_wrench = contact_wrench[: coupled.support_constraint_dim]
    for index, spec in enumerate(coupled.active_support_specs):
        force = matvec(
            terms.support_force_bases[index],
            support_wrench[3 * index : 3 * index + 3],
        )
        full_index = coupled.support_contact_specs.index(spec)
        full_forces[3 * full_index : 3 * full_index + 3] = force
    return full_forces


def support_names(coupled) -> tuple[str, ...]:
    return tuple(
        "robot_{}/{}".format(robot_index + 1, coupled.support_frame_names[frame_index])
        for robot_index, frame_index in coupled.support_contact_specs
    )


def build_initial_state(pin, robot_model, payload_model, coupled):
    q1 = align_base_to_ground(
        pin,
        robot_model,
        place_free_flyer(pin.neutral(robot_model), -0.7, 0.0, 0.8, 0.0),
        coupled.support_frame_ids,
        coupled.support_contact_radius,
        0.0,
    )
    q2 = align_base_to_ground(
        pin,
        robot_model,
        place_free_flyer(pin.neutral(robot_model), 0.7, 0.0, 0.8, math.pi),
        coupled.support_frame_ids,
        coupled.support_contact_radius,
        0.0,
    )
    qp = place_free_flyer(
        pin.neutral(payload_model), 0.0, 0.0, 1.04, 0.0
    )
    return stack_configuration(q1, q2, qp), np.zeros(coupled.nv)


def run_case(
    coupled,
    q,
    v,
    label: str,
    active_mask: tuple[bool, bool, bool, bool],
    friction_coefficient: float,
    external_generalized_force: np.ndarray,
):
    coupled.set_support_active_mask(active_mask)
    terms = coupled.evaluate(q, v)
    acceleration, contact_wrench = coupled.solve_constrained_dynamics(
        terms,
        np.zeros(coupled.nu),
        external_generalized_force=external_generalized_force,
    )
    dynamics_error = coupled.dynamics_residual(
        terms,
        acceleration,
        np.zeros(coupled.nu),
        contact_wrench,
        external_generalized_force=external_generalized_force,
    )
    acceleration_error = coupled.acceleration_constraint_residual(
        terms, acceleration
    )
    forces = physical_support_forces(coupled, terms, contact_wrench)
    statuses = classify_support_contacts(
        forces,
        support_names(coupled),
        active_mask,
        coupled.support_ground_normal,
        friction_coefficient,
    )
    modes = tuple(status.mode for status in statuses)
    active_normals = tuple(
        status.normal_force for status in statuses if status.active
    )
    active_ratios = tuple(
        status.friction_ratio for status in statuses if status.active
    )
    print(f"case: {label}")
    print(f"  active mask: {active_mask}")
    print(f"  Jc shape/rank: {terms.contact_jacobian.shape}/{elimination_rank(terms.contact_jacobian)}")
    print(f"  contact modes: {modes}")
    print(f"  active normal forces: {active_normals}")
    print(f"  active friction ratios: {active_ratios}")
    print(f"  dynamics residual: {vector_norm(dynamics_error):.6g}")
    print(f"  acceleration residual: {vector_norm(acceleration_error):.6g}")
    if elimination_rank(terms.contact_jacobian) < coupled.contact_wrench_dim:
        raise RuntimeError(f"{label}: contact Jacobian is rank deficient")
    if vector_norm(dynamics_error) > 1e-7:
        raise RuntimeError(f"{label}: dynamics residual is too large")
    if vector_norm(acceleration_error) > 1e-7:
        raise RuntimeError(f"{label}: acceleration residual is too large")
    return statuses


def main() -> None:
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
    q, v = build_initial_state(pin, robot_model, payload_model, coupled)
    zero_external = np.zeros(coupled.nv)
    robot_1_lift_force = np.zeros(coupled.nv)
    robot_1_lift_force[2] = 4000.0

    nominal = run_case(
        coupled,
        q,
        v,
        "nominal sticking contact",
        (True, True, True, True),
        0.6,
        zero_external,
    )
    if any(status.mode != "stick" for status in nominal if status.active):
        raise RuntimeError("Nominal contacts should be inside the friction cone")

    low_friction = run_case(
        coupled,
        q,
        v,
        "low-friction slip diagnostic",
        (True, True, True, True),
        0.01,
        zero_external,
    )
    if not any(status.mode == "slip" for status in low_friction):
        raise RuntimeError("Low-friction case did not trigger slip detection")

    lift_off = run_case(
        coupled,
        q,
        v,
        "unilateral lift-off diagnostic",
        (True, True, True, True),
        0.6,
        robot_1_lift_force,
    )
    if not any(status.mode == "lift_off" for status in lift_off):
        raise RuntimeError("Lift-off case did not detect a negative normal force")

    for label, mask in (
        ("robot 1 left swing", (False, True, True, True)),
        ("robot 2 right swing", (True, True, True, False)),
        ("robot 1 double lift-off", (False, False, True, True)),
        ("all contacts restored", (True, True, True, True)),
    ):
        run_case(coupled, q, v, label, mask, 0.6, zero_external)

    coupled.set_support_active_mask((True, True, True, True))
    initial_terms = coupled.evaluate(q, v)
    initial_acceleration, initial_wrench = coupled.solve_constrained_dynamics(
        initial_terms,
        np.zeros(coupled.nu),
        external_generalized_force=robot_1_lift_force,
    )
    initial_forces = physical_support_forces(
        coupled, initial_terms, initial_wrench
    )
    initial_statuses = classify_support_contacts(
        initial_forces,
        support_names(coupled),
        (True, True, True, True),
        coupled.support_ground_normal,
        0.6,
    )
    unilateral_mask = tuple(
        status.normal_force >= 0.0 for status in initial_statuses
    )
    if not any(unilateral_mask):
        raise RuntimeError("The active-set update would remove every contact")
    print(f"active-set update from lift-off: {unilateral_mask}")
    final_statuses = run_case(
        coupled,
        q,
        v,
        "after unilateral active-set update",
        unilateral_mask,
        0.6,
        robot_1_lift_force,
    )
    if any(
        status.active and status.normal_force < -1e-8
        for status in final_statuses
    ):
        raise RuntimeError("Active-set update left a negative active normal force")
    del initial_acceleration
    print("contact friction/unilateral/switching check passed")


if __name__ == "__main__":
    main()
