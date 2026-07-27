#!/usr/bin/env python3
"""Integrate a rolling dual-robot model under joint torque feedback."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from check_coupled_dynamics import place_free_flyer, stack_configuration
from check_models import parser_path
from check_rolling_constraints import align_base_to_ground
from contact_validation import classify_support_contacts, vector_norm
from coupled_dynamics import CoupledDynamicsModel, matvec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--time-step", type=float, default=0.005)
    parser.add_argument("--wheel-speed", type=float, default=0.2)
    parser.add_argument("--kp", type=float, default=20.0)
    parser.add_argument("--kd", type=float, default=4.0)
    parser.add_argument("--ground-position-gain", type=float, default=80.0)
    parser.add_argument("--ground-velocity-gain", type=float, default=16.0)
    parser.add_argument("--friction-coefficient", type=float, default=0.6)
    return parser.parse_args()


def joint_qv_indices(model):
    pairs = []
    for joint_id in range(2, model.njoints):
        if int(model.nvs[joint_id]) != 1 or int(model.nqs[joint_id]) != 1:
            continue
        pairs.append((int(model.idx_qs[joint_id]), int(model.idx_vs[joint_id])))
    return pairs


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


def main() -> None:
    args = parse_args()
    if args.duration <= 0.0 or args.time_step <= 0.0:
        raise SystemExit("duration and time-step must be positive")
    if args.friction_coefficient < 0.0:
        raise SystemExit("friction-coefficient must be nonnegative")

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
    robot_pairs = joint_qv_indices(robot_model)
    if len(robot_pairs) != len(coupled.robot_actuated_velocity_indices):
        raise RuntimeError("Unexpected actuated joint layout")
    wheel_joint_ids = (
        robot_model.getJointId("wheel_L_Joint"),
        robot_model.getJointId("wheel_R_Joint"),
    )
    wheel_q_indices = tuple(int(robot_model.idx_qs[joint_id]) for joint_id in wheel_joint_ids)

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
        0.0,
    )
    q2 = align_base_to_ground(
        pin,
        robot_model,
        q2,
        coupled.support_frame_ids,
        coupled.support_contact_radius,
        0.0,
    )
    qp = place_free_flyer(
        pin.neutral(payload_model), 0.0, 0.0, 1.04, 0.0
    )
    q = stack_configuration(q1, q2, qp)
    v = np.zeros(coupled.nv)
    q1_initial = q1.copy()
    q2_initial = q2.copy()
    terms_initial = coupled.evaluate(q, v)
    q_des_1 = q1.copy()
    q_des_2 = q2.copy()
    max_ground_height_error = 0.0
    max_constraint_velocity = 0.0
    max_tracking_error = 0.0
    worst_tracking_joint = ("none", 0.0)
    max_friction_ratio = 0.0
    min_normal_force = float("inf")
    mode_counts = {"stick": 0, "slip": 0, "lift_off": 0, "inactive": 0}
    steps = max(1, int(round(args.duration / args.time_step)))

    for step_id in range(steps):
        time_now = step_id * args.time_step
        for wheel_q_index in wheel_q_indices:
            q_des_1[wheel_q_index] = (
                q1_initial[wheel_q_index] + args.wheel_speed * time_now
            )
            q_des_2[wheel_q_index] = (
                q2_initial[wheel_q_index] + args.wheel_speed * time_now
            )
        q1_current, q2_current, _ = coupled.split_configuration(q)
        v1, v2, _ = coupled.split_velocity(v)
        terms = coupled.evaluate(q, v)
        desired_acceleration = np.zeros(coupled.nv)
        desired_velocity_1 = np.zeros(robot_model.nv)
        desired_velocity_2 = np.zeros(robot_model.nv)
        for local_index, (q_index, v_index) in enumerate(robot_pairs):
            if q_index in wheel_q_indices:
                desired_velocity_1[v_index] = args.wheel_speed
                desired_velocity_2[v_index] = args.wheel_speed
            desired_acceleration[v_index] = (
                args.kp * (q_des_1[q_index] - q1_current[q_index])
                + args.kd * (desired_velocity_1[v_index] - v1[v_index])
            )
            desired_acceleration[robot_model.nv + v_index] = (
                args.kp * (q_des_2[q_index] - q2_current[q_index])
                + args.kd * (desired_velocity_2[v_index] - v2[v_index])
            )
            limit = float(robot_model.effortLimit[v_index])
            error_1 = abs(q_des_1[q_index] - q1_current[q_index])
            error_2 = abs(q_des_2[q_index] - q2_current[q_index])
            max_tracking_error = max(max_tracking_error, error_1, error_2)
            joint_name = robot_model.names[2 + local_index]
            if error_1 > worst_tracking_joint[1]:
                worst_tracking_joint = (f"robot_1/{joint_name}", error_1)
            if error_2 > worst_tracking_joint[1]:
                worst_tracking_joint = (f"robot_2/{joint_name}", error_2)

        desired_acceleration = coupled.project_acceleration_to_constraints(
            terms, desired_acceleration
        )
        actuated_torque, _ = coupled.solve_minimum_norm_inverse_dynamics(
            terms, desired_acceleration
        )
        for local_index, (_, v_index) in enumerate(robot_pairs):
            limit = float(robot_model.effortLimit[v_index])
            if limit > 0.0 and math.isfinite(limit):
                actuated_torque[local_index] = float(
                    np.clip(actuated_torque[local_index], -limit, limit)
                )
                second_index = len(robot_pairs) + local_index
                actuated_torque[second_index] = float(
                    np.clip(actuated_torque[second_index], -limit, limit)
                )

        constraint_rhs = -terms.contact_jacobian_dot_velocity.copy()
        constraint_row = 0
        for spec, _force_basis in zip(
            coupled.active_support_specs, terms.support_force_bases
        ):
            robot_data = (
                coupled.robot_data_1 if spec[0] == 0 else coupled.robot_data_2
            )
            frame_id = coupled.support_frame_ids[spec[1]]
            ground_height_error = (
                coupled.support_contact_position(robot_data, frame_id)[2]
            )
            normal_velocity = terms.closure_velocity[constraint_row]
            constraint_rhs[constraint_row] += (
                -args.ground_position_gain * ground_height_error
                - args.ground_velocity_gain * normal_velocity
            )
            constraint_row += 3

        acceleration, contact_wrench = coupled.solve_constrained_dynamics(
            terms, actuated_torque, constraint_rhs=constraint_rhs
        )
        support_wrench = contact_wrench[: coupled.support_constraint_dim]
        support_forces = np.concatenate(
            tuple(
                matvec(
                    force_basis,
                    support_wrench[3 * index : 3 * index + 3],
                )
                for index, force_basis in enumerate(terms.support_force_bases)
            )
        )
        statuses = classify_support_contacts(
            support_forces,
            tuple(
                coupled.contact_names[: len(coupled.active_support_specs)]
            ),
            tuple(True for _ in coupled.active_support_specs),
            coupled.support_ground_normal,
            args.friction_coefficient,
        )
        for status in statuses:
            mode_counts[status.mode] += 1
            max_friction_ratio = max(max_friction_ratio, status.friction_ratio)
            min_normal_force = min(min_normal_force, status.normal_force)

        v = v + args.time_step * acceleration
        q = stack_configuration(
            pin.integrate(robot_model, q1_current, args.time_step * v[: robot_model.nv]),
            pin.integrate(
                robot_model,
                q2_current,
                args.time_step * v[robot_model.nv : 2 * robot_model.nv],
            ),
            pin.integrate(
                payload_model,
                q[2 * robot_model.nq :],
                args.time_step * v[2 * robot_model.nv :],
            ),
        )

        post_terms = coupled.evaluate(q, v)
        v = coupled.project_velocity_to_constraints(post_terms, v)
        post_terms = coupled.evaluate(q, v)
        for spec in coupled.active_support_specs:
            robot_data = (
                coupled.robot_data_1 if spec[0] == 0 else coupled.robot_data_2
            )
            frame_id = coupled.support_frame_ids[spec[1]]
            max_ground_height_error = max(
                max_ground_height_error,
                abs(
                    coupled.support_contact_position(robot_data, frame_id)[2]
                ),
            )
        max_constraint_velocity = max(
            max_constraint_velocity,
            vector_norm(post_terms.closure_velocity),
        )

    q1_final, q2_final, _ = coupled.split_configuration(q)
    wheel_displacement = max(
        max(
            abs(q1_final[index] - q1_initial[index])
            for index in wheel_q_indices
        ),
        max(
            abs(q2_final[index] - q2_initial[index])
            for index in wheel_q_indices
        ),
    )
    print("scenario: rolling wheel torque control")
    print(f"duration: {args.duration:.6g} s, dt: {args.time_step:.6g} s")
    print(f"wheel speed command: {args.wheel_speed:.6g} rad/s")
    print(f"wheel joint displacement: {wheel_displacement:.6g} rad")
    print(f"maximum joint tracking error: {max_tracking_error:.6g} rad")
    print(
        "worst tracked joint: "
        f"{worst_tracking_joint[0]} ({worst_tracking_joint[1]:.6g} rad)"
    )
    print(f"maximum ground height error: {max_ground_height_error:.6g} m")
    print(f"maximum constraint velocity norm: {max_constraint_velocity:.6g}")
    print(f"minimum normal force: {min_normal_force:.6g} N")
    print(f"maximum friction ratio: {max_friction_ratio:.6g}")
    print(f"contact mode counts: {mode_counts}")
    print(f"initial KKT constraint rows: {terms_initial.contact_jacobian.shape[0]}")
    print("rolling torque-control integration completed")


if __name__ == "__main__":
    main()
