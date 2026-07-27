#!/usr/bin/env python3
"""Animate supported dual-robot grasping and cooperative carrying.

The demo keeps the four wheel/ankle contact-point positions fixed, solves
arm-only IK for the two rigid grasp frames, and moves the payload through a
small synchronized trajectory. At the end it evaluates the constrained KKT
system and prints the support forces and grasp wrenches.

This is a staged software validation demo: the displayed motion is generated
by kinematic IK, while the support/grasp constrained dynamics are evaluated at
the displayed configurations. It is not yet a torque-controlled simulator.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from animate_cooperative_carry import (
    arm_joint_indices,
    place_free_flyer,
    solve_arm_ik,
)
from check_coupled_dynamics import elimination_rank
from coupled_dynamics import CoupledDynamicsModel
from visualize_meshcat import (
    build_models,
    expose_child_startup_patch,
    patch_meshcat_server_subprocess,
    patch_meshcat_transform,
    patch_windows_ssl_store,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-type",
        choices=("WF_TRON1A", "SF_TRON1A"),
        default="WF_TRON1A",
    )
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--payload-height", type=float, default=1.04)
    parser.add_argument("--amplitude-y", type=float, default=0.02)
    parser.add_argument("--amplitude-z", type=float, default=0.02)
    parser.add_argument("--amplitude-yaw", type=float, default=0.05)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start MeshCat without opening a browser automatically",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Exit after the trajectory instead of waiting for Enter",
    )
    return parser.parse_args()


def vector_norm(vector: np.ndarray) -> float:
    return float(np.sqrt(sum(value * value for value in vector)))


def stack_configuration(q1, q2, qp):
    return np.concatenate((q1, q2, qp))


def main() -> None:
    args = parse_args()
    if args.duration <= 0 or args.fps <= 0:
        raise SystemExit("duration and fps must be positive")

    patch_windows_ssl_store()
    expose_child_startup_patch()

    try:
        import pinocchio as pin
        import meshcat
        from pinocchio.visualize import MeshcatVisualizer

        patch_meshcat_transform()
    except ImportError as exc:
        raise SystemExit(
            "This demo requires Pinocchio and meshcat in the active environment."
        ) from exc

    patch_meshcat_server_subprocess()
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

    robot_model, robot_collision, robot_visual = build_models(
        pin, robot_urdf, root
    )
    payload_model, payload_collision, payload_visual = build_models(
        pin, payload_urdf, root
    )
    coupled = CoupledDynamicsModel(pin, robot_model, payload_model)

    viewer = meshcat.Visualizer()
    robot_1 = MeshcatVisualizer(robot_model, robot_collision, robot_visual)
    robot_1.initViewer(viewer=viewer, open=not args.no_browser)
    robot_1.loadViewerModel(rootNodeName="robot_1")
    robot_2 = MeshcatVisualizer(robot_model, robot_collision, robot_visual)
    robot_2.initViewer(viewer=viewer, open=False)
    robot_2.loadViewerModel(rootNodeName="robot_2")
    payload = MeshcatVisualizer(
        payload_model, payload_collision, payload_visual
    )
    payload.initViewer(viewer=viewer, open=False)
    payload.loadViewerModel(rootNodeName="payload")

    arm_names = ("J1", "J2", "J3", "J4", "J5", "J6")
    arm_q_indices, arm_v_indices = arm_joint_indices(robot_model, arm_names)
    robot_frame = robot_model.getFrameId("link6")
    left_frame = payload_model.getFrameId("grasp_left")
    right_frame = payload_model.getFrameId("grasp_right")
    robot_data_1 = robot_model.createData()
    robot_data_2 = robot_model.createData()
    payload_data = payload_model.createData()

    q_robot_1 = place_free_flyer(
        pin.neutral(robot_model), -0.7, 0.0, 0.8, 0.0
    )
    q_robot_2 = place_free_flyer(
        pin.neutral(robot_model), 0.7, 0.0, 0.8, math.pi
    )
    q_payload = place_free_flyer(
        pin.neutral(payload_model), 0.0, 0.0, args.payload_height, 0.0
    )

    steps = max(2, int(round(args.duration * args.fps)))
    print("scene: supported cooperative carry demo")
    print(f"robot_type: {args.robot_type}")
    print(f"support model: fixed 3D position at {coupled.support_frame_names}")
    print(f"support constraint dimension: {coupled.support_constraint_dim}")
    print(f"grasp constraint dimension: {coupled.grasp_constraint_dim}")
    print(f"stacked state: nq={coupled.nq}, nv={coupled.nv}, nu={coupled.nu}")
    print(f"MeshCat URL: {viewer.url()}")
    print("trajectory: synchronized payload y/z translation and yaw rotation")

    max_error = 0.0
    max_constraint_rank = 0
    start_time = time.perf_counter()
    final_q = None
    for step_id in range(steps):
        phase = 2.0 * math.pi * step_id / (steps - 1)
        q_payload = place_free_flyer(
            q_payload,
            0.0,
            args.amplitude_y * math.sin(phase),
            args.payload_height + args.amplitude_z * math.sin(phase),
            args.amplitude_yaw * math.sin(phase),
        )
        pin.forwardKinematics(payload_model, payload_data, q_payload)
        pin.updateFramePlacements(payload_model, payload_data)

        q_robot_1, error_1 = solve_arm_ik(
            pin,
            robot_model,
            robot_data_1,
            q_robot_1,
            robot_frame,
            payload_data.oMf[left_frame],
            arm_q_indices,
            arm_v_indices,
        )
        q_robot_2, error_2 = solve_arm_ik(
            pin,
            robot_model,
            robot_data_2,
            q_robot_2,
            robot_frame,
            payload_data.oMf[right_frame],
            arm_q_indices,
            arm_v_indices,
        )

        error_norm = max(vector_norm(error_1), vector_norm(error_2))
        max_error = max(max_error, error_norm)
        q_stacked = stack_configuration(q_robot_1, q_robot_2, q_payload)
        terms = coupled.evaluate(q_stacked, np.zeros(coupled.nv))
        max_constraint_rank = max(
            max_constraint_rank,
            elimination_rank(terms.contact_jacobian),
        )

        robot_1.display(q_robot_1)
        robot_2.display(q_robot_2)
        payload.display(q_payload)
        final_q = q_stacked

        target_time = start_time + (step_id + 1) / args.fps
        time.sleep(max(0.0, target_time - time.perf_counter()))

    if final_q is None:
        raise RuntimeError("The trajectory did not produce a configuration")

    final_terms = coupled.evaluate(final_q, np.zeros(coupled.nv))
    zero_torque = np.zeros(coupled.nu)
    solved_acceleration, solved_contact_wrench = (
        coupled.solve_constrained_dynamics(final_terms, zero_torque)
    )
    dynamics_residual = coupled.dynamics_residual(
        final_terms,
        solved_acceleration,
        zero_torque,
        solved_contact_wrench,
    )
    acceleration_residual = coupled.acceleration_constraint_residual(
        final_terms, solved_acceleration
    )

    print(f"maximum end-effector pose error: {max_error:.6g}")
    print(f"contact Jacobian rank: {max_constraint_rank}/{coupled.contact_wrench_dim}")
    print(
        "solved support forces [r1-left, r1-right, r2-left, r2-right]: "
        f"{solved_contact_wrench[:coupled.support_constraint_dim]}"
    )
    print(
        "solved grasp wrenches [left, right]: "
        f"{solved_contact_wrench[coupled.support_constraint_dim:]}"
    )
    print(f"solved acceleration norm: {vector_norm(solved_acceleration):.6g}")
    print(f"KKT dynamics residual norm: {vector_norm(dynamics_residual):.6g}")
    print(
        "KKT acceleration-constraint residual norm: "
        f"{vector_norm(acceleration_residual):.6g}"
    )
    print(
        "The displayed trajectory is kinematic IK; the KKT result is the "
        "support/grasp dynamics diagnostic at the final pose."
    )

    if vector_norm(dynamics_residual) > 1e-7:
        raise RuntimeError("KKT dynamics residual is too large")
    if vector_norm(acceleration_residual) > 1e-7:
        raise RuntimeError("KKT acceleration constraint residual is too large")
    if max_constraint_rank < coupled.contact_wrench_dim:
        raise RuntimeError("Support/grasp contact Jacobian is rank deficient")
    if not args.no_wait:
        print("Press Enter to close MeshCat.")
        input()


if __name__ == "__main__":
    main()
