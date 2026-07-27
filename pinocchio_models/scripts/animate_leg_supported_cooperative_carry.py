#!/usr/bin/env python3
"""Animate supported cooperative carrying with coordinated leg motion.

The robot bases follow small prescribed motions. At every frame, the eight
leg joints are solved by position-level IK so that both wheel/ankle endpoint
positions remain fixed. The arms then track the two payload grasp frames.

This is a Pinocchio kinematic validation of whole-body coordination. It does
not yet integrate the constrained dynamics or enforce torque/friction limits.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from animate_cooperative_carry import (
    arm_joint_indices,
    gram_matrix,
    place_free_flyer,
    solve_arm_ik,
    solve_linear,
    transpose_matvec,
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
    parser.add_argument("--payload-amplitude-y", type=float, default=0.015)
    parser.add_argument("--payload-amplitude-z", type=float, default=0.015)
    parser.add_argument("--payload-amplitude-yaw", type=float, default=0.04)
    parser.add_argument("--base-amplitude-x", type=float, default=0.02)
    parser.add_argument("--base-amplitude-y", type=float, default=0.015)
    parser.add_argument("--base-amplitude-z", type=float, default=0.015)
    parser.add_argument("--base-amplitude-yaw", type=float, default=0.05)
    parser.add_argument(
        "--grasp-tolerance",
        type=float,
        default=1e-3,
        help="Maximum allowed six-dimensional grasp pose error",
    )
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


def leg_joint_names(robot_type: str) -> tuple[str, ...]:
    foot_joint = "wheel" if robot_type == "WF_TRON1A" else "ankle"
    return (
        "abad_L_Joint",
        "hip_L_Joint",
        "knee_L_Joint",
        f"{foot_joint}_L_Joint",
        "abad_R_Joint",
        "hip_R_Joint",
        "knee_R_Joint",
        f"{foot_joint}_R_Joint",
    )


def joint_indices(model, names: tuple[str, ...]):
    q_indices = []
    v_indices = []
    for name in names:
        joint_id = model.getJointId(name)
        if joint_id >= model.njoints:
            raise ValueError(f"Missing leg joint: {name}")
        q_indices.append(int(model.idx_qs[joint_id]))
        v_indices.append(int(model.idx_vs[joint_id]))
    return q_indices, v_indices


def support_position_error(pin, model, data, q, frame_ids, targets):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    errors = [
        targets[index] - data.oMf[frame_id].translation
        for index, frame_id in enumerate(frame_ids)
    ]
    return np.concatenate(errors)


def solve_leg_position_ik(
    pin,
    model,
    data,
    q,
    frame_ids,
    targets,
    q_indices,
    v_indices,
    iterations: int = 50,
):
    damping = 3e-3
    for _ in range(iterations):
        error = support_position_error(
            pin, model, data, q, frame_ids, targets
        )
        if vector_norm(error) < 1e-7:
            break
        jacobians = []
        for frame_id in frame_ids:
            jacobian = pin.computeFrameJacobian(
                model,
                data,
                q,
                frame_id,
                pin.LOCAL_WORLD_ALIGNED,
            )
            jacobians.append(jacobian[:3, v_indices])
        task_jacobian = np.vstack(jacobians)
        normal = gram_matrix(task_jacobian)
        normal += (damping * damping) * np.eye(normal.shape[0])
        step = transpose_matvec(
            task_jacobian,
            solve_linear(normal, error),
        )
        step = np.clip(step, -0.06, 0.06)
        dq = np.zeros(model.nv)
        dq[v_indices] = step
        q = pin.integrate(model, q, dq)
        q[q_indices] = np.clip(
            q[q_indices],
            model.lowerPositionLimit[q_indices],
            model.upperPositionLimit[q_indices],
        )
    final_error = support_position_error(
        pin, model, data, q, frame_ids, targets
    )
    return q, final_error


def main() -> None:
    args = parse_args()
    if args.duration <= 0 or args.fps <= 0:
        raise SystemExit("duration and fps must be positive")
    if args.grasp_tolerance <= 0:
        raise SystemExit("grasp-tolerance must be positive")

    patch_windows_ssl_store()
    expose_child_startup_patch()

    try:
        import pinocchio as pin
        import meshcat
        import meshcat.geometry as meshcat_geometry
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
    coupled = CoupledDynamicsModel(
        pin,
        robot_model,
        payload_model,
        support_mode="fixed_3d_position",
    )

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
    leg_names = leg_joint_names(args.robot_type)
    leg_q_indices, leg_v_indices = joint_indices(robot_model, leg_names)
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

    pin.forwardKinematics(robot_model, robot_data_1, q_robot_1)
    pin.updateFramePlacements(robot_model, robot_data_1)
    pin.forwardKinematics(robot_model, robot_data_2, q_robot_2)
    pin.updateFramePlacements(robot_model, robot_data_2)
    support_targets_1 = tuple(
        robot_data_1.oMf[frame_id].translation.copy()
        for frame_id in coupled.support_frame_ids
    )
    support_targets_2 = tuple(
        robot_data_2.oMf[frame_id].translation.copy()
        for frame_id in coupled.support_frame_ids
    )
    support_marker = meshcat_geometry.Sphere(0.035)
    support_stem = meshcat_geometry.Box((0.012, 0.012, 0.18))
    support_material = meshcat_geometry.MeshLambertMaterial(color=0x2ECC71)
    for robot_name, targets in (
        ("robot_1", support_targets_1),
        ("robot_2", support_targets_2),
    ):
        for index, target in enumerate(targets):
            marker = viewer[f"support_targets/{robot_name}/{index}"]
            marker.set_object(support_marker, support_material)
            transform = np.eye(4)
            transform[:3, 3] = target + np.array((0.0, 0.0, 0.18))
            marker.set_transform(transform)
            stem = viewer[f"support_targets/{robot_name}/{index}/stem"]
            stem.set_object(support_stem, support_material)
            stem_transform = np.eye(4)
            stem_transform[:3, 3] = target + np.array((0.0, 0.0, 0.09))
            stem.set_transform(stem_transform)

    steps = max(2, int(round(args.duration * args.fps)))
    print("scene: leg-supported cooperative carry demo")
    print(f"robot_type: {args.robot_type}")
    print(f"leg joints: {leg_names}")
    print(f"support frames: {coupled.support_frame_names}")
    print(f"stacked state: nq={coupled.nq}, nv={coupled.nv}, nu={coupled.nu}")
    print(
        "base amplitudes: "
        f"x={args.base_amplitude_x} m, "
        f"y={args.base_amplitude_y} m, "
        f"z={args.base_amplitude_z} m, "
        f"yaw={args.base_amplitude_yaw} rad"
    )
    print("green markers above the feet show the fixed leg support targets")
    print(f"MeshCat URL: {viewer.url()}")
    print("trajectory: base motion + leg compensation + payload carry")

    max_leg_error = 0.0
    max_grasp_error = 0.0
    max_leg_motion = 0.0
    max_constraint_rank = 0
    initial_leg_q = np.concatenate(
        (q_robot_1[leg_q_indices], q_robot_2[leg_q_indices])
    )
    start_time = time.perf_counter()
    final_q = None
    for step_id in range(steps):
        phase = 2.0 * math.pi * step_id / (steps - 1)
        body_x = args.base_amplitude_x * math.sin(phase)
        body_y = args.base_amplitude_y * math.sin(phase)
        body_z = args.base_amplitude_z * math.sin(phase)
        body_yaw = args.base_amplitude_yaw * math.sin(phase)
        q_robot_1 = place_free_flyer(
            q_robot_1,
            -0.7 + body_x,
            body_y,
            0.8 + body_z,
            body_yaw,
        )
        q_robot_2 = place_free_flyer(
            q_robot_2,
            0.7 - body_x,
            body_y,
            0.8 + body_z,
            math.pi - body_yaw,
        )
        q_robot_1, leg_error_1 = solve_leg_position_ik(
            pin,
            robot_model,
            robot_data_1,
            q_robot_1,
            coupled.support_frame_ids,
            support_targets_1,
            leg_q_indices,
            leg_v_indices,
        )
        q_robot_2, leg_error_2 = solve_leg_position_ik(
            pin,
            robot_model,
            robot_data_2,
            q_robot_2,
            coupled.support_frame_ids,
            support_targets_2,
            leg_q_indices,
            leg_v_indices,
        )
        leg_error = max(vector_norm(leg_error_1), vector_norm(leg_error_2))
        max_leg_error = max(max_leg_error, leg_error)
        current_leg_q = np.concatenate(
            (q_robot_1[leg_q_indices], q_robot_2[leg_q_indices])
        )
        max_leg_motion = max(
            max_leg_motion,
            vector_norm(current_leg_q - initial_leg_q),
        )

        q_payload = place_free_flyer(
            q_payload,
            0.0,
            args.payload_amplitude_y * math.sin(phase),
            args.payload_height
            + args.payload_amplitude_z * math.sin(phase),
            args.payload_amplitude_yaw * math.sin(phase),
        )
        pin.forwardKinematics(payload_model, payload_data, q_payload)
        pin.updateFramePlacements(payload_model, payload_data)
        q_robot_1, grasp_error_1 = solve_arm_ik(
            pin,
            robot_model,
            robot_data_1,
            q_robot_1,
            robot_frame,
            payload_data.oMf[left_frame],
            arm_q_indices,
            arm_v_indices,
            iterations=60,
        )
        q_robot_2, grasp_error_2 = solve_arm_ik(
            pin,
            robot_model,
            robot_data_2,
            q_robot_2,
            robot_frame,
            payload_data.oMf[right_frame],
            arm_q_indices,
            arm_v_indices,
            iterations=60,
        )
        grasp_error = max(vector_norm(grasp_error_1), vector_norm(grasp_error_2))
        max_grasp_error = max(max_grasp_error, grasp_error)

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
    final_leg_q = np.concatenate(
        (q_robot_1[leg_q_indices], q_robot_2[leg_q_indices])
    )

    print(f"maximum leg support position error: {max_leg_error:.6g} m")
    print(f"maximum grasp pose error: {max_grasp_error:.6g}")
    print(f"maximum leg joint motion norm: {max_leg_motion:.6g}")
    print(f"final leg joint motion norm: {vector_norm(final_leg_q - initial_leg_q):.6g}")
    print(f"contact Jacobian rank: {max_constraint_rank}/{coupled.contact_wrench_dim}")
    print(f"KKT dynamics residual norm: {vector_norm(dynamics_residual):.6g}")
    print(
        "KKT acceleration-constraint residual norm: "
        f"{vector_norm(acceleration_residual):.6g}"
    )
    print(
        "The displayed motion is whole-body kinematic IK; the final pose is "
        "checked with supported coupled dynamics."
    )

    if max_leg_error > 1e-5:
        raise RuntimeError("Leg support position tracking error is too large")
    if max_grasp_error > args.grasp_tolerance:
        raise RuntimeError(
            "Grasp pose tracking error is too large: "
            f"{max_grasp_error:.6g} > {args.grasp_tolerance:.6g}"
        )
    if max_constraint_rank < coupled.contact_wrench_dim:
        raise RuntimeError("Support/grasp contact Jacobian is rank deficient")
    if vector_norm(dynamics_residual) > 1e-7:
        raise RuntimeError("KKT dynamics residual is too large")
    if vector_norm(acceleration_residual) > 1e-7:
        raise RuntimeError("KKT acceleration constraint residual is too large")
    if not args.no_wait:
        print("Press Enter to close MeshCat.")
        input()


if __name__ == "__main__":
    main()
