#!/usr/bin/env python3
"""Animate a small two-robot cooperative carry trajectory.

The robots keep their bases fixed and track the two fixed payload grasp frames
with damped least-squares arm IK. This is a kinematic reachability demo; it
does not solve contact wrenches, friction, or whole-body dynamics.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

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


def place_free_flyer(q, x: float, y: float, z: float, yaw: float):
    q[:3] = [x, y, z]
    q[3:7] = [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
    return q


def solve_linear(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Solve a small dense system without depending on LAPACK SVD."""

    augmented = np.column_stack((matrix.astype(float), vector.astype(float)))
    size = augmented.shape[0]
    for column in range(size):
        pivot_row = column + int(
            np.argmax(np.abs(augmented[column:, column]))
        )
        pivot = augmented[pivot_row, column]
        if abs(pivot) < 1e-12:
            raise RuntimeError("Singular damped IK system")
        if pivot_row != column:
            augmented[[column, pivot_row]] = augmented[[pivot_row, column]]
        augmented[column] /= augmented[column, column]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row, column]
            if abs(factor) > 1e-12:
                augmented[row] -= factor * augmented[column]
    return augmented[:, -1]


def gram_matrix(matrix: np.ndarray) -> np.ndarray:
    result = np.zeros((matrix.shape[0], matrix.shape[0]))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[0]):
            result[row, column] = sum(
                matrix[row, k] * matrix[column, k]
                for k in range(matrix.shape[1])
            )
    return result


def transpose_matvec(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.array(
        [
            sum(matrix[row, column] * vector[row] for row in range(matrix.shape[0]))
            for column in range(matrix.shape[1])
        ]
    )


def arm_joint_indices(model, names: tuple[str, ...]):
    q_indices = []
    v_indices = []
    for name in names:
        joint_id = model.getJointId(name)
        q_indices.append(int(model.idx_qs[joint_id]))
        v_indices.append(int(model.idx_vs[joint_id]))
    return q_indices, v_indices


def solve_arm_ik(
    pin,
    model,
    data,
    q,
    frame_id: int,
    target,
    q_indices,
    v_indices,
    iterations: int = 60,
):
    damping = 2e-3
    for _ in range(iterations):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current = data.oMf[frame_id]
        error = pin.log6(current.actInv(target)).vector
        if float(np.sqrt(np.dot(error, error))) < 1e-5:
            break
        jacobian = pin.computeFrameJacobian(
            model, data, q, frame_id, pin.LOCAL
        )
        arm_jacobian = jacobian[:, v_indices]
        normal = gram_matrix(arm_jacobian)
        normal += (damping * damping) * np.eye(6)
        step = transpose_matvec(
            arm_jacobian, solve_linear(normal, error)
        )
        step = np.clip(step, -0.12, 0.12)
        dq = np.zeros(model.nv)
        dq[v_indices] = step
        q = pin.integrate(model, q, dq)
        q[q_indices] = np.clip(
            q[q_indices],
            model.lowerPositionLimit[q_indices],
            model.upperPositionLimit[q_indices],
        )
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    final_error = pin.log6(data.oMf[frame_id].actInv(target)).vector
    return q, final_error


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

    q_robot_1 = place_free_flyer(
        pin.neutral(robot_model), -0.7, 0.0, 0.8, 0.0
    )
    q_robot_2 = place_free_flyer(
        pin.neutral(robot_model), 0.7, 0.0, 0.8, math.pi
    )
    q_payload = place_free_flyer(
        pin.neutral(payload_model), 0.0, 0.0, args.payload_height, 0.0
    )
    robot_data_1 = robot_model.createData()
    robot_data_2 = robot_model.createData()
    payload_data = payload_model.createData()

    steps = max(2, int(round(args.duration * args.fps)))
    print("scene: cooperative carry kinematic demo")
    print(f"robot_type: {args.robot_type}")
    print(f"robot model: nq={robot_model.nq}, nv={robot_model.nv}")
    print(f"payload model: nq={payload_model.nq}, nv={payload_model.nv}")
    print(f"MeshCat URL: {viewer.url()}")
    print("trajectory: payload y/z translation and yaw rotation")

    max_error = 0.0
    start_time = time.perf_counter()
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
        error_norm = max(
            float(np.sqrt(np.dot(error_1, error_1))),
            float(np.sqrt(np.dot(error_2, error_2))),
        )
        max_error = max(max_error, error_norm)
        robot_1.display(q_robot_1)
        robot_2.display(q_robot_2)
        payload.display(q_payload)

        target_time = start_time + (step_id + 1) / args.fps
        time.sleep(max(0.0, target_time - time.perf_counter()))

    print(f"maximum end-effector pose error: {max_error:.6g}")
    print("This demo is kinematic only; no contact wrench is solved.")
    if not args.no_wait:
        print("Press Enter to close MeshCat.")
        input()


if __name__ == "__main__":
    main()
