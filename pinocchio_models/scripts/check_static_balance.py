#!/usr/bin/env python3
"""Check a first quasi-static two-robot payload balance.

The payload gravity is split between the two fixed grasp frames. Robot bases
are treated as externally supported, so this is not yet a full whole-body
contact dynamics solve.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from check_models import parser_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-type",
        choices=("WF_TRON1A", "SF_TRON1A"),
        default="WF_TRON1A",
    )
    parser.add_argument(
        "--left-force-ratio",
        type=float,
        default=0.5,
        help="Fraction of the vertical payload support force assigned left",
    )
    return parser.parse_args()


def place_free_flyer(q, x: float, y: float, z: float, yaw: float):
    q[:3] = [x, y, z]
    q[3:7] = [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
    return q


def transpose_matvec(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.array(
        [
            sum(matrix[row, column] * vector[row] for row in range(matrix.shape[0]))
            for column in range(matrix.shape[1])
        ]
    )


def vector_norm(vector: np.ndarray) -> float:
    return float(np.sqrt(sum(value * value for value in vector)))


def load_model(pin, path: Path):
    return pin.buildModelFromUrdf(
        parser_path(path), pin.JointModelFreeFlyer()
    )


def main() -> None:
    try:
        import pinocchio as pin
    except ImportError as exc:
        raise SystemExit("Pinocchio is not installed in this environment.") from exc

    args = parse_args()
    if not 0.0 <= args.left_force_ratio <= 1.0:
        raise SystemExit("left-force-ratio must be between 0 and 1")

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
    robot_model = load_model(pin, robot_urdf)
    payload_model = load_model(pin, payload_urdf)
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
        pin.neutral(payload_model), 0.0, 0.0, 1.04, 0.0
    )

    pin.computeGeneralizedGravity(robot_model, robot_data_1, q_robot_1)
    pin.computeGeneralizedGravity(robot_model, robot_data_2, q_robot_2)
    pin.computeGeneralizedGravity(payload_model, payload_data, q_payload)
    pin.forwardKinematics(robot_model, robot_data_1, q_robot_1)
    pin.updateFramePlacements(robot_model, robot_data_1)
    pin.forwardKinematics(robot_model, robot_data_2, q_robot_2)
    pin.updateFramePlacements(robot_model, robot_data_2)
    pin.forwardKinematics(payload_model, payload_data, q_payload)
    pin.updateFramePlacements(payload_model, payload_data)

    robot_frame = robot_model.getFrameId("link6")
    left_frame = payload_model.getFrameId("grasp_left")
    right_frame = payload_model.getFrameId("grasp_right")
    pin.computeJointJacobians(robot_model, robot_data_1, q_robot_1)
    pin.computeJointJacobians(robot_model, robot_data_2, q_robot_2)
    pin.computeJointJacobians(payload_model, payload_data, q_payload)
    jr_left = pin.getFrameJacobian(
        robot_model, robot_data_1, robot_frame, pin.LOCAL_WORLD_ALIGNED
    )
    jr_right = pin.getFrameJacobian(
        robot_model, robot_data_2, robot_frame, pin.LOCAL_WORLD_ALIGNED
    )
    jp_left = pin.getFrameJacobian(
        payload_model, payload_data, left_frame, pin.LOCAL_WORLD_ALIGNED
    )
    jp_right = pin.getFrameJacobian(
        payload_model, payload_data, right_frame, pin.LOCAL_WORLD_ALIGNED
    )

    payload_gravity = payload_data.g
    total_vertical_force = payload_gravity[2]
    left_wrench = np.zeros(6)
    right_wrench = np.zeros(6)
    left_wrench[2] = args.left_force_ratio * total_vertical_force
    right_wrench[2] = (1.0 - args.left_force_ratio) * total_vertical_force
    force_difference_moment = -0.354 * (
        left_wrench[2] - right_wrench[2]
    )
    left_wrench[4] = 0.5 * force_difference_moment
    right_wrench[4] = 0.5 * force_difference_moment
    payload_contact_force = transpose_matvec(
        jp_left, left_wrench
    ) + transpose_matvec(jp_right, right_wrench)
    payload_residual = payload_gravity - payload_contact_force

    left_robot_wrench = -left_wrench
    right_robot_wrench = -right_wrench
    robot_required_1 = robot_data_1.g - transpose_matvec(
        jr_left, left_robot_wrench
    )
    robot_required_2 = robot_data_2.g - transpose_matvec(
        jr_right, right_robot_wrench
    )
    arm_velocity_indices = list(range(14, 20))
    base_velocity_indices = list(range(0, 6))

    print(f"robot_type: {args.robot_type}")
    print("payload model: payload_with_handles")
    print(f"payload total mass: {sum(i.mass for i in payload_model.inertias):.6g} kg")
    print(f"payload gravity wrench: {payload_gravity[:6]}")
    print(f"left contact wrench on payload: {left_wrench}")
    print(f"right contact wrench on payload: {right_wrench}")
    print(f"payload balance residual norm: {vector_norm(payload_residual):.6g}")
    print(
        f"internal force norm (left-right): "
        f"{vector_norm(left_wrench - right_wrench):.6g}"
    )
    print(
        f"robot_1 arm torque max: "
        f"{np.max(np.abs(robot_required_1[arm_velocity_indices])):.6g}"
    )
    print(
        f"robot_2 arm torque max: "
        f"{np.max(np.abs(robot_required_2[arm_velocity_indices])):.6g}"
    )
    print(
        f"robot_1 fixed-base reaction norm: "
        f"{vector_norm(robot_required_1[base_velocity_indices]):.6g}"
    )
    print(
        f"robot_2 fixed-base reaction norm: "
        f"{vector_norm(robot_required_2[base_velocity_indices]):.6g}"
    )

    if vector_norm(payload_residual) > 1e-8:
        raise RuntimeError("Payload gravity is not balanced by the two contacts")
    print("static balance check passed")


if __name__ == "__main__":
    main()
