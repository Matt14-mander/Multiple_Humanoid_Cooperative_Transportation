#!/usr/bin/env python3
"""Check the first two-robot payload grasp configuration.

This script validates nominal frame placement and the rank of the stacked
rigid-grasp Jacobian. It does not solve inverse kinematics or contact forces.
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
        "--payload-model",
        choices=("box", "with-handles"),
        default="with-handles",
    )
    parser.add_argument("--position-tolerance", type=float, default=5e-3)
    parser.add_argument("--orientation-tolerance", type=float, default=5e-3)
    return parser.parse_args()


def place_free_flyer(q, x: float, y: float, z: float, yaw: float):
    q[:3] = [x, y, z]
    q[3:7] = [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
    return q


def load_model(pin, path: Path):
    return pin.buildModelFromUrdf(
        parser_path(path), pin.JointModelFreeFlyer()
    )


def frame_error(pin, robot_data, robot_frame, payload_data, payload_frame):
    relative = robot_data.oMf[robot_frame].actInv(payload_data.oMf[payload_frame])
    error = pin.log6(relative).vector
    return error, relative


def elimination_rank(matrix: np.ndarray, tolerance: float = 1e-8) -> int:
    """Compute row rank without calling the environment's LAPACK SVD."""

    work = np.array(matrix, dtype=float, copy=True)
    rows, columns = work.shape
    scale = max(float(np.max(np.abs(work))), 1.0)
    rank = 0
    for column in range(columns):
        if rank >= rows:
            break
        pivot_row = rank + int(np.argmax(np.abs(work[rank:, column])))
        pivot = work[pivot_row, column]
        if abs(pivot) <= tolerance * scale:
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
    payload_filename = {
        "box": "payload_box.urdf",
        "with-handles": "payload_with_handles.urdf",
    }[args.payload_model]
    payload_urdf = root / "payload" / payload_filename

    robot_model = load_model(pin, robot_urdf)
    robot_data_1 = robot_model.createData()
    robot_data_2 = robot_model.createData()
    payload_model = load_model(pin, payload_urdf)
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

    robot_frame = robot_model.getFrameId("link6")
    payload_frames = {
        "robot_1": payload_model.getFrameId("grasp_left"),
        "robot_2": payload_model.getFrameId("grasp_right"),
    }
    if any(frame_id >= payload_model.nframes for frame_id in payload_frames.values()):
        raise RuntimeError(
            "The selected payload does not provide grasp_left and grasp_right frames."
        )

    pin.forwardKinematics(robot_model, robot_data_1, q_robot_1)
    pin.updateFramePlacements(robot_model, robot_data_1)
    pin.forwardKinematics(robot_model, robot_data_2, q_robot_2)
    pin.updateFramePlacements(robot_model, robot_data_2)
    pin.forwardKinematics(payload_model, payload_data, q_payload)
    pin.updateFramePlacements(payload_model, payload_data)

    errors = {}
    for name, robot_data in (
        ("robot_1", robot_data_1),
        ("robot_2", robot_data_2),
    ):
        error, relative = frame_error(
            pin,
            robot_data,
            robot_frame,
            payload_data,
            payload_frames[name],
        )
        errors[name] = error
        print(f"{name} link6 position: {robot_data.oMf[robot_frame].translation}")
        print(
            f"{name} grasp position: "
            f"{payload_data.oMf[payload_frames[name]].translation}"
        )
        print(
            f"{name} closure error: position={np.linalg.norm(error[:3]):.6g} m, "
            f"orientation={np.linalg.norm(error[3:]):.6g} rad"
        )
        if not np.all(np.isfinite(error)):
            raise RuntimeError(f"Non-finite relative transform for {name}")

    pin.computeJointJacobians(robot_model, robot_data_1, q_robot_1)
    pin.updateFramePlacements(robot_model, robot_data_1)
    pin.computeJointJacobians(robot_model, robot_data_2, q_robot_2)
    pin.updateFramePlacements(robot_model, robot_data_2)
    pin.computeJointJacobians(payload_model, payload_data, q_payload)
    pin.updateFramePlacements(payload_model, payload_data)

    jr_1 = pin.getFrameJacobian(
        robot_model, robot_data_1, robot_frame, pin.LOCAL_WORLD_ALIGNED
    )
    jr_2 = pin.getFrameJacobian(
        robot_model, robot_data_2, robot_frame, pin.LOCAL_WORLD_ALIGNED
    )
    jp_left = pin.getFrameJacobian(
        payload_model,
        payload_data,
        payload_frames["robot_1"],
        pin.LOCAL_WORLD_ALIGNED,
    )
    jp_right = pin.getFrameJacobian(
        payload_model,
        payload_data,
        payload_frames["robot_2"],
        pin.LOCAL_WORLD_ALIGNED,
    )
    zero_robot = np.zeros((6, robot_model.nv))
    zero_payload = np.zeros((6, payload_model.nv))
    constraint_jacobian = np.vstack(
        (
            np.hstack((jr_1, zero_robot, -jp_left)),
            np.hstack((zero_robot, jr_2, -jp_right)),
        )
    )
    rank = elimination_rank(constraint_jacobian)

    print(f"payload model: {args.payload_model}")
    print(
        f"stacked velocity dimensions: "
        f"{robot_model.nv}+{robot_model.nv}+{payload_model.nv}="
        f"{2 * robot_model.nv + payload_model.nv}"
    )
    print(f"constraint rows: {constraint_jacobian.shape[0]}")
    print(f"constraint Jacobian rank: {rank}")

    for name, error in errors.items():
        if np.linalg.norm(error[:3]) > args.position_tolerance:
            raise RuntimeError(f"{name} position closure is outside tolerance")
        if np.linalg.norm(error[3:]) > args.orientation_tolerance:
            raise RuntimeError(
                f"{name} orientation closure is outside tolerance"
            )
    if args.payload_model == "with-handles" and rank < 12:
        raise RuntimeError(
            "The two full-pose grasp constraints are rank deficient at the "
            "nominal configuration."
        )
    print("grasp kinematics check passed")


if __name__ == "__main__":
    main()
