#!/usr/bin/env python3
"""Load the staged TRON1 and payload models and check their dimensions."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-type",
        choices=("WF_TRON1A", "SF_TRON1A"),
        default="WF_TRON1A",
        help="TRON1 model to load",
    )
    return parser.parse_args()


def parser_path(path: Path) -> str:
    """Return a path that urdfdom can consume on Windows.

    The underlying urdfdom build may use a narrow filesystem API. Convert the
    workspace path to its ASCII 8.3 representation when the workspace contains
    non-ASCII characters.
    """

    if os.name != "nt":
        return str(path)

    kernel32 = ctypes.windll.kernel32
    get_short_path_name = kernel32.GetShortPathNameW
    get_short_path_name.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
    ]
    get_short_path_name.restype = ctypes.c_uint32

    buffer = ctypes.create_unicode_buffer(32768)
    length = get_short_path_name(str(path), buffer, len(buffer))
    if length == 0:
        raise RuntimeError(
            "Windows could not create an ASCII short path for "
            f"{path}. Move the model workspace to an ASCII-only path or "
            "enable 8.3 short names on the drive."
        )
    return buffer.value


def main() -> None:
    try:
        import pinocchio as pin
    except ImportError as exc:
        raise SystemExit(
            "Pinocchio is not installed in the current Python environment."
        ) from exc

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
    payload_urdf = root / "payload" / "payload_box.urdf"

    robot_model = pin.buildModelFromUrdf(
        parser_path(robot_urdf), pin.JointModelFreeFlyer()
    )
    payload_model = pin.buildModelFromUrdf(
        parser_path(payload_urdf), pin.JointModelFreeFlyer()
    )

    expected_robot = (21, 20)
    expected_payload = (7, 6)
    actual_robot = (robot_model.nq, robot_model.nv)
    actual_payload = (payload_model.nq, payload_model.nv)

    if actual_robot != expected_robot:
        raise RuntimeError(
            f"Unexpected {args.robot_type} dimensions: "
            f"got nq={actual_robot[0]}, nv={actual_robot[1]}, "
            f"expected nq={expected_robot[0]}, nv={expected_robot[1]}"
        )
    if actual_payload != expected_payload:
        raise RuntimeError(
            "Unexpected payload dimensions: "
            f"got nq={actual_payload[0]}, nv={actual_payload[1]}, "
            f"expected nq={expected_payload[0]}, nv={expected_payload[1]}"
        )

    link6_id = robot_model.getFrameId("link6")
    if link6_id >= robot_model.nframes:
        raise RuntimeError("The robot model does not contain the link6 frame")

    print(f"robot_type: {args.robot_type}")
    print(f"robot_1: nq={robot_model.nq}, nv={robot_model.nv}")
    print(f"robot_2: nq={robot_model.nq}, nv={robot_model.nv}")
    print(f"payload: nq={payload_model.nq}, nv={payload_model.nv}")
    print(f"robot link6 frame id: {link6_id}")
    print("model checks passed")


if __name__ == "__main__":
    main()
