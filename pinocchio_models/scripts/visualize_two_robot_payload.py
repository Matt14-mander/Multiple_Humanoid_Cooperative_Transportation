#!/usr/bin/env python3
"""Visualize two TRON1 robots and a payload in one MeshCat scene.

This is a kinematic scene check only. It does not enforce grasp constraints or
solve contact forces.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

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
        help="TRON1 model to visualize",
    )
    parser.add_argument(
        "--base-height",
        type=float,
        default=0.8,
        help="Initial floating-base height in meters",
    )
    parser.add_argument(
        "--payload-height",
        type=float,
        default=1.04,
        help="Payload center height in meters",
    )
    parser.add_argument(
        "--payload-model",
        choices=("box", "with-handles"),
        default="with-handles",
        help="Payload geometry to display",
    )
    parser.add_argument(
        "--show-collision",
        action="store_true",
        help="Display collision geometry for all models",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start MeshCat without opening a browser automatically",
    )
    return parser.parse_args()


def place_free_flyer(q, x: float, y: float, z: float, yaw: float):
    """Set a free-flyer pose using Pinocchio's xyzw quaternion layout."""

    q[:3] = [x, y, z]
    q[3:7] = [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
    return q


def main() -> None:
    args = parse_args()
    patch_windows_ssl_store()
    expose_child_startup_patch()

    try:
        import pinocchio as pin
        import meshcat
        from pinocchio.visualize import MeshcatVisualizer
        patch_meshcat_transform()
    except ImportError as exc:
        raise SystemExit(
            "This scene requires Pinocchio and meshcat. "
            "Install meshcat with: pip install meshcat"
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
    payload_filenames = {
        "box": "payload_box.urdf",
        "with-handles": "payload_with_handles.urdf",
    }
    payload_urdf = root / "payload" / payload_filenames[args.payload_model]

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

    if args.show_collision:
        for visualizer in (robot_1, robot_2, payload):
            visualizer.displayVisuals(False)
            visualizer.displayCollisions(True)

    q_robot_1 = pin.neutral(robot_model)
    q_robot_2 = pin.neutral(robot_model)
    place_free_flyer(q_robot_1, -0.7, 0.0, args.base_height, 0.0)
    place_free_flyer(q_robot_2, 0.7, 0.0, args.base_height, math.pi)

    q_payload = pin.neutral(payload_model)
    place_free_flyer(q_payload, 0.0, 0.0, args.payload_height, 0.0)

    robot_1.display(q_robot_1)
    robot_2.display(q_robot_2)
    payload.display(q_payload)

    print(f"robot_type: {args.robot_type}")
    print("scene: robot_1 + robot_2 + payload")
    print(f"robot model: nq={robot_model.nq}, nv={robot_model.nv}")
    print(
        f"payload model: {args.payload_model}, "
        f"nq={payload_model.nq}, nv={payload_model.nv}"
    )
    for frame_name in ("grasp_left", "grasp_right"):
        frame_id = payload_model.getFrameId(frame_name)
        if frame_id < payload_model.nframes:
            print(f"payload frame: {frame_name} (id={frame_id})")
    print(f"MeshCat URL: {viewer.url()}")
    print("This is a visualization-only scene; no grasp constraint is enforced.")
    print("Press Enter to close MeshCat.")
    input()


if __name__ == "__main__":
    main()
