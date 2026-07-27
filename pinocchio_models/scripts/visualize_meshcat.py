#!/usr/bin/env python3
"""Visualize one staged TRON1 plus Airbot arm in MeshCat."""

from __future__ import annotations

import argparse
import atexit
import os
import re
import ssl
import subprocess
import sys
from pathlib import Path

from check_models import parser_path


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
        "--show-collision",
        action="store_true",
        help="Display collision geometry instead of only visual geometry",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start MeshCat without opening a browser automatically",
    )
    return parser.parse_args()


def build_models(pin, urdf_path: Path, package_root: Path):
    urdf = parser_path(urdf_path)
    package_dirs = [parser_path(package_root)]
    root_joint = pin.JointModelFreeFlyer()

    try:
        return pin.buildModelsFromUrdf(urdf, package_dirs, root_joint)
    except TypeError:
        return pin.buildModelsFromUrdf(
            urdf,
            package_dirs=package_dirs,
            root_joint=root_joint,
        )


def patch_windows_ssl_store() -> None:
    """Avoid malformed Windows certificate entries during MeshCat import."""

    if os.name != "nt":
        return

    try:
        ssl.create_default_context()
        return
    except ssl.SSLError:
        pass

    default_cafile = ssl.get_default_verify_paths().cafile

    def create_context_without_windows_store(*args, **kwargs):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        cafile = kwargs.get("cafile") or default_cafile
        capath = kwargs.get("capath")
        cadata = kwargs.get("cadata")
        if cafile or capath or cadata:
            context.load_verify_locations(
                cafile=cafile,
                capath=capath,
                cadata=cadata,
            )
        return context

    ssl.create_default_context = create_context_without_windows_store


def expose_child_startup_patch() -> None:
    """Make sitecustomize.py available to MeshCat's server subprocess."""

    scripts_dir = str(Path(__file__).resolve().parent)
    python_path = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in python_path.split(os.pathsep) if entry]
    if scripts_dir not in entries:
        entries.insert(0, scripts_dir)
    os.environ["PYTHONPATH"] = os.pathsep.join(entries)


def patch_meshcat_server_subprocess() -> None:
    """Preserve the local startup patch in MeshCat's ZMQ child process."""

    import meshcat.visualizer as meshcat_visualizer

    def capture(pattern: str, value: str) -> str:
        match = re.match(pattern, value)
        if not match:
            raise RuntimeError(
                f"Could not parse MeshCat server output: {value!r}"
            )
        return match.group(1)

    def start_server(zmq_url=None, server_args=None):
        args = [sys.executable, "-u", "-m", "meshcat.servers.zmqserver"]
        if zmq_url is not None:
            args.extend(["--zmq-url", zmq_url])
        if server_args:
            args.extend(server_args)

        env = dict(os.environ)
        scripts_dir = str(Path(__file__).resolve().parent)
        python_path = env.get("PYTHONPATH", "")
        entries = [entry for entry in python_path.split(os.pathsep) if entry]
        if scripts_dir not in entries:
            entries.insert(0, scripts_dir)
        env["PYTHONPATH"] = os.pathsep.join(entries)

        server_proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )

        line = b""
        while b"zmq_url" not in line:
            line = server_proc.stdout.readline().strip()
            if server_proc.poll() is not None:
                outs, errs = server_proc.communicate()
                print(outs.decode("utf-8", errors="replace"))
                print(errs.decode("utf-8", errors="replace"))
                raise RuntimeError(
                    "MeshCat server exited prematurely with exit code "
                    f"{server_proc.poll()}"
                )

        zmq_url_value = capture(r"^zmq_url=(.*)$", line.decode("utf-8"))
        web_line = server_proc.stdout.readline().strip().decode("utf-8")
        web_url = capture(r"^web_url=(.*)$", web_line)

        def cleanup():
            if server_proc.poll() is None:
                server_proc.kill()
                server_proc.wait()

        atexit.register(cleanup)
        return server_proc, zmq_url_value, web_url

    meshcat_visualizer.start_zmq_server_as_subprocess = start_server


def patch_meshcat_transform() -> None:
    """Use a Windows-stable equivalent of MeshCat's scale transform."""

    import pinocchio.visualize.meshcat_visualizer as pinocchio_meshcat

    def safe_apply_scaling(homogeneous_transform, scale):
        transform = homogeneous_transform.copy()
        scale_array = __import__("numpy").array(scale).flatten()
        transform[:3, :3] *= scale_array.reshape(1, 3)
        return transform

    pinocchio_meshcat.applyScalingOnHomegeneousTransform = safe_apply_scaling


def main() -> None:
    args = parse_args()
    patch_windows_ssl_store()
    expose_child_startup_patch()

    try:
        import pinocchio as pin
        from pinocchio.visualize import MeshcatVisualizer
        patch_meshcat_transform()
        patch_meshcat_server_subprocess()
    except ImportError as exc:
        raise SystemExit(
            "MeshCat visualization requires Pinocchio and the meshcat package. "
            "Install meshcat in the active environment with: pip install meshcat"
        ) from exc

    root = Path(__file__).resolve().parents[1]
    urdf_path = (
        root
        / "robot_description"
        / "pointfoot"
        / args.robot_type
        / "urdf"
        / "robot_with_arm.urdf"
    )

    model, collision_model, visual_model = build_models(pin, urdf_path, root)
    visualizer = MeshcatVisualizer(model, collision_model, visual_model)
    visualizer.initViewer(open=not args.no_browser)
    visualizer.loadViewerModel()

    if args.show_collision:
        visualizer.displayVisuals(False)
        visualizer.displayCollisions(True)

    q = pin.neutral(model)
    q[2] = args.base_height
    visualizer.display(q)

    print(f"robot_type: {args.robot_type}")
    print(f"nq={model.nq}, nv={model.nv}")
    print(f"MeshCat URL: {visualizer.viewer.url()}")
    print("Press Enter to close MeshCat.")
    input()


if __name__ == "__main__":
    main()
