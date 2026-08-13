"""Build two official WFYG_TRON2A robots and one cooperative payload."""

import argparse
import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .configuration import load_config
from .paths import DEFAULT_CONFIG, GENERATED_MODEL, SOURCE_MESH_DIR, SOURCE_MJCF


REFERENCE_ATTRIBUTES = {
    "joint",
    "objname",
    "site",
    "body",
    "body1",
    "body2",
    "target",
}


def _absolute(path: Path) -> str:
    return path.resolve().as_posix()


def _indent(element: ET.Element, level: int = 0) -> None:
    pad = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = pad + "  "
        for child in element:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not element.tail or not element.tail.strip()):
        element.tail = pad


def _prefix_subtree(body: ET.Element, prefix: str) -> None:
    for element in body.iter():
        if element.get("name"):
            element.set("name", prefix + element.get("name"))
        for attribute in REFERENCE_ATTRIBUTES:
            if element.get(attribute):
                element.set(attribute, prefix + element.get(attribute))
        if element.tag == "geom" and element.get("mesh"):
            element.set("mesh", "tron2_" + element.get("mesh"))
    root_joint = body.find("joint[@type='free']")
    if root_joint is not None and not root_joint.get("name"):
        root_joint.set("name", prefix + "root_free")


def _copy_robot(source_body, prefix, pose):
    x, y, z, yaw = [float(value) for value in pose]
    body = copy.deepcopy(source_body)
    _prefix_subtree(body, prefix)
    body.set("pos", f"{x} {y} {z}")
    body.set("quat", f"{math.cos(yaw / 2.0)} 0 0 {math.sin(yaw / 2.0)}")
    pick = body.find(".//body[@name='" + prefix + "gripper_pick']")
    if pick is None:
        raise ValueError("gripper_pick was not found in WFYG_TRON2A")
    ET.SubElement(
        pick,
        "site",
        {
            "name": prefix + "ee_site",
            "size": "0.025",
            "rgba": "1 0.15 0.15 1",
        },
    )
    return body


def _add_payload(worldbody, config):
    px, py, pz, yaw = [float(value) for value in config["payload_pose"]]
    size = np.asarray(config["payload_body_size_m"], dtype=float)
    half = size / 2.0
    mass = float(config["payload_mass_kg"])
    com = np.asarray(config.get("payload_com_offset_m", [0, 0, 0]), dtype=float)
    if mass <= 0.0 or size.shape != (3,) or np.any(size <= 0.0):
        raise ValueError("payload mass and dimensions must be positive")
    if com.shape != (3,) or np.any(np.abs(com) > half):
        raise ValueError("payload COM must be inside the payload body")
    inertia = mass / 12.0 * np.array(
        [size[1] ** 2 + size[2] ** 2, size[0] ** 2 + size[2] ** 2,
         size[0] ** 2 + size[1] ** 2]
    )
    payload = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "payload_body",
            "pos": f"{px} {py} {pz}",
            "quat": f"{math.cos(yaw / 2.0)} 0 0 {math.sin(yaw / 2.0)}",
        },
    )
    ET.SubElement(payload, "freejoint", {"name": "payload_free"})
    ET.SubElement(
        payload,
        "inertial",
        {
            "pos": "{} {} {}".format(*com),
            "mass": str(mass),
            "diaginertia": "{} {} {}".format(*inertia),
        },
    )
    ET.SubElement(
        payload,
        "geom",
        {
            "name": "payload_collision",
            "type": "box",
            "size": "{} {} {}".format(*half),
            "mass": "0",
            "rgba": "0.22 0.46 0.78 1",
            "friction": "1.0 0.02 0.002",
        },
    )
    handle_center = float(config["handle_center_y_m"])
    handle_half = np.asarray(config["handle_size_m"], dtype=float) / 2.0
    for side, sign in (("left", -1.0), ("right", 1.0)):
        handle = ET.SubElement(
            payload,
            "body",
            {"name": "payload_grasp_" + side, "pos": f"0 {sign * handle_center} 0"},
        )
        ET.SubElement(
            handle,
            "geom",
            {
                "name": "payload_handle_" + side,
                "type": "box",
                "size": "{} {} {}".format(*handle_half),
                "mass": "0",
                "rgba": "0.88 0.65 0.12 1",
            },
        )


def _copy_named_section(source_root, root, tag, prefixes):
    source = source_root.find(tag)
    if source is None:
        return
    destination = ET.SubElement(root, tag)
    for prefix in prefixes:
        for source_item in source:
            item = copy.deepcopy(source_item)
            if item.get("name"):
                item.set("name", prefix + item.get("name"))
            for attribute in REFERENCE_ATTRIBUTES:
                if item.get(attribute):
                    item.set(attribute, prefix + item.get(attribute))
            destination.append(item)


def build_scene(config_path=DEFAULT_CONFIG, output_path=GENERATED_MODEL,
                payload_mass_kg=None, payload_com_offset_m=None):
    config = load_config(config_path)
    model_config = config["model"]
    if model_config.get("robot_type") != "WFYG_TRON2A":
        raise ValueError("robot_type must be WFYG_TRON2A")
    if payload_mass_kg is not None:
        model_config["payload_mass_kg"] = float(payload_mass_kg)
    if payload_com_offset_m is not None:
        model_config["payload_com_offset_m"] = [
            float(value) for value in payload_com_offset_m
        ]
    if not SOURCE_MJCF.exists():
        raise FileNotFoundError("TRON2 source MJCF is missing: " + str(SOURCE_MJCF))

    source_root = ET.parse(SOURCE_MJCF).getroot()
    source_body = source_root.find("./worldbody/body[@name='base_Link']")
    if source_body is None:
        raise ValueError("base_Link was not found in WFYG_TRON2A MJCF")

    root = ET.Element("mujoco", {"model": "dual_tron2_cooperative_carry"})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(root, "size", {"njmax": "10000", "nconmax": "2000"})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": str(model_config["timestep"]),
            "integrator": "implicitfast",
            "solver": "Newton",
            "iterations": "80",
            "tolerance": "1e-9",
        },
    )
    default = source_root.find("default")
    if default is not None:
        root.append(copy.deepcopy(default))

    asset = ET.SubElement(root, "asset")
    for mesh in source_root.findall("./asset/mesh"):
        filename = mesh.get("file")
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": "tron2_" + mesh.get("name"),
                "file": _absolute(SOURCE_MESH_DIR / filename),
            },
        )

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "10 10 0.25",
            "rgba": "0.42 0.45 0.48 1",
            "friction": "1.0 0.02 0.002",
            "condim": "3",
        },
    )
    ET.SubElement(worldbody, "light", {"directional": "true", "pos": "0 0 6", "dir": "0 0 -1"})
    ET.SubElement(worldbody, "camera", {"name": "overview", "pos": "1.6 -3.2 2.1", "xyaxes": "0.9 0.4 0 -0.2 0.45 0.87"})
    worldbody.append(_copy_robot(source_body, "r1_", model_config["robot_1_pose"]))
    worldbody.append(_copy_robot(source_body, "r2_", model_config["robot_2_pose"]))
    _add_payload(worldbody, model_config)

    _copy_named_section(source_root, root, "actuator", ("r1_", "r2_"))
    _copy_named_section(source_root, root, "sensor", ("r1_", "r2_"))

    equality = ET.SubElement(root, "equality")
    fixed = str(bool(model_config.get("fixed_bases", True))).lower()
    grasp = str(model_config.get("grasp_mode", "soft_weld") != "none").lower()
    for prefix in ("r1_", "r2_"):
        ET.SubElement(
            equality,
            "weld",
            {
                "name": prefix + "base_fix",
                "body1": prefix + "base_Link",
                "active": fixed,
                "solref": "0.002 1",
                "solimp": "0.95 0.99 0.001",
            },
        )
    for prefix, side in (("r1_", "left"), ("r2_", "right")):
        ET.SubElement(
            equality,
            "weld",
            {
                "name": prefix + "grasp_weld",
                "body1": prefix + "gripper_pick",
                "body2": "payload_grasp_" + side,
                "active": grasp,
                "solref": "0.005 1",
                "solimp": "0.95 0.99 0.001",
            },
        )

    _indent(root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=GENERATED_MODEL)
    parser.add_argument("--payload-mass", type=float)
    parser.add_argument("--payload-com", nargs=3, type=float)
    args = parser.parse_args()
    print(build_scene(args.config, args.output, args.payload_mass, args.payload_com))


if __name__ == "__main__":
    main()
