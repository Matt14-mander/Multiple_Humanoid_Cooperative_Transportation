"""Build the fixed-base, single-WFYG_TRON2A CHIP sanity scene."""

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from .configuration import load_config
from .paths import DEFAULT_CONFIG, GENERATED_MODEL, SOURCE_MESH_DIR, SOURCE_MJCF


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


def build_scene(config_path=DEFAULT_CONFIG, output_path=GENERATED_MODEL):
    config = load_config(config_path)
    model_config = config["model"]
    if model_config.get("robot_type") != "WFYG_TRON2A":
        raise ValueError("robot_type must be WFYG_TRON2A")
    if not SOURCE_MJCF.exists():
        raise FileNotFoundError("TRON2 source MJCF is missing: " + str(SOURCE_MJCF))

    root = copy.deepcopy(ET.parse(SOURCE_MJCF).getroot())
    root.set("model", "single_tron2_chip")
    compiler = root.find("compiler")
    compiler.set("meshdir", _absolute(SOURCE_MESH_DIR))
    for mesh in root.findall("./asset/mesh"):
        filename = mesh.get("file")
        if filename:
            mesh.set("file", _absolute(SOURCE_MESH_DIR / filename))
    option = root.find("option")
    option.set("timestep", str(model_config["timestep"]))
    option.set("integrator", "implicitfast")
    option.set("solver", "Newton")
    option.set("iterations", "80")

    base = root.find("./worldbody/body[@name='base_Link']")
    if base is None:
        raise ValueError("base_Link was not found in WFYG_TRON2A MJCF")
    base.set("pos", "{} {} {}".format(*model_config["base_position"]))
    for joint in base.findall(".//joint"):
        name = joint.get("name", "")
        if name.startswith("arm"):
            # The source CAD model omits reflected motor/gear inertia.  A small
            # armature term prevents unrealistically large wrist acceleration.
            joint.set("armature", "0.03" if name in {"arm4_Joint", "arm5_Joint", "arm6_Joint"} else "0.02")
            joint.set("damping", "0.15")
        elif name.startswith("gripper"):
            joint.set("armature", "0.002")
            joint.set("damping", "0.10")
    pick = base.find(".//body[@name='gripper_pick']")
    if pick is None:
        raise ValueError("gripper_pick was not found in WFYG_TRON2A MJCF")
    ET.SubElement(
        pick,
        "site",
        {"name": "ee_site", "size": "0.025", "rgba": "1 0.15 0.15 1"},
    )

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality,
        "weld",
        {
            "name": "base_fix",
            "body1": "base_Link",
            "active": str(bool(model_config.get("fixed_base", True))).lower(),
            "solref": "0.002 1",
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
    args = parser.parse_args()
    print(build_scene(args.config, args.output))


if __name__ == "__main__":
    main()
