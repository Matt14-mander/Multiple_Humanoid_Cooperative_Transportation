"""Generate a self-contained MJCF for two WF_TRON1A robots and one payload.

The TRON1 wheel-leg subtree is taken from tron1-mujoco-sim.  The AIRBOT arm
chain, inertias, joint axes and limits are read from tron1-rl-deploy-arm's
robot_with_arm.urdf.  Mesh paths are written as absolute paths so the generated
model is independent of the current working directory on Windows.
"""

import argparse
import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Optional

from .configuration import load_config
from .paths import ARM_MESH_DIR, ARM_URDF, BASE_MESH_DIR, BASE_MJCF, GENERATED_MODEL


ARM_CHAIN = [
    "airbot_arm_base_joint",
    "airbot_arm_joint",
    "J1",
    "J2",
    "J3",
    "J4",
    "J5",
    "J6",
]


def _numbers(value: Optional[str], default: str = "0 0 0") -> str:
    return value if value else default


def _half_size(value: str) -> str:
    return " ".join(str(float(item) / 2.0) for item in value.split())


def _urdf_rpy_quat(value: Optional[str]) -> str:
    """Convert URDF fixed-axis roll/pitch/yaw to MuJoCo w-x-y-z quat."""

    roll, pitch, yaw = [float(item) for item in _numbers(value).split()]
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return "{} {} {} {}".format(
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _origin_pose(origin: Optional[ET.Element]) -> Dict[str, str]:
    return {
        "pos": _numbers(origin.get("xyz") if origin is not None else None),
        "quat": _urdf_rpy_quat(
            origin.get("rpy") if origin is not None else None
        ),
    }


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


def _absolute(path: Path) -> str:
    return path.resolve().as_posix()


def _prefix_robot_subtree(body: ET.Element, prefix: str) -> None:
    reference_attributes = {"joint", "objname", "site", "body1", "body2"}
    for element in body.iter():
        if "name" in element.attrib:
            element.set("name", prefix + element.get("name"))
        for attribute in reference_attributes:
            if attribute in element.attrib:
                element.set(attribute, prefix + element.get(attribute))
        if element.tag == "geom" and "mesh" in element.attrib:
            element.set("mesh", "tron1_" + element.get("mesh"))

    free_joint = body.find("joint[@type='free']")
    if free_joint is not None and "name" not in free_joint.attrib:
        free_joint.set("name", prefix + "root_free")


def _add_inertial(body: ET.Element, link: ET.Element) -> None:
    inertial = link.find("inertial")
    if inertial is None:
        return
    origin = inertial.find("origin")
    mass = inertial.find("mass")
    inertia = inertial.find("inertia")
    if mass is None or inertia is None:
        return
    attributes = {
        "pos": _numbers(origin.get("xyz") if origin is not None else None),
        "mass": mass.get("value"),
        "fullinertia": "{} {} {} {} {} {}".format(
            inertia.get("ixx"),
            inertia.get("iyy"),
            inertia.get("izz"),
            inertia.get("ixy"),
            inertia.get("ixz"),
            inertia.get("iyz"),
        ),
    }
    if origin is not None and _numbers(origin.get("rpy")) not in {
        "0 0 0",
        "0. 0. 0.",
    }:
        raise ValueError(
            "Non-zero URDF inertial rotations require rotating fullinertia"
        )
    ET.SubElement(body, "inertial", attributes)


def _add_visual(body: ET.Element, link: ET.Element) -> None:
    visual = link.find("visual")
    if visual is None:
        return
    origin = visual.find("origin")
    geometry = visual.find("geometry")
    if geometry is None:
        return
    common = {"class": "visual", **_origin_pose(origin)}
    mesh = geometry.find("mesh")
    box = geometry.find("box")
    if mesh is not None:
        common["type"] = "mesh"
        common["mesh"] = "arm_" + link.get("name")
    elif box is not None:
        common["type"] = "box"
        common["size"] = _half_size(box.get("size"))
    else:
        return
    ET.SubElement(body, "geom", common)


def _add_collision(body: ET.Element, link: ET.Element, prefix: str) -> None:
    collision = link.find("collision")
    if collision is None:
        return
    geometry = collision.find("geometry")
    box = geometry.find("box") if geometry is not None else None
    if box is None:
        return
    origin = collision.find("origin")
    ET.SubElement(
        body,
        "geom",
        {
            "name": prefix + link.get("name") + "_collision",
            "type": "box",
            "size": _half_size(box.get("size")),
            **_origin_pose(origin),
            "rgba": "0.2 0.7 0.9 0.15",
            "friction": "0.8 0.01 0.001",
        },
    )


def _append_arm(base_body: ET.Element, prefix: str, urdf_root: ET.Element) -> None:
    links: Dict[str, ET.Element] = {
        element.get("name"): element for element in urdf_root.findall("link")
    }
    joints: Dict[str, ET.Element] = {
        element.get("name"): element for element in urdf_root.findall("joint")
    }
    parent_body = base_body
    for joint_name in ARM_CHAIN:
        source_joint = joints[joint_name]
        child_name = source_joint.find("child").get("link")
        child_link = links[child_name]
        origin = source_joint.find("origin")
        child_body = ET.SubElement(
            parent_body,
            "body",
            {"name": prefix + child_name, **_origin_pose(origin)},
        )
        _add_inertial(child_body, child_link)
        _add_visual(child_body, child_link)
        _add_collision(child_body, child_link, prefix)

        if source_joint.get("type") in {"revolute", "continuous"}:
            axis = source_joint.find("axis")
            limit = source_joint.find("limit")
            attributes = {
                "name": prefix + joint_name,
                "type": "hinge",
                "axis": _numbers(axis.get("xyz") if axis is not None else None, "0 0 1"),
                "damping": "0.01",
                "armature": "0.01",
            }
            if limit is not None and source_joint.get("type") != "continuous":
                attributes.update(
                    {
                        "limited": "true",
                        "range": f"{limit.get('lower')} {limit.get('upper')}",
                    }
                )
            ET.SubElement(child_body, "joint", attributes)

        parent_body = child_body

    ET.SubElement(
        parent_body,
        "site",
        {"name": prefix + "ee_site", "size": "0.025", "rgba": "1 0.2 0.2 1"},
    )
    gripper = ET.SubElement(
        parent_body,
        "body",
        {"name": prefix + "gripper_stub", "pos": "0 0 -0.08"},
    )
    ET.SubElement(
        gripper,
        "geom",
        {
            "name": prefix + "gripper_palm",
            "type": "box",
            "size": "0.035 0.035 0.012",
            "rgba": "0.25 0.25 0.25 1",
        },
    )


def _copy_robot(
    source_body: ET.Element,
    prefix: str,
    pose: Iterable[float],
    urdf_root: ET.Element,
) -> ET.Element:
    x, y, z, yaw = [float(value) for value in pose]
    body = copy.deepcopy(source_body)
    _prefix_robot_subtree(body, prefix)
    body.set("pos", f"{x} {y} {z}")
    body.set("quat", f"{math.cos(yaw / 2.0)} 0 0 {math.sin(yaw / 2.0)}")
    _append_arm(body, prefix, urdf_root)
    return body


def _add_payload(worldbody: ET.Element, model_config: Dict[str, object]) -> None:
    px, py, pz, yaw = [float(v) for v in model_config["payload_pose"]]
    sx, sy, sz = [float(v) for v in model_config["payload_body_size_m"]]
    hx, hy, hz = [float(v) for v in model_config["handle_size_m"]]
    total_mass = float(model_config["payload_mass_kg"])
    handle_mass = min(0.5, total_mass * 0.05)
    body_mass = total_mass - 2.0 * handle_mass
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
        "geom",
        {
            "name": "payload_collision",
            "type": "box",
            "size": f"{sx / 2} {sy / 2} {sz / 2}",
            "mass": str(body_mass),
            "rgba": "0.25 0.45 0.75 1",
            "friction": "0.8 0.01 0.001",
        },
    )
    handle_axis = str(model_config.get("handle_axis", "x")).lower()
    if handle_axis not in {"x", "y"}:
        raise ValueError("model.handle_axis must be 'x' or 'y'")
    if "handle_center_m" in model_config:
        handle_center = float(model_config["handle_center_m"])
    else:
        handle_center = float(model_config["handle_center_x_m"])
    for name, sign in (("left", -1.0), ("right", 1.0)):
        handle_pos = (
            f"{sign * handle_center} 0 0"
            if handle_axis == "x"
            else f"0 {sign * handle_center} 0"
        )
        handle = ET.SubElement(
            payload,
            "body",
            {"name": "payload_grasp_" + name, "pos": handle_pos},
        )
        ET.SubElement(
            handle,
            "geom",
            {
                "name": "payload_handle_" + name,
                "type": "box",
                "size": f"{hx / 2} {hy / 2} {hz / 2}",
                "mass": str(handle_mass),
                "rgba": "0.85 0.65 0.15 1",
                "friction": "1.0 0.02 0.002",
            },
        )
        ET.SubElement(
            handle,
            "site",
            {"name": "payload_grasp_" + name + "_site", "size": "0.02", "rgba": "0 1 0 1"},
        )


def build_scene(
    config_path: Path,
    output_path: Path,
    payload_mass_kg: float = None,
) -> Path:
    config = load_config(config_path)
    model_config = config["model"]
    if payload_mass_kg is not None:
        if payload_mass_kg <= 0.0:
            raise ValueError("payload_mass_kg must be positive")
        model_config["payload_mass_kg"] = float(payload_mass_kg)
    if model_config.get("robot_type") != "WF_TRON1A":
        raise ValueError("The first Windows milestone supports WF_TRON1A only")
    if not BASE_MJCF.exists() or not ARM_URDF.exists():
        raise FileNotFoundError("Required upstream TRON1 model assets are missing")

    base_root = ET.parse(str(BASE_MJCF)).getroot()
    urdf_root = ET.parse(str(ARM_URDF)).getroot()
    source_body = base_root.find("./worldbody/body[@name='base_Link']")
    if source_body is None:
        raise ValueError("base_Link was not found in the upstream WF_TRON1A MJCF")

    root = ET.Element("mujoco", {"model": "dual_tron1_cooperative_carry"})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(root, "size", {"njmax": "2000", "nconmax": "500"})
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
    default = base_root.find("default")
    if default is not None:
        root.append(copy.deepcopy(default))

    asset = ET.SubElement(root, "asset")
    for source_mesh in base_root.findall("./asset/mesh"):
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": "tron1_" + source_mesh.get("name"),
                "file": _absolute(BASE_MESH_DIR / source_mesh.get("file")),
            },
        )
    arm_mesh_names = ["airbot_arm", "link1", "link2", "link3", "link4", "link5", "link6"]
    for mesh_name in arm_mesh_names:
        mesh_path = ARM_MESH_DIR / f"{mesh_name}.STL"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing AIRBOT mesh: {mesh_path}")
        ET.SubElement(
            asset,
            "mesh",
            {"name": "arm_" + mesh_name, "file": _absolute(mesh_path)},
        )

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "10 10 0.25",
            "rgba": "0.45 0.48 0.52 1",
            "friction": "0.9 0.01 0.001",
            "condim": "3",
        },
    )
    ET.SubElement(worldbody, "light", {"directional": "true", "pos": "0 0 8", "dir": "0 0 -1"})
    ET.SubElement(worldbody, "camera", {"name": "overview", "pos": "0 -4 2.4", "xyaxes": "1 0 0 0 0.5 0.866"})
    worldbody.append(_copy_robot(source_body, "r1_", model_config["robot_1_pose"], urdf_root))
    worldbody.append(_copy_robot(source_body, "r2_", model_config["robot_2_pose"], urdf_root))
    _add_payload(worldbody, model_config)

    actuator = ET.SubElement(root, "actuator")
    for prefix in ("r1_", "r2_"):
        for source_motor in base_root.findall("./actuator/motor"):
            motor = copy.deepcopy(source_motor)
            motor.set("name", prefix + source_motor.get("name"))
            motor.set("joint", prefix + source_motor.get("joint"))
            actuator.append(motor)
        for joint_name, limit in (
            ("J1", 18), ("J2", 18), ("J3", 18),
            ("J4", 3), ("J5", 3), ("J6", 3),
        ):
            ET.SubElement(
                actuator,
                "motor",
                {
                    "name": prefix + joint_name,
                    "joint": prefix + joint_name,
                    "gear": "1",
                    "ctrllimited": "true",
                    "ctrlrange": f"{-limit} {limit}",
                },
            )

    sensor = ET.SubElement(root, "sensor")
    for prefix in ("r1_", "r2_"):
        for source_sensor in base_root.findall("./sensor/*"):
            item = copy.deepcopy(source_sensor)
            if item.get("name"):
                item.set("name", prefix + item.get("name"))
            for attribute in ("joint", "site", "objname"):
                if item.get(attribute):
                    item.set(attribute, prefix + item.get(attribute))
            sensor.append(item)
        for joint_name in ("J1", "J2", "J3", "J4", "J5", "J6"):
            ET.SubElement(sensor, "jointpos", {"name": prefix + joint_name + "_q", "joint": prefix + joint_name})
            ET.SubElement(sensor, "jointvel", {"name": prefix + joint_name + "_dq", "joint": prefix + joint_name})

    equality = ET.SubElement(root, "equality")
    fixed_bases = bool(model_config.get("fixed_bases", True))
    grasp_active = model_config.get("grasp_mode", "soft_weld") != "none"
    for prefix in ("r1_", "r2_"):
        ET.SubElement(
            equality,
            "weld",
            {
                "name": prefix + "base_fix",
                "body1": prefix + "base_Link",
                "active": str(fixed_bases).lower(),
                "solref": "0.002 1",
                "solimp": "0.95 0.99 0.001",
            },
        )
    for prefix, handle in (("r1_", "left"), ("r2_", "right")):
        ET.SubElement(
            equality,
            "weld",
            {
                "name": prefix + "grasp_weld",
                "body1": prefix + "link6",
                "body2": "payload_grasp_" + handle,
                "active": str(grasp_active).lower(),
                "solref": "0.02 1",
                "solimp": "0.85 0.95 0.01",
            },
        )

    _indent(root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(str(output_path), encoding="utf-8", xml_declaration=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=GENERATED_MODEL)
    args = parser.parse_args()
    config_path = args.config if args.config is not None else GENERATED_MODEL.parents[2] / "configs" / "wf_dual.json"
    print(build_scene(config_path, args.output))


if __name__ == "__main__":
    main()
