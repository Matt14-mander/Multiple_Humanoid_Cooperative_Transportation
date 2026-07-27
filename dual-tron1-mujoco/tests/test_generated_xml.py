import xml.etree.ElementTree as ET
from pathlib import Path

from dual_tron1_mujoco.build_scene import build_scene
from dual_tron1_mujoco.paths import DEFAULT_CONFIG


def test_generated_xml_has_unique_names_and_assets(tmp_path: Path):
    output = build_scene(DEFAULT_CONFIG, tmp_path / "scene.xml")
    root = ET.parse(str(output)).getroot()
    # MuJoCo allows the same string in different object namespaces (for
    # example a joint, its actuator and its position sensor).  Names must only
    # be unique within a namespace/tag.
    for tag in ("body", "joint", "geom", "site", "mesh", "motor", "weld"):
        names = [item.get("name") for item in root.iter(tag) if item.get("name")]
        assert len(names) == len(set(names)), tag
    assert root.find("./worldbody/body[@name='r1_base_Link']") is not None
    assert root.find("./worldbody/body[@name='r2_base_Link']") is not None
    assert root.find("./worldbody/body[@name='payload_body']") is not None
    assert len(root.findall("./actuator/motor")) == 28
    for mesh in root.findall("./asset/mesh"):
        assert Path(mesh.get("file")).exists()


def test_expected_equalities_are_present(tmp_path: Path):
    output = build_scene(DEFAULT_CONFIG, tmp_path / "scene.xml")
    root = ET.parse(str(output)).getroot()
    names = {item.get("name") for item in root.findall("./equality/weld")}
    assert names == {
        "r1_base_fix",
        "r2_base_fix",
        "r1_grasp_weld",
        "r2_grasp_weld",
    }
