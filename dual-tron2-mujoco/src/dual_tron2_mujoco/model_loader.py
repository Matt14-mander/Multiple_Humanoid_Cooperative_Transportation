"""CJK-safe MuJoCo model loader."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


def load_model(model_path: Path) -> mujoco.MjModel:
    path = Path(model_path).resolve()
    xml = path.read_text(encoding="utf-8")
    root = ET.fromstring(xml)
    assets = {}
    for mesh in root.findall("./asset/mesh"):
        filename = mesh.get("file")
        if filename:
            assets[filename] = Path(filename).read_bytes()
    return mujoco.MjModel.from_xml_string(xml, assets)

