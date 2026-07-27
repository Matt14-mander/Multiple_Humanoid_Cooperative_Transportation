"""Load MJCF robustly from Windows paths, including paths with CJK text."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict

import mujoco


def load_model(model_path: Path) -> mujoco.MjModel:
    """Load XML and mesh bytes in Python before passing them to MuJoCo.

    On Windows, MuJoCo's native file opener can fail for otherwise valid paths
    containing non-ASCII characters.  Supplying an in-memory asset dictionary
    keeps the project runnable from a Chinese-named workspace.
    """

    path = Path(model_path).resolve()
    xml = path.read_text(encoding="utf-8")
    root = ET.fromstring(xml)
    assets: Dict[str, bytes] = {}
    for mesh in root.findall("./asset/mesh"):
        filename = mesh.get("file")
        if filename:
            mesh_path = Path(filename)
            if not mesh_path.is_absolute():
                mesh_path = path.parent / mesh_path
            assets[filename] = mesh_path.read_bytes()
    return mujoco.MjModel.from_xml_string(xml, assets)
