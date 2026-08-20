from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
TRON2_ROOT = WORKSPACE_ROOT / "tron2-robot-description" / "tron2" / "WFYG_TRON2A"
SOURCE_MJCF = TRON2_ROOT / "xml" / "robot.xml"
SOURCE_MESH_DIR = TRON2_ROOT / "meshes"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "arm_sanity.json"
GENERATED_MODEL = PROJECT_ROOT / "models" / "generated" / "single_wfyg_tron2a.xml"

