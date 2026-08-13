from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
TRON2_ROOT = WORKSPACE_ROOT / "tron2-robot-description" / "tron2" / "WFYG_TRON2A"
SOURCE_MJCF = TRON2_ROOT / "xml" / "robot.xml"
SOURCE_URDF = TRON2_ROOT / "urdf" / "robot.urdf"
SOURCE_MESH_DIR = TRON2_ROOT / "meshes"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "wfyg_dual_hold.json"
GENERATED_MODEL = PROJECT_ROOT / "models" / "generated" / "dual_tron2_payload.xml"

