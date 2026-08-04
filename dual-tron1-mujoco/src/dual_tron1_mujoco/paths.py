from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
BASE_MJCF = (
    WORKSPACE_ROOT
    / "tron1-mujoco-sim"
    / "robot-description"
    / "pointfoot"
    / "WF_TRON1A"
    / "xml"
    / "robot.xml"
)
BASE_MESH_DIR = BASE_MJCF.parent.parent / "meshes"
ARM_URDF = (
    WORKSPACE_ROOT
    / "tron1-rl-deploy-arm"
    / "src"
    / "robot-description"
    / "pointfoot"
    / "WF_TRON1A"
    / "urdf"
    / "robot_with_arm.urdf"
)
ARM_MESH_DIR = ARM_URDF.parent.parent / "meshes"
POLICY_DIR = (
    WORKSPACE_ROOT
    / "tron1-rl-deploy-arm"
    / "src"
    / "robot_controllers"
    / "config"
    / "pointfoot"
    / "WF_TRON1A"
    / "policy"
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "wf_dual.json"
FORWARD_CONFIG = PROJECT_ROOT / "configs" / "wf_dual_forward.json"
CARRY_CONFIG = PROJECT_ROOT / "configs" / "wf_dual_carry.json"
CARRY_HOLD_CONFIG = PROJECT_ROOT / "configs" / "wf_dual_carry_hold.json"
CARRY_BALANCE_CONFIG = PROJECT_ROOT / "configs" / "wf_dual_carry_balance.json"
GENERATED_MODEL = PROJECT_ROOT / "models" / "generated" / "dual_tron1_payload.xml"
FORWARD_MODEL = PROJECT_ROOT / "models" / "generated" / "dual_tron1_forward.xml"
CARRY_MODEL = PROJECT_ROOT / "models" / "generated" / "dual_tron1_carry.xml"
CARRY_HOLD_MODEL = PROJECT_ROOT / "models" / "generated" / "dual_tron1_carry_hold.xml"
CARRY_BALANCE_MODEL = PROJECT_ROOT / "models" / "generated" / "dual_tron1_carry_balance.xml"
AIRBOT_OBSERVER_MODEL = PROJECT_ROOT / "models" / "generated" / "airbot_observer_validation.xml"
