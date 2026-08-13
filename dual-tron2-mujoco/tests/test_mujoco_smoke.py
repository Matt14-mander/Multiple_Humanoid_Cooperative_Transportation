from pathlib import Path

import mujoco
import numpy as np

from dual_tron2_mujoco.build_scene import build_scene
from dual_tron2_mujoco.configuration import load_config
from dual_tron2_mujoco.model_loader import load_model
from dual_tron2_mujoco.paths import DEFAULT_CONFIG
from dual_tron2_mujoco.run_sim import _apply_initial_joint_positions, run


def test_generated_dual_tron2_model_has_expected_interface(tmp_path: Path):
    output = build_scene(DEFAULT_CONFIG, tmp_path / "dual_tron2.xml")
    model = load_model(output)
    assert (model.nq, model.nv, model.nu) == (57, 54, 36)
    for prefix in ("r1_", "r2_"):
        for name in (
            "base_Link", "arm6_Link", "gripper_pick"
        ):
            assert mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, prefix + name
            ) >= 0
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, prefix + "grasp_weld"
        ) >= 0


def test_initial_gripper_and_payload_handles_are_aligned(tmp_path: Path):
    output = build_scene(DEFAULT_CONFIG, tmp_path / "aligned.xml")
    model = load_model(output)
    data = mujoco.MjData(model)
    _apply_initial_joint_positions(model, data, load_config(DEFAULT_CONFIG))
    mujoco.mj_forward(model, data)
    for prefix, side in (("r1_", "left"), ("r2_", "right")):
        gripper = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, prefix + "gripper_pick"
        )
        handle = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "payload_grasp_" + side
        )
        assert np.linalg.norm(data.xpos[gripper] - data.xpos[handle]) < 1e-5


def test_short_asymmetric_payload_run_is_stable(tmp_path: Path):
    result = run(
        DEFAULT_CONFIG,
        tmp_path / "run.xml",
        headless=True,
        duration_s=0.25,
        rebuild=True,
        payload_mass_kg=2.0,
        payload_com_offset_m=(0.05, -0.03, 0.02),
    )
    assert np.all(np.isfinite(result["data"].qpos))
    assert np.max(result["grasp_gaps"]) < 0.005
    assert np.linalg.norm(result["payload_displacement"]) < 0.03

