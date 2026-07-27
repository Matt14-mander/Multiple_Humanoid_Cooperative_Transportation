from pathlib import Path

import numpy as np
import pytest

from dual_tron1_mujoco.build_scene import build_scene
from dual_tron1_mujoco.paths import CARRY_CONFIG, DEFAULT_CONFIG
from dual_tron1_mujoco.model_loader import load_model


mujoco = pytest.importorskip("mujoco")


def test_model_compiles_with_expected_dimensions(tmp_path: Path):
    output = build_scene(DEFAULT_CONFIG, tmp_path / "scene.xml")
    model = load_model(output)
    assert (model.nq, model.nv, model.nu) == (49, 46, 28)
    assert model.neq == 4


def test_neutral_end_effectors_match_payload_handles(tmp_path: Path):
    output = build_scene(DEFAULT_CONFIG, tmp_path / "scene.xml")
    model = load_model(output)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for robot_name, handle_name in (
        ("r1_link6", "payload_grasp_left"),
        ("r2_link6", "payload_grasp_right"),
    ):
        robot_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, robot_name
        )
        handle_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, handle_name
        )
        assert np.linalg.norm(data.xpos[robot_id] - data.xpos[handle_id]) < 0.002


def test_short_headless_integration_stays_finite(tmp_path: Path):
    output = build_scene(DEFAULT_CONFIG, tmp_path / "scene.xml")
    model = load_model(output)
    data = mujoco.MjData(model)
    for _ in range(20):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))


def test_parallel_carry_formation_has_aligned_grasps(tmp_path: Path):
    output = build_scene(CARRY_CONFIG, tmp_path / "carry.xml")
    model = load_model(output)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    base_positions = []
    for prefix, handle_name in (
        ("r1_", "payload_grasp_left"),
        ("r2_", "payload_grasp_right"),
    ):
        base_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_Link"
        )
        ee_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, prefix + "link6"
        )
        handle_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, handle_name
        )
        base_positions.append(data.xpos[base_id].copy())
        assert np.linalg.norm(data.xpos[ee_id] - data.xpos[handle_id]) < 0.002

    assert abs(base_positions[0][0] - base_positions[1][0]) < 1e-9
    assert base_positions[0][1] < 0.0 < base_positions[1][1]
