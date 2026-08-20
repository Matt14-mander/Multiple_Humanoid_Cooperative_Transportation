from pathlib import Path

import mujoco
import numpy as np

from tron2_chip.build_scene import build_scene
from tron2_chip.model_loader import load_model
from tron2_chip.paths import DEFAULT_CONFIG
from tron2_chip.run_sanity import run


def test_generated_single_model_has_expected_interface(tmp_path: Path):
    output = build_scene(DEFAULT_CONFIG, tmp_path / "single.xml")
    model = load_model(output)
    assert (model.nq, model.nv, model.nu) == (25, 24, 18)
    for kind, name in (
        (mujoco.mjtObj.mjOBJ_BODY, "base_Link"),
        (mujoco.mjtObj.mjOBJ_BODY, "gripper_pick"),
        (mujoco.mjtObj.mjOBJ_SITE, "ee_site"),
        (mujoco.mjtObj.mjOBJ_EQUALITY, "base_fix"),
    ):
        assert mujoco.mj_name2id(model, kind, name) >= 0


def test_short_hindsight_wiring_run_is_finite(tmp_path: Path):
    result = run(
        DEFAULT_CONFIG,
        tmp_path / "single.xml",
        headless=True,
        duration_s=1.1,
        rebuild=True,
        compliance=(0.002, 0.0, 0.0),
        force=(10.0, 0.0, 0.0),
        record_path=tmp_path / "run.csv",
    )
    assert np.all(np.isfinite(result["data"].qpos))
    np.testing.assert_allclose(result["expected_goal_shift"], [-0.02, 0.0, 0.0])
    assert result["peak_ee_offset"] < 1e-3
    assert result["peak_control_fraction"] < 0.25
    assert np.linalg.norm(result["final_displacement"]) < 1e-3
    assert result["csv"].exists()
