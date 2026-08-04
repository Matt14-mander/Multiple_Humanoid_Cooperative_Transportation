from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

from dual_tron1_mujoco.build_scene import build_scene
from dual_tron1_mujoco.configuration import load_config
from dual_tron1_mujoco.model_loader import load_model
from dual_tron1_mujoco.mujoco_momentum_observer import (
    MujocoDualArmMomentumObserver,
)
from dual_tron1_mujoco.paths import CARRY_HOLD_CONFIG


def _observer_model(tmp_path: Path):
    output = build_scene(
        CARRY_HOLD_CONFIG,
        tmp_path / "observer_carry_hold.xml",
        payload_mass_kg=0.5,
    )
    model = load_model(output)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    config = load_config(CARRY_HOLD_CONFIG)
    return model, data, MujocoDualArmMomentumObserver(
        model, config["momentum_observer"]
    )


def test_full_model_fixed_tool_is_merged_into_observer_dynamics(tmp_path: Path):
    model, data, observer = _observer_model(tmp_path)

    for arm in observer.arms:
        q = np.array([data.qpos[j.qpos_adr] for j in arm["joints"]])
        v = np.array([data.qvel[j.dof_adr] for j in arm["joints"]])
        pin_data = arm["pin_model"].createData()
        pin_bias = pin.rnea(
            arm["pin_model"], pin_data, q, v, np.zeros(6)
        )

        assert np.isclose(arm["tool"]["mass_kg"], 0.1176)
        assert np.allclose(
            pin_bias, data.qfrc_bias[arm["dofs"]], atol=2e-7
        )


def test_observer_scores_all_mujoco_constraint_forces(tmp_path: Path):
    _, data, observer = _observer_model(tmp_path)
    arm = observer.arms[0]
    expected = np.arange(6, dtype=float)
    data.qfrc_constraint[arm["dofs"]] = expected

    assert np.array_equal(observer._constraint_joint_truth(data, arm), expected)
