from pathlib import Path

import numpy as np
import pytest

from dual_tron1_mujoco.build_scene import build_scene
from dual_tron1_mujoco.carry_controller import CooperativeCarryHoldController
from dual_tron1_mujoco.configuration import load_config
from dual_tron1_mujoco.model_loader import load_model
from dual_tron1_mujoco.paths import CARRY_HOLD_CONFIG
from dual_tron1_mujoco.internal_force_controller import (
    MujocoInternalForceController,
)
from internal_force_suppression.config.ifsm_config import IFSMConfig


mujoco = pytest.importorskip("mujoco")


def _controller(tmp_path: Path, payload_mass=2.0):
    output = build_scene(
        CARRY_HOLD_CONFIG,
        tmp_path / "carry_hold.xml",
        payload_mass_kg=payload_mass,
    )
    model = load_model(output)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    config = load_config(CARRY_HOLD_CONFIG)
    controller_config = dict(config["control"])
    controller_config.update(config["carry_impedance"])
    controller = CooperativeCarryHoldController(
        model, data, controller_config
    )
    return model, data, controller


def test_static_gravity_wrench_is_shared_symmetrically(tmp_path: Path):
    _, data, controller = _controller(tmp_path, payload_mass=2.0)
    controller.update(data)

    assert controller.last_payload_wrench[2] == pytest.approx(19.62, rel=1e-3)
    assert controller.last_arm_wrenches[0, 2] == pytest.approx(9.81, rel=1e-3)
    assert controller.last_arm_wrenches[1, 2] == pytest.approx(9.81, rel=1e-3)
    assert np.allclose(
        controller.last_arm_wrenches[0],
        controller.last_arm_wrenches[1],
        atol=1e-8,
    )


def test_arm_commands_are_finite_and_within_limits(tmp_path: Path):
    _, data, controller = _controller(tmp_path, payload_mass=2.0)
    controller.update(data)

    assert np.all(np.isfinite(controller.last_arm_torques))
    assert np.max(np.abs(controller.last_arm_torques[:, :3])) <= 18.0
    assert np.max(np.abs(controller.last_arm_torques[:, 3:])) <= 3.0
    assert controller.update_count == 1
    assert np.all(controller.saturation_fractions >= 0.0)
    assert np.all(controller.saturation_fractions <= 1.0)


def test_mujoco_ifsm_reconstructs_and_suppresses_internal_wrench(tmp_path: Path):
    model, data, carry = _controller(tmp_path, payload_mass=2.0)
    ifsm = MujocoInternalForceController(model, IFSMConfig().to_dict())

    for _ in range(20):
        correction, diagnostics = ifsm.update(data, model.opt.timestep)
        carry.update(data, correction)
        mujoco.mj_step(model, data)

    assert diagnostics["contact_wrenches"].shape == (2, 6)
    assert diagnostics["internal_wrenches"].shape == (2, 6)
    assert diagnostics["correction_wrenches"].shape == (2, 6)
    assert np.all(np.isfinite(diagnostics["contact_wrenches"]))
    assert np.all(np.isfinite(diagnostics["correction_wrenches"]))
    assert np.linalg.norm(
        ifsm.last_force_info["grasp_matrix"]
        @ ifsm.last_force_info["F_internal"]
    ) < 1e-8
    assert np.max(np.linalg.norm(correction[:, :3], axis=1)) <= 5.0 + 1e-9
    assert np.max(np.linalg.norm(correction[:, 3:], axis=1)) <= 1.0 + 1e-9
