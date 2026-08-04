from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

from dual_tron1_mujoco.airbot_observer_validation import (
    build_airbot_observer_model,
    build_airbot_pinocchio_model,
    run_validation_case,
)
from dual_tron1_mujoco.model_loader import load_model


def test_airbot_mujoco_and_pinocchio_dynamics_match(tmp_path: Path):
    model_path = build_airbot_observer_model(tmp_path / "airbot.xml")
    mj_model = load_model(model_path)
    mj_data = mujoco.MjData(mj_model)
    pin_model = build_airbot_pinocchio_model()
    pin_data = pin_model.createData()
    q = np.array([0.0, -1.10, 1.20, 0.0, 0.30, 0.0])
    v = np.array([0.10, -0.08, 0.06, -0.04, 0.03, -0.02])

    mj_data.qpos[:] = q
    mj_data.qvel[:] = v
    mujoco.mj_forward(mj_model, mj_data)
    mj_mass = np.zeros((mj_model.nv, mj_model.nv))
    mujoco.mj_fullM(mj_model, mj_mass, mj_data.qM)
    pin_mass = pin.crba(pin_model, pin_data, q)
    pin_bias = pin.rnea(pin_model, pin_data, q, v, np.zeros(6))

    assert np.allclose(mj_mass, pin_mass, atol=2e-8)
    assert np.allclose(mj_data.qfrc_bias, pin_bias, atol=2e-7)


def test_modeled_tool_inertia_matches_mujoco_plant(tmp_path: Path):
    model_path = build_airbot_observer_model(
        tmp_path / "airbot_tool.xml", unmodeled_tool_mass_kg=0.20
    )
    mj_model = load_model(model_path)
    mj_data = mujoco.MjData(mj_model)
    pin_model = build_airbot_pinocchio_model(tool_mass_kg=0.20)
    pin_data = pin_model.createData()
    q = np.array([0.0, -1.10, 1.20, 0.0, 0.30, 0.0])
    v = np.array([0.10, -0.08, 0.06, -0.04, 0.03, -0.02])

    mj_data.qpos[:] = q
    mj_data.qvel[:] = v
    mujoco.mj_forward(mj_model, mj_data)
    mj_mass = np.zeros((mj_model.nv, mj_model.nv))
    mujoco.mj_fullM(mj_model, mj_mass, mj_data.qM)
    pin_mass = pin.crba(pin_model, pin_data, q)
    pin_bias = pin.rnea(pin_model, pin_data, q, v, np.zeros(6))

    assert np.allclose(mj_mass, pin_mass, atol=2e-8)
    assert np.allclose(mj_data.qfrc_bias, pin_bias, atol=2e-7)


def test_pinocchio_model_error_terms_are_independently_perturbed():
    nominal = build_airbot_pinocchio_model()
    mass_error = build_airbot_pinocchio_model(mass_scale=0.90)
    com_error = build_airbot_pinocchio_model(com_offset_m=(0.01, 0.0, 0.0))
    gravity_error = build_airbot_pinocchio_model(gravity_scale=0.95)

    for joint_id in range(1, nominal.njoints):
        nominal_inertia = nominal.inertias[joint_id]
        mass_inertia = mass_error.inertias[joint_id]
        com_inertia = com_error.inertias[joint_id]

        assert np.isclose(mass_inertia.mass, 0.90 * nominal_inertia.mass)
        assert np.allclose(mass_inertia.lever, nominal_inertia.lever)
        assert np.allclose(mass_inertia.inertia, nominal_inertia.inertia)
        assert np.isclose(com_inertia.mass, nominal_inertia.mass)
        assert np.allclose(
            com_inertia.lever,
            nominal_inertia.lever + np.array([0.01, 0.0, 0.0]),
        )
        assert np.allclose(com_inertia.inertia, nominal_inertia.inertia)

    assert np.allclose(mass_error.gravity.linear, nominal.gravity.linear)
    assert np.allclose(com_error.gravity.linear, nominal.gravity.linear)
    assert np.allclose(
        gravity_error.gravity.linear, 0.95 * nominal.gravity.linear
    )


def test_zero_force_observer_has_negligible_false_positive(tmp_path: Path):
    result = run_validation_case(
        "zero_force", model_path=tmp_path / "airbot.xml"
    )

    assert result["joint_torque_rmse_nm"] < 0.01
    assert result["force_rmse_n"] < 0.05
    assert result["moment_rmse_nm"] < 0.02
    assert result["actuator_saturation_fraction"] == 0.0


def test_static_wrench_estimate_matches_mujoco_truth(tmp_path: Path):
    result = run_validation_case(
        "constant_wrench_static", model_path=tmp_path / "airbot.xml"
    )

    assert result["joint_torque_rmse_nm"] < 0.01
    assert result["force_rmse_n"] < 0.02
    assert result["moment_rmse_nm"] < 0.01
    assert result["force_direction_error_deg"] < 0.5
    assert result["response_time_90_s"] < 0.10
    assert result["actuator_saturation_fraction"] == 0.0


def test_moving_wrench_estimate_remains_accurate(tmp_path: Path):
    result = run_validation_case(
        "constant_wrench_motion", model_path=tmp_path / "airbot.xml"
    )

    assert result["joint_torque_rmse_nm"] < 0.03
    assert result["force_rmse_n"] < 0.15
    assert result["moment_rmse_nm"] < 0.04
    assert result["force_direction_error_deg"] < 3.0
    assert result["actuator_saturation_fraction"] == 0.0
