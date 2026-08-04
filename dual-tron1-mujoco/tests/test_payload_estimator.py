from pathlib import Path

import numpy as np
import pytest

from dual_tron1_mujoco.build_scene import build_scene
from dual_tron1_mujoco.carry_controller import CooperativeCarryHoldController
from dual_tron1_mujoco.configuration import load_config
from dual_tron1_mujoco.model_loader import load_model
from dual_tron1_mujoco.paths import CARRY_HOLD_CONFIG
from dual_tron1_mujoco.payload_estimator import (
    CALIBRATING,
    FROZEN,
    REIDENTIFICATION_REQUIRED,
    PayloadEstimatorConfig,
    WindowedPayloadEstimator,
    payload_regressor,
    skew,
)


mujoco = pytest.importorskip("mujoco")


def _rotation(roll: float, pitch: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]
    )
    rotation_y = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]
    )
    return rotation_y @ rotation_x


def _contact_sample(
    mass: float,
    com_body: np.ndarray,
    rotation: np.ndarray,
    acceleration: np.ndarray = np.zeros(3),
):
    gravity = np.array([0.0, 0.0, -9.81])
    theta = np.concatenate(([mass], mass * com_body))
    object_wrench = payload_regressor(rotation, acceleration, gravity) @ theta
    handle_offsets = np.array([[0.0, -0.35, 0.0], [0.0, 0.35, 0.0]])
    contact_points = (rotation @ handle_offsets.T).T
    grasp_matrix = np.zeros((6, 12))
    for index, point in enumerate(contact_points):
        column = 6 * index
        grasp_matrix[:3, column : column + 3] = np.eye(3)
        grasp_matrix[3:, column : column + 3] = skew(point)
        grasp_matrix[3:, column + 3 : column + 6] = np.eye(3)
    contact_wrenches = (
        np.linalg.pinv(grasp_matrix) @ object_wrench
    ).reshape(2, 6)
    return contact_wrenches, contact_points, gravity


def _identify(estimator, mass: float, com_body: np.ndarray):
    for index in range(36):
        roll = np.deg2rad(-5.0 + 10.0 * (index % 4) / 3.0)
        pitch = np.deg2rad(-4.0 + 8.0 * ((index // 4) % 4) / 3.0)
        rotation = _rotation(roll, pitch)
        contact_wrenches, contact_points, gravity = _contact_sample(
            mass, com_body, rotation
        )
        estimator.add_sample(
            contact_wrenches,
            contact_points,
            np.zeros(3),
            rotation,
            np.zeros(3),
            gravity,
        )


def test_single_static_pose_does_not_claim_full_com_observability():
    estimator = WindowedPayloadEstimator(
        PayloadEstimatorConfig(minimum_samples=4)
    )
    rotation = np.eye(3)
    contact_wrenches, contact_points, gravity = _contact_sample(
        3.0, np.array([0.08, -0.04, 0.06]), rotation
    )
    for _ in range(8):
        estimate = estimator.add_sample(
            contact_wrenches,
            contact_points,
            np.zeros(3),
            rotation,
            np.zeros(3),
            gravity,
        )

    assert estimate.observable_rank < 4
    assert not estimate.ready
    assert not estimator.freeze_if_ready()
    assert estimator.state == CALIBRATING


def test_multi_pose_window_estimates_mass_and_com_then_freezes():
    estimator = WindowedPayloadEstimator(
        PayloadEstimatorConfig(minimum_samples=20)
    )
    true_mass = 4.5
    true_com = np.array([0.09, -0.06, 0.04])
    _identify(estimator, true_mass, true_com)

    estimate = estimator.estimate
    assert estimate.ready
    assert estimate.observable_rank == 4
    assert estimate.mass_kg == pytest.approx(true_mass, rel=1e-5)
    assert np.allclose(estimate.com_body_m, true_com, atol=1e-5)
    assert estimator.freeze_if_ready()
    assert estimator.state == FROZEN


def test_frozen_estimate_does_not_drift_and_requests_reidentification():
    config = PayloadEstimatorConfig(
        minimum_samples=20,
        innovation_threshold=0.5,
        innovation_consecutive_samples=3,
    )
    estimator = WindowedPayloadEstimator(config)
    _identify(estimator, 2.0, np.array([0.04, 0.02, -0.03]))
    assert estimator.freeze_if_ready()
    frozen_theta = estimator.theta.copy()

    rotation = _rotation(0.05, -0.04)
    changed_wrenches, points, gravity = _contact_sample(
        5.0, np.array([-0.08, 0.07, 0.02]), rotation
    )
    for _ in range(3):
        estimator.add_sample(
            changed_wrenches,
            points,
            np.zeros(3),
            rotation,
            np.zeros(3),
            gravity,
        )

    assert np.array_equal(estimator.theta, frozen_theta)
    assert estimator.state == REIDENTIFICATION_REQUIRED


def test_frozen_estimate_ignores_isolated_transport_impulses():
    config = PayloadEstimatorConfig(
        minimum_samples=20,
        innovation_threshold=0.5,
        innovation_consecutive_samples=3,
    )
    estimator = WindowedPayloadEstimator(config)
    true_mass = 3.5
    true_com = np.array([0.06, -0.03, 0.02])
    _identify(estimator, true_mass, true_com)
    assert estimator.freeze_if_ready()
    frozen_theta = estimator.theta.copy()

    rotation = _rotation(0.03, -0.02)
    nominal, points, gravity = _contact_sample(
        true_mass, true_com, rotation
    )
    for index in range(300):
        contacts = nominal.copy()
        if index % 50 == 0:
            contacts[0, 2] += 5.0
        estimator.monitor(
            contacts,
            points,
            np.zeros(3),
            rotation,
            np.zeros(3),
            gravity,
        )

    assert np.array_equal(estimator.theta, frozen_theta)
    assert estimator.state == FROZEN


def test_estimated_offset_com_changes_vertical_load_distribution(tmp_path: Path):
    output = build_scene(
        CARRY_HOLD_CONFIG,
        tmp_path / "carry_hold.xml",
        payload_mass_kg=2.0,
    )
    model = load_model(output)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    config = load_config(CARRY_HOLD_CONFIG)
    controller_config = dict(config["control"])
    controller_config.update(config["carry_impedance"])
    controller = CooperativeCarryHoldController(model, data, controller_config)
    controller.set_payload_estimate(2.0, np.array([0.0, 0.10, 0.0]))

    controller.update(data)

    assert np.sum(controller.last_arm_wrenches[:, 2]) == pytest.approx(
        19.62, rel=1e-6
    )
    assert (
        controller.last_arm_wrenches[1, 2]
        > controller.last_arm_wrenches[0, 2]
    )


def test_scene_supports_repeatable_mass_and_com_demo_variants(tmp_path: Path):
    output = build_scene(
        CARRY_HOLD_CONFIG,
        tmp_path / "offset_payload.xml",
        payload_mass_kg=6.0,
        payload_com_offset_m=(0.08, -0.05, 0.03),
    )
    model = load_model(output)
    payload_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "payload_body"
    )

    assert model.body_subtreemass[payload_id] == pytest.approx(6.0)
    assert np.allclose(
        model.body_ipos[payload_id], np.array([0.08, -0.05, 0.03])
    )
