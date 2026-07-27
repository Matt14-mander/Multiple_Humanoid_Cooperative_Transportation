from pathlib import Path

import mujoco
import numpy as np
import pytest

from dual_tron1_mujoco.build_scene import build_scene
from dual_tron1_mujoco.commands import (
    DualCommandCoordinator,
    FormationHoldCoordinator,
    VelocitySchedule,
)
from dual_tron1_mujoco.model_index import ModelIndex
from dual_tron1_mujoco.model_loader import load_model
from dual_tron1_mujoco.observation import ObservationBuilder
from dual_tron1_mujoco.paths import FORWARD_CONFIG, POLICY_DIR
from dual_tron1_mujoco.configuration import load_config
from dual_tron1_mujoco.policy_controller import (
    OnnxPolicyBackend,
    RobotPolicyController,
)


class ZeroPolicyBackend:
    def infer(self, observation, history):
        return np.zeros(14)


@pytest.fixture()
def forward_model(tmp_path: Path):
    output = build_scene(FORWARD_CONFIG, tmp_path / "forward.xml")
    model = load_model(output)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_observation_and_history_match_exported_shapes(forward_model):
    model, data = forward_model
    builder = ObservationBuilder(model, "r1_")
    observation = builder.build(data, np.zeros(3), np.zeros(14))
    assert observation.shape == (74,)
    np.testing.assert_allclose(observation[3:6], [0.0, 0.0, -1.0])
    assert observation[9] == 1.0
    history = builder.push_history(observation)
    assert history.shape == (1, 370)
    for frame in history.reshape(5, 74):
        np.testing.assert_allclose(frame, observation)


def test_each_robot_has_independent_history(forward_model):
    model, data = forward_model
    first = ObservationBuilder(model, "r1_")
    second = ObservationBuilder(model, "r2_")
    first.push_history(first.build(data, [0.1, 0, 0], np.zeros(14)))
    second.push_history(second.build(data, [0.0, 0, 0], np.zeros(14)))
    assert not np.shares_memory(first.history, second.history)
    assert not np.array_equal(first.history, second.history)


def test_world_command_is_converted_using_each_heading(forward_model):
    model, data = forward_model
    coordinator = DualCommandCoordinator(
        model, VelocitySchedule(world_vx=0.1, start_s=0.0, stop_s=1.0)
    )
    assert coordinator.command(data, "r1_")[0] == pytest.approx(0.1)
    assert coordinator.command(data, "r2_")[0] == pytest.approx(0.1)

    r2_root = ModelIndex(model).joint("r2_root_free")
    data.qpos[r2_root.qpos_adr + 3 : r2_root.qpos_adr + 7] = [0, 0, 0, 1]
    mujoco.mj_forward(model, data)
    assert coordinator.command(data, "r2_")[0] == pytest.approx(-0.1)


def test_real_onnx_backend_returns_one_action_per_actuator():
    pytest.importorskip("onnxruntime")
    backend = OnnxPolicyBackend(POLICY_DIR)
    actions = backend.infer(
        np.zeros(74, dtype=np.float32),
        np.zeros((1, 370), dtype=np.float32),
    )
    assert actions.shape == (14,)
    assert np.all(np.isfinite(actions))


def test_policy_can_leave_arm_actuators_for_impedance_controller(forward_model):
    model, data = forward_model
    config = load_config(FORWARD_CONFIG)["control"]
    controller = RobotPolicyController(
        model, "r1_", config, ZeroPolicyBackend()
    )
    index = ModelIndex(model)
    arm_actuators = [
        index.actuator("r1_" + joint)
        for joint in ("J1", "J2", "J3", "J4", "J5", "J6")
    ]
    data.ctrl[arm_actuators] = 1.234

    controller.update(data, np.zeros(3), actuate_arms=False)

    np.testing.assert_allclose(data.ctrl[arm_actuators], 1.234)


def test_formation_hold_corrects_forward_and_lateral_drift(forward_model):
    model, data = forward_model
    coordinator = FormationHoldCoordinator(
        model,
        data,
        {"start_s": 0.0, "filter_time_constant_s": 0.0},
    )
    index = ModelIndex(model)
    r1_root = index.joint("r1_root_free")

    data.qpos[r1_root.qpos_adr] -= 0.20
    data.qpos[r1_root.qpos_adr + 1] += 0.20
    mujoco.mj_forward(model, data)
    command = coordinator.command(data, "r1_")

    assert command[0] > 0.0
    assert command[1] == 0.0
    assert command[2] < 0.0
