from pathlib import Path

import numpy as np

from tron2_chip.backends.isaaclab.chip_terms import actor_goal
from tron2_chip.backends.isaaclab.task_cfg import FixedBaseChipTaskCfg
from tron2_chip.core.action_scaling import ActionScaler
from tron2_chip.core.observations import ActorObservationBuilder
from tron2_chip.core.policy_spec import PolicySpec
from tron2_chip.deployment.policy_runtime import CallablePolicyBackend, PolicyRuntime
from tron2_chip.deployment.safety_filter import ActionSafetyFilter
from tron2_chip.evaluation.metrics import compliance_metrics
from tron2_chip.paths import DEFAULT_CONFIG
from tron2_chip.run_deployment import run


def test_policy_spec_round_trip_and_observation_shape(tmp_path: Path):
    spec = PolicySpec.fixed_base_arm(history_steps=3, control_dt_s=0.02)
    path = tmp_path / "policy_spec.json"
    spec.save(path)
    loaded = PolicySpec.load(path)
    assert loaded == spec
    assert loaded.sha256 == spec.sha256

    builder = ActorObservationBuilder(spec)
    proprioception = np.zeros(len(spec.proprioception_names))
    builder.reset(proprioception)
    observation = builder.build(np.zeros(3), np.full(3, 0.002))
    assert observation.shape == (spec.actor_observation_size,)


def test_policy_runtime_applies_slew_rate_and_action_scaling():
    spec = PolicySpec.fixed_base_arm(history_steps=1)
    backend = CallablePolicyBackend(lambda _: np.full(spec.action_size, 2.0))
    safety = ActionSafetyFilter(spec.action_size, magnitude_limit=1.0, rate_limit_per_s=5.0)
    runtime = PolicyRuntime(spec, backend, safety_filter=safety)
    action = runtime.infer(np.zeros(spec.actor_observation_size), dt_s=0.02)
    np.testing.assert_allclose(action, 0.1)
    scaler = ActionScaler(np.full(spec.action_size, 0.25))
    np.testing.assert_allclose(scaler.residual(action), 0.025)


def test_training_and_deployment_actor_goals_are_separated():
    reference = np.array([[0.4, 0.0, 1.0]])
    force = np.array([[10.0, 0.0, 0.0]])
    compliance = np.array([[0.002, 0.0, 0.0]])
    np.testing.assert_allclose(
        actor_goal(reference, force, compliance, training=True),
        [[0.38, 0.0, 1.0]],
    )
    np.testing.assert_allclose(
        actor_goal(reference, force, compliance, training=False), reference
    )


def test_isaaclab_task_contract_matches_policy_control_period():
    task = FixedBaseChipTaskCfg(num_envs=32)
    task.validate()
    spec = PolicySpec.fixed_base_arm(
        history_steps=task.history_steps, control_dt_s=task.control_dt_s
    )
    assert task.control_dt_s == 0.02
    assert spec.control_dt_s == task.control_dt_s


def test_deployment_mode_yields_by_commanded_compliance(tmp_path: Path):
    csv_path = tmp_path / "deployment.csv"
    result = run(
        config_path=DEFAULT_CONFIG,
        model_path=tmp_path / "single.xml",
        headless=True,
        duration_s=1.2,
        rebuild=True,
        compliance=(0.002, 0.0, 0.0),
        force=(10.0, 0.0, 0.0),
        record_path=csv_path,
        quiet=True,
    )
    metrics = compliance_metrics(csv_path, 0.002, "x")
    assert result["goal_mode"] == "deployment"
    np.testing.assert_allclose(result["expected_goal_shift"], 0.0)
    np.testing.assert_allclose(result["expected_response"], [0.02, 0.0, 0.0])
    assert metrics["relative_error"] < 0.1
    assert metrics["peak_control_fraction"] < 0.25
