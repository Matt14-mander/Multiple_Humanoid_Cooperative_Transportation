from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tron2_chip.backends.isaaclab.intent_terms import (  # noqa: E402
    BatchedIntentHistory,
    planar_intent_label,
    shift_wrench_reference,
)
from tron2_chip.intent_estimation.dataset import IntentDataset  # noqa: E402
from tron2_chip.intent_estimation.losses import physical_metrics, regression_loss  # noqa: E402
from tron2_chip.intent_estimation.network import (  # noqa: E402
    IntentEstimatorMLP,
    PhysicalUnitIntentEstimator,
)
from tron2_chip.intent_estimation.normalization import NormalizationStats  # noqa: E402
from tron2_chip.intent_estimation.spec import IntentEstimatorSpec  # noqa: E402
from tron2_chip.intent_estimation.trainer import (  # noqa: E402
    IntentTrainingConfig,
    train_intent_estimator,
)


def test_network_and_physical_wrapper_shapes():
    spec = IntentEstimatorSpec.airbot_planar()
    model = IntentEstimatorMLP(spec.input_size)
    prediction = model(torch.zeros(5, spec.input_size))
    assert prediction.shape == (5, 3)
    input_stats = NormalizationStats(np.zeros(spec.input_size), np.ones(spec.input_size))
    output_stats = NormalizationStats(np.array([1, 2, 3]), np.array([2, 3, 4]))
    wrapped = PhysicalUnitIntentEstimator(model, input_stats, output_stats)
    assert wrapped(torch.zeros(5, spec.input_size)).shape == (5, 3)


def test_wrench_shift_and_yaw_frame_label():
    force = torch.tensor([[1.0, 0.0, 0.0]])
    torque = torch.zeros(1, 3)
    from_point = torch.tensor([[0.0, 1.0, 0.0]])
    to_point = torch.zeros(1, 3)
    force, shifted = shift_wrench_reference(force, torque, from_point, to_point)
    torch.testing.assert_close(shifted, torch.tensor([[0.0, 0.0, -1.0]]))
    label = planar_intent_label(force, shifted, torch.tensor([torch.pi / 2]))
    torch.testing.assert_close(label, torch.tensor([[0.0, -1.0, -1.0]]), atol=1e-6, rtol=1e-6)


def test_batched_history_matches_declared_feature_layout():
    history = BatchedIntentHistory(2, 2, 2, "cpu")
    env_ids = torch.arange(2)
    history.reset(env_ids, torch.ones(2, 2), torch.zeros(2, 2), torch.full((2, 2), 3.0))
    history.append(torch.full((2, 2), 2.0), torch.full((2, 2), 4.0), torch.full((2, 2), 5.0))
    assert history.features().shape == (2, 12)
    torch.testing.assert_close(
        history.features()[0],
        torch.tensor([1, 1, 2, 2, 0, 0, 4, 4, 3, 3, 5, 5], dtype=torch.float32),
    )


def test_regression_loss_and_metrics_are_finite():
    target = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    prediction = target + 0.1
    assert regression_loss(prediction, target).item() > 0.0
    metrics = physical_metrics(prediction, target)
    assert torch.all(torch.isfinite(metrics["rmse"]))


def test_minimal_training_writes_complete_artifact_bundle(tmp_path: Path):
    spec = IntentEstimatorSpec.airbot_planar(history_steps=1)
    rng = np.random.default_rng(2)
    episode_ids = np.repeat(np.arange(10), 8)
    features = rng.normal(size=(80, spec.input_size)).astype(np.float32)
    weights = rng.normal(size=(spec.input_size, 3)).astype(np.float32)
    targets = features @ weights
    dataset = IntentDataset(features, targets, episode_ids)
    config = IntentTrainingConfig(
        epochs=2, batch_size=16, hidden_sizes=(16, 16), patience=2, seed=1
    )
    _, _, _, report = train_intent_estimator(dataset, spec, tmp_path, config)
    for name in (
        "intent_estimator.pt",
        "intent_spec.json",
        "input_normalization.npz",
        "output_normalization.npz",
        "training_report.json",
    ):
        assert (tmp_path / name).is_file()
    assert report["split_samples"]["train"] > 0

