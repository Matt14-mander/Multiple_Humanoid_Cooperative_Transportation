from pathlib import Path

import numpy as np
import pytest

from tron2_chip.intent_estimation import (
    CallableIntentBackend,
    IntentDataset,
    IntentEstimatorRuntime,
    IntentEstimatorSpec,
    IntentHistoryBuffer,
    NormalizationStats,
    split_by_episode,
)


def test_airbot_spec_is_versioned_and_has_paint_dimensions(tmp_path: Path):
    spec = IntentEstimatorSpec.airbot_planar()
    assert spec.input_size == 72
    assert spec.output_size == 3
    assert spec.history_duration_s == pytest.approx(0.08)
    path = tmp_path / "intent_spec.json"
    spec.save(path)
    assert IntentEstimatorSpec.load(path) == spec


def test_history_uses_causal_channel_major_layout():
    spec = IntentEstimatorSpec.airbot_planar(history_steps=2)
    history = IntentHistoryBuffer(spec)
    history.reset(np.ones(6), np.full(6, 10.0), np.full(6, 100.0))
    history.append(np.full(6, 2.0), np.full(6, 20.0), np.full(6, 200.0))
    expected = np.concatenate(
        (
            np.r_[np.ones(6), np.full(6, 2.0)],
            np.r_[np.full(6, 10.0), np.full(6, 20.0)],
            np.r_[np.full(6, 100.0), np.full(6, 200.0)],
        )
    )
    np.testing.assert_allclose(history.vector(), expected)


def test_history_reset_repeats_current_sample_instead_of_zero_padding():
    spec = IntentEstimatorSpec.airbot_planar(history_steps=3)
    history = IntentHistoryBuffer(spec)
    history.reset(np.arange(6), np.zeros(6), np.ones(6))
    q, dq, action = history.arrays()
    np.testing.assert_allclose(q, np.tile(np.arange(6), (3, 1)))
    np.testing.assert_allclose(dq, 0.0)
    np.testing.assert_allclose(action, 1.0)


def test_normalization_round_trip_and_constant_channel():
    values = np.array([[1.0, 4.0], [3.0, 4.0]], dtype=np.float32)
    stats = NormalizationStats.fit(values)
    normalized = stats.transform(values)
    assert np.all(np.isfinite(normalized))
    np.testing.assert_allclose(stats.inverse(normalized), values)


def test_dataset_split_keeps_episodes_disjoint(tmp_path: Path):
    episode_ids = np.repeat(np.arange(10), 4)
    dataset = IntentDataset(
        np.zeros((40, 12), dtype=np.float32),
        np.zeros((40, 3), dtype=np.float32),
        episode_ids,
    )
    path = tmp_path / "dataset.npz"
    dataset.save(path)
    loaded = IntentDataset.load(path, expected_input_size=12)
    train, validation, test = split_by_episode(loaded, 0.2, 0.2, seed=3)
    train_ids = set(train.episode_ids.tolist())
    validation_ids = set(validation.episode_ids.tolist())
    test_ids = set(test.episode_ids.tolist())
    assert train_ids.isdisjoint(validation_ids | test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert len(train) + len(validation) + len(test) == len(dataset)


def test_runtime_builds_history_denormalizes_clips_and_filters():
    spec = IntentEstimatorSpec.airbot_planar(history_steps=1)
    input_stats = NormalizationStats(np.zeros(spec.input_size), np.ones(spec.input_size))
    output_stats = NormalizationStats(np.array([10.0, 20.0, 1.0]), np.array([2.0, 4.0, 0.5]))
    backend = CallableIntentBackend(lambda _: np.array([2.0, -2.0, 4.0]))
    runtime = IntentEstimatorRuntime(
        spec,
        backend,
        input_normalizer=input_stats,
        output_normalizer=output_stats,
        output_limits=[12.0, 12.0, 2.0],
        smoothing=0.5,
    )
    runtime.reset(np.zeros(6))
    first = runtime.infer(np.zeros(6), np.zeros(6), np.zeros(6))
    np.testing.assert_allclose(first, [12.0, 12.0, 2.0])

    backend.function = lambda _: np.array([-2.0, -2.0, -4.0])
    second = runtime.infer(np.zeros(6), np.zeros(6), np.zeros(6))
    np.testing.assert_allclose(second, [9.0, 12.0, 0.5])
