from pathlib import Path

import numpy as np
import pytest

from tron2_chip.external_wrench_estimation.data import (
    ExactBalancedSampler, FEATURE_DIMS, FEATURE_FIELDS, FORBIDDEN_INPUT_FIELDS, WindowDataset,
    assert_no_leakage, fit_train_normalization, load_episodes,
)
from tron2_chip.external_wrench_estimation.onnx_runtime import OnnxWrenchEstimatorRuntime


PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT.parents[1] / "MHCT" / "data" / "raw"
ROOTS = [DATA_ROOT / "tron2_external_wrench_v3_multilevel_paired5",
         DATA_ROOT / "tron2_external_wrench_v3_multilevel_paired15_tail"]


@pytest.fixture(scope="module")
def loaded():
    if not all(path.is_dir() for path in ROOTS):
        pytest.skip("external 1020-episode dataset is not present in this checkout")
    return load_episodes(ROOTS)


def test_forbidden_fields_are_not_inputs():
    assert not set(FEATURE_FIELDS).intersection(FORBIDDEN_INPUT_FIELDS)
    for name in FORBIDDEN_INPUT_FIELDS:
        with pytest.raises(ValueError): assert_no_leakage(FEATURE_FIELDS + (name,))


def test_versioned_onnx_bundle_runs_without_policy_session():
    pytest.importorskip("onnxruntime")
    artifact = PROJECT / "models" / "external_wrench_gru_10ep"
    runtime = OnnxWrenchEstimatorRuntime(artifact)
    observation = {name: np.zeros(FEATURE_DIMS[name], dtype=np.float32) for name in FEATURE_FIELDS}
    prediction = runtime.update(observation)
    assert prediction.shape == (6,)
    assert np.isfinite(prediction).all()
    runtime.reset()


def test_exact_episode_and_source_disjoint_splits(loaded):
    episodes, audit = loaded
    assert len(episodes) == 1020
    sources = {s: {e.source_trajectory_id for e in episodes if e.split == s}
               for s in ("train", "validation", "test")}
    assert sources["train"].isdisjoint(sources["validation"])
    assert sources["train"].isdisjoint(sources["test"])
    assert sources["validation"].isdisjoint(sources["test"])
    assert audit["source_split_overlap"] == {"train_validation": 0, "train_test": 0, "validation_test": 0}


def test_falls_drop_terminal_done_frame(loaded):
    episodes, _ = loaded
    falls = [e for e in episodes if e.fall]
    assert falls
    assert all(e.first_done_index == e.retained_frames for e in falls)
    assert all(e.retained_frames < e.original_frames for e in falls)


def test_normalization_is_train_only_and_val_mutation_cannot_change_it(loaded):
    episodes, _ = loaded
    before = fit_train_normalization(episodes)
    validation = next(e for e in episodes if e.split == "validation")
    original = validation.features.copy()
    validation.features[:] = 1e9
    after = fit_train_normalization(episodes)
    validation.features[:] = original
    np.testing.assert_array_equal(before[0].mean, after[0].mean)
    np.testing.assert_array_equal(before[0].std, after[0].std)
    assert before[0].fitted_split == "train"


def test_window_padding_and_episode_reset(loaded):
    episodes, _ = loaded
    input_norm, output_norm = fit_train_normalization(episodes)
    dataset = WindowDataset(episodes, "validation", input_norm, output_norm, 30)
    x, y, _, _ = dataset[0]
    assert x.shape == (30, len(input_norm.mean)) and y.shape == (6,)
    np.testing.assert_allclose(x.numpy(), np.repeat(x.numpy()[0:1], 30, axis=0), atol=1e-6)


def test_train_sampler_is_exactly_balanced_without_touching_eval(loaded):
    episodes, _ = loaded
    input_norm, output_norm = fit_train_normalization(episodes)
    train = WindowDataset(episodes, "train", input_norm, output_norm)
    validation = WindowDataset(episodes, "validation", input_norm, output_norm)
    sampler = ExactBalancedSampler(train.classes, seed=42, epoch_size=3000)
    sampled_classes = train.classes[list(iter(sampler))]
    np.testing.assert_array_equal(np.bincount(sampled_classes, minlength=3), [1000, 1000, 1000])
    # Evaluation remains the original, naturally imbalanced source-disjoint sequence.
    assert len(validation) == sum(e.retained_frames for e in episodes if e.split == "validation")
    assert np.bincount(validation.classes, minlength=3)[0] > np.bincount(validation.classes, minlength=3)[1]
