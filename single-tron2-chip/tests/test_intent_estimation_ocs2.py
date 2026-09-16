from pathlib import Path

import numpy as np

from tron2_chip.intent_estimation.dataset import IntentDataset, split_from_labels
from tron2_chip.intent_estimation.prepare_ocs2 import convert_ocs2_rollouts


def _write_episode(path: Path, trajectory_id: str, split: str, accepted=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = 5
    q = np.arange(steps * 6, dtype=np.float32).reshape(steps, 6)
    wrench = np.zeros((steps, 2, 6), dtype=np.float32)
    wrench[:, 0, (0, 1, 5)] = np.array([1.0, 2.0, 3.0])
    np.savez_compressed(
        path,
        arm_position_actual=q,
        arm_velocity_actual=q + 100.0,
        arm_position_reference=q + 200.0,
        future_wrench=wrench,
        meta_trajectory_id=np.asarray(trajectory_id),
        meta_split=np.asarray(split),
        meta_accepted=np.asarray(accepted),
        meta_step_dt=np.asarray(0.02),
    )


def test_convert_ocs2_rollouts_preserves_splits_and_causal_history(tmp_path: Path):
    for index, split in enumerate(("train", "validation", "test")):
        _write_episode(
            tmp_path / "episodes" / split / ("trajectory_{}.npz".format(index)),
            "trajectory_{}".format(index),
            split,
        )
    _write_episode(
        tmp_path / "episodes" / "rejected" / "trajectory_bad.npz",
        "trajectory_bad",
        "train",
        accepted=False,
    )

    dataset, spec, report = convert_ocs2_rollouts(tmp_path, history_steps=2)
    assert dataset.features.shape == (15, 36)
    assert dataset.targets.shape == (15, 3)
    np.testing.assert_allclose(dataset.targets, np.tile([1.0, 2.0, 3.0], (15, 1)))
    np.testing.assert_allclose(dataset.features[0, :12], np.tile(np.arange(6), 2))
    assert spec.control_dt_s == 0.02
    assert report["included_episodes"] == 3
    train, validation, test = split_from_labels(dataset)
    assert {len(train), len(validation), len(test)} == {5}


def test_dataset_split_labels_round_trip(tmp_path: Path):
    dataset = IntentDataset(
        np.zeros((3, 6), dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
        np.array(["a", "b", "c"]),
        np.array(["train", "validation", "test"]),
    )
    path = tmp_path / "dataset.npz"
    dataset.save(path)
    loaded = IntentDataset.load(path)
    np.testing.assert_array_equal(loaded.split_labels, dataset.split_labels)
