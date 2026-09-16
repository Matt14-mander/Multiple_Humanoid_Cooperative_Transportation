"""Validated offline datasets for PAINT intent-estimator regression."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class IntentDataset:
    features: np.ndarray
    targets: np.ndarray
    episode_ids: np.ndarray
    split_labels: np.ndarray | None = None

    def __post_init__(self):
        x = np.asarray(self.features, dtype=np.float32)
        y = np.asarray(self.targets, dtype=np.float32)
        episodes = np.asarray(self.episode_ids)
        splits = None if self.split_labels is None else np.asarray(self.split_labels)
        if x.ndim != 2 or y.ndim != 2 or y.shape[1] != 3:
            raise ValueError("features and targets require shapes [N,D] and [N,3]")
        if x.shape[0] == 0 or x.shape[0] != y.shape[0] or episodes.shape != (x.shape[0],):
            raise ValueError("dataset sample counts do not match")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("dataset contains non-finite values")
        if splits is not None:
            if splits.shape != (x.shape[0],):
                raise ValueError("split_labels must have shape [N]")
            allowed = {"train", "validation", "test"}
            if not set(splits.astype(str).tolist()).issubset(allowed):
                raise ValueError("split_labels contain unsupported values")
        object.__setattr__(self, "features", x)
        object.__setattr__(self, "targets", y)
        object.__setattr__(self, "episode_ids", episodes)
        object.__setattr__(self, "split_labels", splits)

    def __len__(self):
        return self.features.shape[0]

    def subset(self, indices):
        return IntentDataset(
            self.features[indices],
            self.targets[indices],
            self.episode_ids[indices],
            None if self.split_labels is None else self.split_labels[indices],
        )

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "features": self.features,
            "targets": self.targets,
            "episode_ids": self.episode_ids,
        }
        if self.split_labels is not None:
            payload["split_labels"] = self.split_labels
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path, expected_input_size=None):
        with np.load(Path(path), allow_pickle=False) as payload:
            splits = payload["split_labels"] if "split_labels" in payload.files else None
            dataset = cls(
                payload["features"], payload["targets"], payload["episode_ids"], splits
            )
        if expected_input_size is not None and dataset.features.shape[1] != expected_input_size:
            raise ValueError("dataset input size does not match IntentEstimatorSpec")
        return dataset


def split_by_episode(dataset, validation_fraction=0.15, test_fraction=0.15, seed=0):
    """Split whole episodes to prevent adjacent history-window leakage."""
    if validation_fraction < 0.0 or test_fraction < 0.0 or validation_fraction + test_fraction >= 1.0:
        raise ValueError("invalid split fractions")
    episodes = np.unique(dataset.episode_ids)
    if episodes.size < 3 and validation_fraction > 0.0 and test_fraction > 0.0:
        raise ValueError("at least three episodes are required for a three-way split")
    shuffled = np.random.default_rng(seed).permutation(episodes)
    n_test = int(round(episodes.size * test_fraction))
    n_validation = int(round(episodes.size * validation_fraction))
    if test_fraction > 0.0:
        n_test = max(1, n_test)
    if validation_fraction > 0.0:
        n_validation = max(1, n_validation)
    if n_test + n_validation >= episodes.size:
        raise ValueError("split leaves no training episodes")
    test_ids = shuffled[:n_test]
    validation_ids = shuffled[n_test:n_test + n_validation]
    train_ids = shuffled[n_test + n_validation:]

    def select(ids):
        return dataset.subset(np.flatnonzero(np.isin(dataset.episode_ids, ids)))

    return select(train_ids), select(validation_ids), select(test_ids)


def split_from_labels(dataset):
    """Use a source dataset's episode-disjoint split assignments."""
    if dataset.split_labels is None:
        raise ValueError("dataset does not contain split_labels")
    episode_to_split = {}
    for episode, split in zip(dataset.episode_ids, dataset.split_labels):
        previous = episode_to_split.setdefault(str(episode), str(split))
        if previous != str(split):
            raise ValueError("one episode is assigned to multiple splits")

    result = []
    for name in ("train", "validation", "test"):
        indices = np.flatnonzero(dataset.split_labels.astype(str) == name)
        if indices.size == 0:
            raise ValueError("preassigned split is empty: " + name)
        result.append(dataset.subset(indices))
    return tuple(result)
