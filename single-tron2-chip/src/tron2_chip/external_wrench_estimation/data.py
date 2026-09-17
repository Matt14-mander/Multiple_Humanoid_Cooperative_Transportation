"""Strict loading, auditing, windowing and normalization for wrench episodes."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


TARGET_FIELD = "external_wrench_base_at_base_origin"
FORBIDDEN_INPUT_FIELDS = frozenset({
    "future_wrench",
    "ocs2_arm_on_base_wrench_plan",
    "external_wrench_payload_at_body_com",
    TARGET_FIELD,
})
# All fields are causal, runtime-observable proprioception sampled at episode time t,
# immediately before the simulator applies the wrench for [t, t + 0.02 s).
FEATURE_FIELDS = (
    "base_linear_velocity_body",
    "base_angular_velocity_body",
    "projected_gravity_body",
    "arm_position_actual",
    "arm_velocity_actual",
    "leg_position_actual",
    "leg_velocity_actual",
    "gait_phase",
    "previous_leg_action",
)
FEATURE_DIMS = {
    "base_linear_velocity_body": 3,
    "base_angular_velocity_body": 3,
    "projected_gravity_body": 3,
    "arm_position_actual": 6,
    "arm_velocity_actual": 6,
    "leg_position_actual": 10,
    "leg_velocity_actual": 10,
    "gait_phase": 2,
    "previous_leg_action": 10,
}
COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
SPLITS = ("train", "validation", "test")
CLASS_NAMES = ("zero", "positive", "negative")


def assert_no_leakage(feature_fields: Sequence[str]) -> None:
    overlap = FORBIDDEN_INPUT_FIELDS.intersection(feature_fields)
    if overlap:
        raise ValueError(f"label/plan leakage in feature fields: {sorted(overlap)}")


@dataclass
class Episode:
    trajectory_id: str
    source_trajectory_id: str
    split: str
    path: Path
    features: np.ndarray
    targets: np.ndarray
    classes: np.ndarray
    fall: bool
    original_frames: int
    retained_frames: int
    first_done_index: int | None


@dataclass(frozen=True)
class Normalization:
    mean: np.ndarray
    std: np.ndarray
    fitted_split: str
    source_sha256: str

    @classmethod
    def fit(cls, arrays: Iterable[np.ndarray], source_ids: Iterable[str]) -> "Normalization":
        values = np.concatenate([np.asarray(a, dtype=np.float64) for a in arrays], axis=0)
        mean = values.mean(axis=0).astype(np.float32)
        std = values.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        digest = hashlib.sha256("\n".join(sorted(set(source_ids))).encode()).hexdigest()
        return cls(mean, std, "train", digest)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values.astype(np.float32) - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values.astype(np.float32) * self.std + self.mean

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, std=self.std, fitted_split=self.fitted_split,
                 source_sha256=self.source_sha256)

    @classmethod
    def load(cls, path: Path) -> "Normalization":
        with np.load(path, allow_pickle=False) as z:
            return cls(z["mean"], z["std"], str(z["fitted_split"]), str(z["source_sha256"]))


def _read_manifest(root: Path) -> dict[str, dict[str, str]]:
    with (root / "rollout_manifest.csv").open(newline="", encoding="utf-8") as handle:
        return {row["trajectory_id"]: row for row in csv.DictReader(handle)}


def _verify_contract(root: Path) -> dict:
    contract = json.loads((root / "wrench_contract.json").read_text(encoding="utf-8"))
    expected = {"frame": "base_Link", "reference_point": "base_Link_origin",
                "semantics": "external_on_payload_proxy"}
    if contract.get("schema_version") != "2.0" or contract.get("base_label") != expected:
        raise ValueError(f"unexpected wrench contract in {root}")
    if tuple(contract.get("component_order", ())) != COMPONENTS:
        raise ValueError(f"unexpected wrench component order in {root}")
    return contract


def load_episodes(roots: Sequence[Path], feature_fields: Sequence[str] = FEATURE_FIELDS) -> tuple[list[Episode], dict]:
    assert_no_leakage(feature_fields)
    episodes: list[Episode] = []
    root_rows = []
    seen_trajectory_ids: set[str] = set()
    source_split: dict[str, str] = {}
    for root in map(Path, roots):
        contract = _verify_contract(root)
        manifest = _read_manifest(root)
        files = sorted((root / "episodes").rglob("*.npz"))
        if len(files) != len(manifest):
            raise ValueError(f"manifest/file count mismatch in {root}: {len(manifest)} != {len(files)}")
        root_rows.append({
            "root": str(root.resolve()), "episodes": len(files),
            "schema_version": contract["schema_version"],
            "manifest_accepted": sum(row["accepted"].lower() == "true" for row in manifest.values()),
            "manifest_falls": sum(row["fall"].lower() == "true" for row in manifest.values()),
            "manifest_split_episodes": {
                split: sum(row["split"] == split for row in manifest.values()) for split in SPLITS
            },
        })
        for path in files:
            with np.load(path, allow_pickle=False) as z:
                required = set(feature_fields) | {TARGET_FIELD, "done", "external_wrench_active",
                                                  "meta_trajectory_id", "meta_source_trajectory_id",
                                                  "meta_split", "meta_step_dt", "meta_wrench_component_order",
                                                  "meta_external_wrench_sign"}
                missing = sorted(required.difference(z.files))
                if missing:
                    raise ValueError(f"{path} missing fields {missing}")
                trajectory_id = str(z["meta_trajectory_id"])
                source_id = str(z["meta_source_trajectory_id"])
                split = str(z["meta_split"])
                if trajectory_id in seen_trajectory_ids:
                    raise ValueError(f"duplicate trajectory_id across roots: {trajectory_id}")
                seen_trajectory_ids.add(trajectory_id)
                if split not in SPLITS:
                    raise ValueError(f"unsupported split {split} in {path}")
                previous = source_split.setdefault(source_id, split)
                if previous != split:
                    raise ValueError(f"source {source_id} crosses splits: {previous}, {split}")
                if abs(float(z["meta_step_dt"]) - 0.02) > 1e-9:
                    raise ValueError(f"unexpected sample period in {path}")
                if tuple(z["meta_wrench_component_order"].tolist()) != COMPONENTS:
                    raise ValueError(f"component order mismatch in {path}")
                arrays = [np.asarray(z[name], dtype=np.float32) for name in feature_fields]
                n = len(arrays[0])
                if any(len(a) != n for a in arrays):
                    raise ValueError(f"feature length mismatch in {path}")
                for name, array in zip(feature_fields, arrays):
                    if array.ndim != 2 or array.shape[1] != FEATURE_DIMS[name]:
                        raise ValueError(f"unexpected shape for {name} in {path}: {array.shape}")
                features = np.concatenate(arrays, axis=1)
                targets = np.asarray(z[TARGET_FIELD], dtype=np.float32)
                if targets.shape != (n, 6) or not np.isfinite(features).all() or not np.isfinite(targets).all():
                    raise ValueError(f"invalid feature/target values in {path}")
                done = np.asarray(z["done"], dtype=bool)
                first_done = int(np.flatnonzero(done)[0]) if done.any() else None
                row = manifest.get(trajectory_id)
                if row is None:
                    raise ValueError(f"missing manifest row for {trajectory_id}")
                fall = row["fall"].lower() == "true"
                if row["source_trajectory_id"] != source_id or row["split"] != split or int(row["frames"]) != n:
                    raise ValueError(f"manifest/NPZ metadata mismatch for {trajectory_id}")
                # A fall's done frame is already terminal/invalid. Retain [0, first_done),
                # which is the largest interval unambiguously preceding the fall.
                retained = first_done if fall and first_done is not None else n
                if retained < 1:
                    raise ValueError(f"no pre-fall frames in {path}")
                active = np.asarray(z["external_wrench_active"], dtype=bool)[:retained]
                target_nonzero = np.any(targets[:retained] != 0.0, axis=1)
                if not np.array_equal(active, target_nonzero):
                    raise ValueError(f"active flag/target mismatch in {path}")
                sign = int(z["meta_external_wrench_sign"])
                classes = np.zeros(retained, dtype=np.int64)
                classes[active & (sign > 0)] = 1
                classes[active & (sign < 0)] = 2
                episodes.append(Episode(
                    trajectory_id, source_id, split, path,
                    features[:retained], targets[:retained], classes,
                    fall, n, retained, first_done,
                ))
    if len(episodes) != 1020:
        raise ValueError(f"expected exactly 1020 episodes, found {len(episodes)}")
    split_sources = {split: sorted({e.source_trajectory_id for e in episodes if e.split == split}) for split in SPLITS}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            overlap = set(split_sources[left]).intersection(split_sources[right])
            if overlap:
                raise ValueError(f"source leakage between {left}/{right}: {sorted(overlap)}")
    audit = build_audit(episodes, root_rows, feature_fields, split_sources)
    return episodes, audit


def build_audit(episodes: Sequence[Episode], root_rows: list[dict], feature_fields: Sequence[str],
                split_sources: Mapping[str, list[str]]) -> dict:
    offset = 0
    feature_layout = []
    for name in feature_fields:
        width = FEATURE_DIMS[name]
        feature_layout.append({"field": name, "slice": [offset, offset + width], "dimension": width})
        offset += width
    per_split = {}
    for split in SPLITS:
        selected = [e for e in episodes if e.split == split]
        counts = np.bincount(np.concatenate([e.classes for e in selected]), minlength=3)
        per_split[split] = {
            "episodes": len(selected), "sources": len(split_sources[split]),
            "source_trajectory_ids": split_sources[split],
            "frames_retained": int(sum(e.retained_frames for e in selected)),
            "class_counts_before_sampling": dict(zip(CLASS_NAMES, map(int, counts))),
            "episode_length_frames": {
                "min": min(e.retained_frames for e in selected),
                "max": max(e.retained_frames for e in selected),
                "mean": float(np.mean([e.retained_frames for e in selected])),
            },
            "target_component_min": dict(zip(COMPONENTS, map(float, np.min(np.concatenate([e.targets for e in selected]), axis=0)))),
            "target_component_max": dict(zip(COMPONENTS, map(float, np.max(np.concatenate([e.targets for e in selected]), axis=0)))),
        }
    fall_eps = [e for e in episodes if e.fall]
    return {
        "data_roots": root_rows,
        "episodes_total": len(episodes),
        "unique_sources": len({e.source_trajectory_id for e in episodes}),
        "label": {"field": TARGET_FIELD, "frame": "base_Link", "reference_point": "base_Link_origin",
                  "components": list(COMPONENTS), "units": ["N", "N", "N", "Nm", "Nm", "Nm"],
                  "prediction_horizon_s": 0.0},
        "features": {"fields": list(feature_fields), "layout": feature_layout,
                     "dimension": int(episodes[0].features.shape[1]),
                     "forbidden_fields": sorted(FORBIDDEN_INPUT_FIELDS), "leakage_check": "passed"},
        "window": {"frames": 30, "duration_s": 0.6, "sample_dt_s": 0.02,
                   "alignment": "history indices max(0,t-29)..t; target at t",
                   "episode_reset": "history never crosses episode boundary",
                   "padding": "left-repeat episode frame 0 until 30 frames"},
        "splits": per_split,
        "source_split_overlap": {"train_validation": 0, "train_test": 0, "validation_test": 0},
        "falls": {"episodes": len(fall_eps), "criterion": "manifest fall=true; drop first done frame and all following",
                  "original_frames": int(sum(e.original_frames for e in fall_eps)),
                  "retained_pre_fall_frames": int(sum(e.retained_frames for e in fall_eps)),
                  "dropped_terminal_frames": int(sum(e.original_frames - e.retained_frames for e in fall_eps))},
    }


class WindowDataset(Dataset):
    def __init__(self, episodes: Sequence[Episode], split: str, input_norm: Normalization,
                 output_norm: Normalization, window_frames: int = 30):
        self.episodes = [e for e in episodes if e.split == split]
        self.window_frames = window_frames
        self.input_norm = input_norm
        self.output_norm = output_norm
        self.index = [(ep_i, t) for ep_i, ep in enumerate(self.episodes) for t in range(ep.retained_frames)]
        self.classes = np.asarray([self.episodes[e].classes[t] for e, t in self.index], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        ep_i, t = self.index[item]
        ep = self.episodes[ep_i]
        start = max(0, t - self.window_frames + 1)
        window = ep.features[start:t + 1]
        if len(window) < self.window_frames:
            pad = np.repeat(ep.features[0:1], self.window_frames - len(window), axis=0)
            window = np.concatenate((pad, window), axis=0)
        x = self.input_norm.transform(window)
        y = self.output_norm.transform(ep.targets[t:t + 1])[0]
        return torch.from_numpy(x), torch.from_numpy(y), int(ep.classes[t]), ep.source_trajectory_id


class ExactBalancedSampler(Sampler[int]):
    """Exactly 1/3 zero, positive and negative draws; train split only."""
    def __init__(self, classes: np.ndarray, seed: int, epoch_size: int | None = None):
        self.pools = [np.flatnonzero(classes == i) for i in range(3)]
        if any(len(p) == 0 for p in self.pools):
            raise ValueError("all three balance classes must be present")
        size = len(classes) if epoch_size is None else epoch_size
        self.per_class = max(1, size // 3)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.per_class * 3

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        draws = np.concatenate([rng.choice(pool, self.per_class, replace=len(pool) < self.per_class)
                                for pool in self.pools])
        rng.shuffle(draws)
        return iter(draws.tolist())


def fit_train_normalization(episodes: Sequence[Episode]) -> tuple[Normalization, Normalization]:
    train = [e for e in episodes if e.split == "train"]
    sources = [e.source_trajectory_id for e in train]
    return (Normalization.fit((e.features for e in train), sources),
            Normalization.fit((e.targets for e in train), sources))
