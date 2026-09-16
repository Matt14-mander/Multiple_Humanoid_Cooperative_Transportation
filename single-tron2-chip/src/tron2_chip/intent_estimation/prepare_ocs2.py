"""Convert TRON2 OCS2 rollout episodes into the intent-estimator contract."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .dataset import IntentDataset
from .spec import IntentEstimatorSpec


REQUIRED_KEYS = (
    "arm_position_actual",
    "arm_velocity_actual",
    "arm_position_reference",
    "future_wrench",
    "meta_trajectory_id",
    "meta_split",
    "meta_accepted",
    "meta_step_dt",
)


def _history_features(q, dq, previous_action, history_steps):
    """Build oldest-to-newest channel-major histories with reset padding."""
    sample_count = q.shape[0]
    offsets = np.arange(1 - history_steps, 1)
    indices = np.maximum(np.arange(sample_count)[:, None] + offsets[None, :], 0)
    return np.concatenate(
        (
            q[indices].reshape(sample_count, -1),
            dq[indices].reshape(sample_count, -1),
            previous_action[indices].reshape(sample_count, -1),
        ),
        axis=1,
    ).astype(np.float32)


def convert_ocs2_rollouts(
    rollout_root,
    history_steps=4,
    label_horizon=0,
    include_rejected=False,
    wrench_frame="world",
):
    root = Path(rollout_root)
    files = sorted((root / "episodes").glob("*/*.npz"))
    if not files:
        raise FileNotFoundError("no episode NPZ files found under " + str(root / "episodes"))

    feature_parts = []
    target_parts = []
    episode_parts = []
    split_parts = []
    skipped = []
    step_sizes = set()
    source_counts = Counter()

    for path in files:
        with np.load(path, allow_pickle=False) as episode:
            missing = [name for name in REQUIRED_KEYS if name not in episode.files]
            if missing:
                raise ValueError("{} misses keys: {}".format(path, ", ".join(missing)))
            accepted = bool(episode["meta_accepted"])
            trajectory_id = str(episode["meta_trajectory_id"])
            if not accepted and not include_rejected:
                skipped.append(trajectory_id)
                continue

            q = np.asarray(episode["arm_position_actual"], dtype=np.float32)
            dq = np.asarray(episode["arm_velocity_actual"], dtype=np.float32)
            action = np.asarray(episode["arm_position_reference"], dtype=np.float32)
            wrench = np.asarray(episode["future_wrench"], dtype=np.float32)
            if q.ndim != 2 or q.shape[1] != 6 or dq.shape != q.shape or action.shape != q.shape:
                raise ValueError(trajectory_id + " has invalid arm signal shapes")
            if wrench.ndim != 3 or wrench.shape[0] != q.shape[0] or wrench.shape[2] != 6:
                raise ValueError(trajectory_id + " has invalid future_wrench shape")
            if not 0 <= label_horizon < wrench.shape[1]:
                raise ValueError("label_horizon is outside future_wrench horizon")

            previous_action = np.concatenate((action[:1], action[:-1]), axis=0)
            features = _history_features(q, dq, previous_action, history_steps)
            targets = wrench[:, label_horizon, :][:, (0, 1, 5)].astype(np.float32)
            split = str(episode["meta_split"])
            if split not in {"train", "validation", "test"}:
                raise ValueError(trajectory_id + " has unsupported split " + split)

            feature_parts.append(features)
            target_parts.append(targets)
            episode_parts.append(np.full(q.shape[0], trajectory_id))
            split_parts.append(np.full(q.shape[0], split))
            step_sizes.add(float(episode["meta_step_dt"]))
            source_counts[split] += 1

    if not feature_parts:
        raise ValueError("no usable episodes remain after filtering")
    if len(step_sizes) != 1:
        raise ValueError("episodes use inconsistent control step sizes")
    control_dt_s = next(iter(step_sizes))
    spec = IntentEstimatorSpec(
        version="tron2-ocs2-future-wrench-planar-v1",
        joint_names=tuple("arm{}_Joint".format(i) for i in range(1, 7)),
        history_steps=int(history_steps),
        control_dt_s=control_dt_s,
        wrench_frame=wrench_frame,
        wrench_reference="ocs2_future_wrench_reference",
    )
    dataset = IntentDataset(
        np.concatenate(feature_parts),
        np.concatenate(target_parts),
        np.concatenate(episode_parts),
        np.concatenate(split_parts),
    )
    target_std = dataset.targets.std(axis=0)
    warnings = [
        "future_wrench is an OCS2 rollout signal, not a verified partner-applied payload wrench label",
        "source wrench reference point must be verified before physical interpretation",
    ]
    for name, standard_deviation in zip(spec.output_names, target_std):
        if standard_deviation < 0.1:
            warnings.append(
                "{} has low dataset variation (std={:.6g})".format(name, standard_deviation)
            )
    report = {
        "source_root": str(root.resolve()),
        "source_episode_files": len(files),
        "included_episodes": int(sum(source_counts.values())),
        "skipped_rejected_episodes": skipped,
        "episodes_by_split": dict(source_counts),
        "samples": len(dataset),
        "history_steps": int(history_steps),
        "control_dt_s": control_dt_s,
        "action_signal": "arm_position_reference shifted by one control step",
        "label_signal": "future_wrench[:, {}, [Fx, Fy, Mz]]".format(label_horizon),
        "target_mean": dataset.targets.mean(axis=0).tolist(),
        "target_std": target_std.tolist(),
        "target_min": dataset.targets.min(axis=0).tolist(),
        "target_max": dataset.targets.max(axis=0).tolist(),
        "spec": spec.to_dict(),
        "warnings": warnings,
    }
    return dataset, spec, report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("datasets/paint_intent_train.npz"))
    parser.add_argument("--spec-output", type=Path, default=Path("datasets/intent_spec.json"))
    parser.add_argument("--report-output", type=Path, default=Path("datasets/preparation_report.json"))
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--label-horizon", type=int, default=0)
    parser.add_argument("--wrench-frame", choices=("world", "base", "base_yaw", "payload"), default="world")
    parser.add_argument("--include-rejected", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset, spec, report = convert_ocs2_rollouts(
        args.rollout_root,
        history_steps=args.history_steps,
        label_horizon=args.label_horizon,
        include_rejected=args.include_rejected,
        wrench_frame=args.wrench_frame,
    )
    dataset.save(args.output)
    spec.save(args.spec_output)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("prepared: samples={} episodes={}".format(len(dataset), report["included_episodes"]))
    print("dataset={}".format(args.output.resolve()))
    for warning in report["warnings"]:
        print("warning: " + warning)


if __name__ == "__main__":
    main()
