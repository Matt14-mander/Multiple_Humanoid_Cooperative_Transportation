"""Command-line entry point for standalone intent-estimator training."""

import argparse
from pathlib import Path

from .dataset import IntentDataset
from .spec import IntentEstimatorSpec
from .trainer import IntentTrainingConfig, train_intent_estimator


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="NPZ with features, targets and episode_ids")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/intent_estimator"))
    parser.add_argument("--spec", type=Path, help="existing intent_spec.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    spec = IntentEstimatorSpec.load(args.spec) if args.spec else IntentEstimatorSpec.airbot_planar()
    dataset = IntentDataset.load(args.dataset, expected_input_size=spec.input_size)
    config = IntentTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        loss=args.loss,
        seed=args.seed,
        device=args.device,
    )
    _, _, _, report = train_intent_estimator(dataset, spec, args.output_dir, config)
    metrics = report["test_metrics"]
    print("completed: samples={} epochs={} device={}".format(len(dataset), report["epochs_completed"], report["device"]))
    print("test_rmse: Fx={:.4f}N Fy={:.4f}N Mz={:.4f}Nm".format(*metrics["rmse"]))
    print("artifacts={}".format(args.output_dir.resolve()))


if __name__ == "__main__":
    main()

