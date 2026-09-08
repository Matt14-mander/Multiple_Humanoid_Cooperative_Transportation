"""Standalone supervised training for the PAINT intent estimator."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .dataset import split_by_episode
from .losses import physical_metrics, regression_loss
from .network import IntentEstimatorMLP
from .normalization import NormalizationStats


@dataclass(frozen=True)
class IntentTrainingConfig:
    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    hidden_sizes: tuple[int, ...] = (128, 128, 128)
    activation: str = "elu"
    loss: str = "mse"
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 0
    device: str = "auto"
    patience: int = 20

    def validate(self):
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0.0:
            raise ValueError("invalid epochs, batch size or learning rate")
        if self.weight_decay < 0.0 or self.patience < 1:
            raise ValueError("invalid weight decay or patience")
        if self.loss not in {"mse", "huber"}:
            raise ValueError("loss must be mse or huber")
        if not 0.0 < self.validation_fraction < 1.0 or not 0.0 < self.test_fraction < 1.0:
            raise ValueError("training requires non-empty validation and test fractions")
        if self.validation_fraction + self.test_fraction >= 1.0:
            raise ValueError("validation and test fractions leave no training split")


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _loader(dataset, input_stats, output_stats, batch_size, shuffle, seed):
    x = torch.from_numpy(input_stats.transform(dataset.features))
    y = torch.from_numpy(output_stats.transform(dataset.targets))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle,
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def _evaluate(model, loader, output_stats, device, loss_kind):
    model.eval()
    losses = []
    predictions = []
    targets = []
    output_mean = torch.as_tensor(output_stats.mean, device=device)
    output_std = torch.clamp(
        torch.as_tensor(output_stats.standard_deviation, device=device),
        min=output_stats.epsilon,
    )
    for features, target in loader:
        features, target = features.to(device), target.to(device)
        prediction = model(features)
        losses.append(regression_loss(prediction, target, kind=loss_kind).item())
        predictions.append(prediction * output_std + output_mean)
        targets.append(target * output_std + output_mean)
    if not predictions:
        raise ValueError("evaluation split is empty")
    metrics = physical_metrics(torch.cat(predictions), torch.cat(targets))
    return float(np.mean(losses)), {
        "rmse": metrics["rmse"].cpu().tolist(),
        "mae": metrics["mae"].cpu().tolist(),
        "r2": metrics["r2"].cpu().tolist(),
        "force_cosine_mean": float(metrics["force_cosine_mean"].cpu()),
    }


def train_intent_estimator(dataset, spec, output_dir: Path, config=None):
    """Train on episode-disjoint splits and write a reproducible artifact bundle."""
    config = IntentTrainingConfig() if config is None else config
    config.validate()
    if dataset.features.shape[1] != spec.input_size:
        raise ValueError("dataset features do not match IntentEstimatorSpec")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    train_set, validation_set, test_set = split_by_episode(
        dataset, config.validation_fraction, config.test_fraction, config.seed
    )
    input_stats = NormalizationStats.fit(train_set.features)
    output_stats = NormalizationStats.fit(train_set.targets)
    train_loader = _loader(train_set, input_stats, output_stats, config.batch_size, True, config.seed)
    validation_loader = _loader(validation_set, input_stats, output_stats, config.batch_size, False, config.seed)
    test_loader = _loader(test_set, input_stats, output_stats, config.batch_size, False, config.seed)

    device = _device(config.device)
    model = IntentEstimatorMLP(
        spec.input_size, spec.output_size, config.hidden_sizes, config.activation
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_state = None
    best_validation = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for features, target in train_loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = regression_loss(model(features), target, kind=config.loss)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        validation_loss, validation_metrics = _evaluate(
            model, validation_loader, output_stats, device, config.loss
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation_loss,
            "validation_rmse": validation_metrics["rmse"],
        }
        history.append(row)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    test_loss, test_metrics = _evaluate(model, test_loader, output_stats, device, config.loss)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec.save(output_dir / "intent_spec.json")
    input_stats.save(output_dir / "input_normalization.npz")
    output_stats.save(output_dir / "output_normalization.npz")
    checkpoint = {
        "model_state_dict": best_state,
        "spec_sha256": spec.sha256,
        "input_size": spec.input_size,
        "output_size": spec.output_size,
        "hidden_sizes": tuple(config.hidden_sizes),
        "activation": config.activation,
        "training_config": asdict(config),
    }
    torch.save(checkpoint, output_dir / "intent_estimator.pt")
    report = {
        "best_validation_loss": best_validation,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "split_samples": {
            "train": len(train_set), "validation": len(validation_set), "test": len(test_set)
        },
        "split_episodes": {
            "train": int(np.unique(train_set.episode_ids).size),
            "validation": int(np.unique(validation_set.episode_ids).size),
            "test": int(np.unique(test_set.episode_ids).size),
        },
        "epochs_completed": len(history),
        "device": str(device),
        "history": history,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return model.cpu(), input_stats, output_stats, report
