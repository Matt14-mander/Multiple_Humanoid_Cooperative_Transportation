"""Train/evaluate the current 6D external-wrench estimator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .data import (CLASS_NAMES, COMPONENTS, ExactBalancedSampler, FEATURE_FIELDS, WindowDataset,
                   fit_train_normalization, load_episodes)
from .metrics import regression_metrics
from .model import ExternalWrenchGRU

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "external_wrench_gru.yaml"

def _device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


@torch.no_grad()
def evaluate(model, loader, output_norm, device, force_threshold, torque_threshold):
    model.eval()
    predictions, targets, classes = [], [], []
    for x, y, cls, _ in loader:
        pred = model(x.to(device)).cpu().numpy()
        predictions.append(output_norm.inverse(pred))
        targets.append(output_norm.inverse(y.numpy()))
        classes.append(cls.numpy())
    return regression_metrics(np.concatenate(targets), np.concatenate(predictions), np.concatenate(classes),
                              force_threshold, torque_threshold)


def run(config_path: Path, epochs_override: int | None = None,
        max_train_batches_override: int | None = None, output_override: Path | None = None) -> dict:
    project = config_path.resolve().parents[1]
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    roots = [(project / value).resolve() for value in cfg["data_roots"]]
    output = output_override or (project / cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    episodes, audit = load_episodes(roots)
    input_norm, output_norm = fit_train_normalization(episodes)
    train = WindowDataset(episodes, "train", input_norm, output_norm, int(cfg["window_frames"]))
    validation = WindowDataset(episodes, "validation", input_norm, output_norm, int(cfg["window_frames"]))
    test = WindowDataset(episodes, "test", input_norm, output_norm, int(cfg["window_frames"]))
    sampler = ExactBalancedSampler(train.classes, seed)
    tcfg = cfg["training"]
    batch_size = int(tcfg["batch_size"])
    workers = int(tcfg["num_workers"])
    train_loader = DataLoader(train, batch_size=batch_size, sampler=sampler, num_workers=workers,
                              pin_memory=torch.cuda.is_available())
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=workers)
    device = _device(tcfg["device"])
    kwargs = {"input_size": int(episodes[0].features.shape[1]), "hidden_size": int(cfg["model"]["hidden_size"]),
              "num_layers": int(cfg["model"]["num_layers"]), "dropout": float(cfg["model"]["dropout"]),
              "output_size": 6}
    model = ExternalWrenchGRU(**kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg["learning_rate"]),
                                  weight_decay=float(tcfg["weight_decay"]))
    epochs = epochs_override or int(tcfg["epochs"])
    max_batches = max_train_batches_override if max_train_batches_override is not None else tcfg["max_train_batches"]
    best_loss, best_state, stale, history = float("inf"), None, 0, []
    started = time.time()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train(); losses = []
        for batch_i, (x, y, _, _) in enumerate(train_loader):
            if max_batches is not None and batch_i >= int(max_batches): break
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x.to(device)), y.to(device))
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); val_sse = 0.0; val_n = 0
        with torch.no_grad():
            for x, y, _, _ in validation_loader:
                err = model(x.to(device)) - y.to(device)
                val_sse += float(torch.sum(err * err).cpu()); val_n += err.numel()
        val_loss = val_sse / val_n
        history.append({"epoch": epoch + 1, "train_mse_normalized": float(np.mean(losses)),
                        "validation_mse_normalized": val_loss})
        print(f"epoch={epoch+1} train={history[-1]['train_mse_normalized']:.6f} validation={val_loss:.6f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= int(tcfg["patience"]): break
    if best_state is None: raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state); model.to(device)
    thresholds = cfg["zero_thresholds"]
    validation_metrics = evaluate(model, validation_loader, output_norm, device,
                                  float(thresholds["force_n"]), float(thresholds["torque_nm"]))
    test_metrics = evaluate(model, test_loader, output_norm, device,
                            float(thresholds["force_n"]), float(thresholds["torque_nm"]))
    audit["splits"]["train"]["class_counts_after_balanced_sampling_per_epoch"] = {
        name: sampler.per_class for name in CLASS_NAMES}
    audit["normalization"] = {"fitted_on": "train only", "train_source_sha256": input_norm.source_sha256,
                              "input_dimensions": len(input_norm.mean), "output_dimensions": len(output_norm.mean)}
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    input_norm.save(output / "input_normalization.npz"); output_norm.save(output / "output_normalization.npz")
    checkpoint = {"model_state_dict": best_state, "model_kwargs": kwargs, "feature_fields": FEATURE_FIELDS,
                  "components": COMPONENTS, "window_frames": int(cfg["window_frames"]), "sample_dt_s": float(cfg["sample_dt_s"])}
    torch.save(checkpoint, output / "model.pt")
    mode = "smoke" if max_train_batches_override is not None else ("bounded_full_dataset" if epochs_override else "full")
    report = {"mode": mode,
              "device": str(device), "epochs_completed": len(history), "best_validation_loss_normalized": best_loss,
              "elapsed_seconds": time.time() - started, "history": history,
              "validation_metrics": validation_metrics,
              "test_unseen_source_metrics": test_metrics,
              "test_sources": audit["splits"]["test"]["source_trajectory_ids"]}
    (output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = run(args.config, args.epochs, args.max_train_batches, args.output_dir)
    focus = report["test_unseen_source_metrics"]["focus"]
    print("test focus: " + " ".join(f"{k}_rmse={v['rmse']:.4f}" for k, v in focus.items()))


if __name__ == "__main__": main()
