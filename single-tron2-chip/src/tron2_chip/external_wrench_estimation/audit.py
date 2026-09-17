"""Run the strict 1020-episode audit without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .data import CLASS_NAMES, ExactBalancedSampler, fit_train_normalization, load_episodes

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "external_wrench_gru.yaml"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")); project = args.config.resolve().parents[1]
    episodes, audit = load_episodes([(project / p).resolve() for p in cfg["data_roots"]])
    input_norm, output_norm = fit_train_normalization(episodes)
    train_classes = np.concatenate([e.classes for e in episodes if e.split == "train"])
    sampler = ExactBalancedSampler(train_classes, int(cfg["seed"]))
    audit["splits"]["train"]["class_counts_after_balanced_sampling_per_epoch"] = {
        name: sampler.per_class for name in CLASS_NAMES
    }
    audit["normalization"] = {
        "fitted_on": "train only", "train_source_sha256": input_norm.source_sha256,
        "input_dimensions": len(input_norm.mean), "output_dimensions": len(output_norm.mean),
    }
    text = json.dumps(audit, indent=2)
    if args.output: args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__": main()
