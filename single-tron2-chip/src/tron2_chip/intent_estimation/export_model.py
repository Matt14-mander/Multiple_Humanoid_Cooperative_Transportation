"""Export a trained intent-estimator artifact bundle to checked ONNX."""

import argparse
from pathlib import Path

import onnx

from .export import export_onnx
from .network import load_estimator_checkpoint
from .normalization import NormalizationStats
from .spec import IntentEstimatorSpec


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()
    artifact_dir = args.artifact_dir
    output = args.output or artifact_dir / "intent_estimator.onnx"
    spec = IntentEstimatorSpec.load(artifact_dir / "intent_spec.json")
    input_stats = NormalizationStats.load(artifact_dir / "input_normalization.npz")
    output_stats = NormalizationStats.load(artifact_dir / "output_normalization.npz")
    model = load_estimator_checkpoint(artifact_dir / "intent_estimator.pt", spec)
    export_onnx(model, spec, input_stats, output_stats, output, args.opset)
    onnx.checker.check_model(onnx.load(output))
    print("exported_and_checked={}".format(output.resolve()))


if __name__ == "__main__":
    main()
