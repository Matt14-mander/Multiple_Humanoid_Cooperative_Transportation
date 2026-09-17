"""Export a trained GRU to ONNX and compare it with PyTorch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .model import ExternalWrenchGRU
from .data import COMPONENTS


def export_and_verify(artifact_dir: Path, samples: int = 8) -> dict:
    checkpoint = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=False)
    model = ExternalWrenchGRU(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    export_x = torch.randn(1, checkpoint["window_frames"], checkpoint["model_kwargs"]["input_size"])
    path = artifact_dir / "external_wrench_gru.onnx"
    torch.onnx.export(model, export_x, path, input_names=["history"], output_names=["wrench_normalized"],
                      dynamic_axes={"history": {0: "batch"}, "wrench_normalized": {0: "batch"}},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    x = torch.randn(samples, checkpoint["window_frames"], checkpoint["model_kwargs"]["input_size"])
    torch_output = model(x).detach().numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"history": x.numpy()})[0]
    error = np.abs(torch_output - onnx_output)
    result = {"onnx_path": str(path.resolve()), "samples": samples,
              "max_abs_error_normalized": float(error.max()), "mean_abs_error_normalized": float(error.mean()),
              "allclose_rtol_1e-4_atol_1e-5": bool(np.allclose(torch_output, onnx_output, rtol=1e-4, atol=1e-5))}
    (artifact_dir / "onnx_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    contract = {
        "schema_version": "1.0",
        "model": "external_wrench_gru.onnx",
        "input_name": "history",
        "input_shape": ["batch", int(checkpoint["window_frames"]), int(checkpoint["model_kwargs"]["input_size"])],
        "feature_fields": list(checkpoint["feature_fields"]),
        "sample_dt_s": float(checkpoint["sample_dt_s"]),
        "output_name": "wrench_normalized",
        "output_components": list(COMPONENTS),
        "output_frame": "base_Link",
        "output_reference_point": "base_Link_origin",
        "output_units": ["N", "N", "N", "Nm", "Nm", "Nm"],
        "normalization": {"input": "input_normalization.npz", "output": "output_normalization.npz"},
    }
    (artifact_dir / "deployment_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    if not result["allclose_rtol_1e-4_atol_1e-5"]: raise RuntimeError(f"ONNX mismatch: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args(); print(json.dumps(export_and_verify(args.artifact_dir), indent=2))


if __name__ == "__main__": main()
