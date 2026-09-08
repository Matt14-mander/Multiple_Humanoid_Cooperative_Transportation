"""Export a raw-feature to physical-wrench estimator for deployment."""

from copy import deepcopy
from pathlib import Path

import torch

from .network import PhysicalUnitIntentEstimator


def export_onnx(model, spec, input_stats, output_stats, path: Path, opset_version=17):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = PhysicalUnitIntentEstimator(
        deepcopy(model).cpu(), input_stats, output_stats
    ).eval()
    example = torch.zeros(1, spec.input_size, dtype=torch.float32)
    torch.onnx.export(
        wrapped,
        example,
        str(path),
        input_names=["arm_history"],
        output_names=["intent_wrench"],
        dynamic_axes={"arm_history": {0: "batch"}, "intent_wrench": {0: "batch"}},
        opset_version=int(opset_version),
    )
    return path
