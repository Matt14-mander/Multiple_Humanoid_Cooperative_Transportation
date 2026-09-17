"""Torch-free ONNX Runtime adapter for sim2sim deployment."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from typing import Mapping

import numpy as np


class OnnxWrenchEstimatorRuntime:
    def __init__(self, artifact_dir: str | Path, providers=None):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("onnxruntime is required for wrench deployment") from error
        artifact_dir = Path(artifact_dir)
        contract = json.loads((artifact_dir / "deployment_contract.json").read_text(encoding="utf-8"))
        self.feature_fields = tuple(contract["feature_fields"])
        self.window_frames = int(contract["input_shape"][1])
        self.sample_dt_s = float(contract["sample_dt_s"])
        with np.load(artifact_dir / contract["normalization"]["input"], allow_pickle=False) as z:
            self.input_mean, self.input_std = z["mean"], z["std"]
        with np.load(artifact_dir / contract["normalization"]["output"], allow_pickle=False) as z:
            self.output_mean, self.output_std = z["mean"], z["std"]
        self.session = ort.InferenceSession(
            str(artifact_dir / contract["model"]), providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = contract["input_name"]
        self.output_name = contract["output_name"]
        self.history: deque[np.ndarray] = deque(maxlen=self.window_frames)

    def reset(self) -> None:
        self.history.clear()

    def update(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        frame = np.concatenate([
            np.asarray(observation[name], dtype=np.float32).reshape(-1) for name in self.feature_fields
        ])
        if frame.shape != self.input_mean.shape:
            raise ValueError(f"wrench feature shape {frame.shape} != {self.input_mean.shape}")
        if not self.history:
            self.history.extend([frame.copy() for _ in range(self.window_frames - 1)])
        self.history.append(frame)
        history = np.stack(self.history)
        normalized = ((history - self.input_mean) / self.input_std).astype(np.float32)[None]
        output = self.session.run([self.output_name], {self.input_name: normalized})[0][0]
        return output.astype(np.float32) * self.output_std + self.output_mean
