"""Stateful inference interface intended for later Isaac Lab integration."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .data import FEATURE_FIELDS, Normalization
from .model import ExternalWrenchGRU


class WrenchEstimatorRuntime:
    def __init__(self, artifact_dir: str | Path, device: str = "cpu"):
        artifact_dir = Path(artifact_dir)
        checkpoint = torch.load(artifact_dir / "model.pt", map_location=device, weights_only=False)
        self.feature_fields = tuple(checkpoint["feature_fields"])
        if self.feature_fields != FEATURE_FIELDS:
            raise ValueError("runtime feature contract differs from checkpoint")
        self.window_frames = int(checkpoint["window_frames"])
        self.input_norm = Normalization.load(artifact_dir / "input_normalization.npz")
        self.output_norm = Normalization.load(artifact_dir / "output_normalization.npz")
        self.model = ExternalWrenchGRU(**checkpoint["model_kwargs"]).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.device = torch.device(device)
        self.history: deque[np.ndarray] = deque(maxlen=self.window_frames)

    def reset(self) -> None:
        self.history.clear()

    @torch.no_grad()
    def update(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        frame = np.concatenate([np.asarray(observation[name], dtype=np.float32).reshape(-1)
                                for name in self.feature_fields])
        if not self.history:
            self.history.extend([frame.copy() for _ in range(self.window_frames - 1)])
        self.history.append(frame)
        window = self.input_norm.transform(np.stack(self.history))[None]
        normalized = self.model(torch.from_numpy(window).to(self.device)).cpu().numpy()
        return self.output_norm.inverse(normalized)[0]

