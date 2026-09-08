"""PyTorch networks for PAINT-style planar wrench regression."""

import torch
from torch import nn


def load_estimator_checkpoint(path, spec, map_location="cpu"):
    """Reconstruct an estimator and reject a mismatched signal contract."""
    checkpoint = torch.load(path, map_location=map_location)
    if checkpoint.get("spec_sha256") != spec.sha256:
        raise ValueError("checkpoint does not match IntentEstimatorSpec")
    model = IntentEstimatorMLP(
        checkpoint["input_size"],
        checkpoint["output_size"],
        tuple(checkpoint["hidden_sizes"]),
        checkpoint["activation"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


class IntentEstimatorMLP(nn.Module):
    """Paper-compatible three-hidden-layer MLP."""

    def __init__(self, input_size, output_size=3, hidden_sizes=(128, 128, 128), activation="elu"):
        super().__init__()
        if input_size < 1 or output_size != 3 or len(hidden_sizes) < 1:
            raise ValueError("invalid intent estimator dimensions")
        activations = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}
        if activation not in activations:
            raise ValueError("unsupported activation: " + activation)
        layers = []
        previous = int(input_size)
        for width in hidden_sizes:
            if width < 1:
                raise ValueError("hidden sizes must be positive")
            layers.extend((nn.Linear(previous, int(width)), activations[activation]()))
            previous = int(width)
        layers.append(nn.Linear(previous, int(output_size)))
        self.model = nn.Sequential(*layers)
        self.input_size = int(input_size)
        self.output_size = int(output_size)

    def forward(self, features):
        if features.shape[-1] != self.input_size:
            raise ValueError("feature tensor does not match estimator input size")
        return self.model(features)


class PhysicalUnitIntentEstimator(nn.Module):
    """Wrap a normalized model with preprocessing and physical-unit output."""

    def __init__(self, model, input_stats, output_stats):
        super().__init__()
        self.model = model
        self.input_clip = float(input_stats.clip)
        self.input_epsilon = float(input_stats.epsilon)
        self.output_epsilon = float(output_stats.epsilon)
        self.register_buffer("input_mean", torch.as_tensor(input_stats.mean, dtype=torch.float32))
        self.register_buffer("input_std", torch.as_tensor(input_stats.standard_deviation, dtype=torch.float32))
        self.register_buffer("output_mean", torch.as_tensor(output_stats.mean, dtype=torch.float32))
        self.register_buffer("output_std", torch.as_tensor(output_stats.standard_deviation, dtype=torch.float32))

    def forward(self, raw_features):
        normalized = (raw_features - self.input_mean) / torch.clamp(self.input_std, min=self.input_epsilon)
        normalized = torch.clamp(normalized, -self.input_clip, self.input_clip)
        prediction = self.model(normalized)
        return prediction * torch.clamp(self.output_std, min=self.output_epsilon) + self.output_mean
