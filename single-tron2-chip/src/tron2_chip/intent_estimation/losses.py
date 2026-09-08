"""Regression losses and physical-unit evaluation metrics."""

import torch
from torch.nn import functional as F


def regression_loss(prediction, target, kind="mse", channel_weights=None):
    if prediction.shape != target.shape or prediction.shape[-1] != 3:
        raise ValueError("prediction and target must have matching [...,3] shapes")
    if kind == "mse":
        error = (prediction - target).square()
    elif kind == "huber":
        error = F.smooth_l1_loss(prediction, target, reduction="none")
    else:
        raise ValueError("unsupported regression loss: " + kind)
    if channel_weights is not None:
        weights = torch.as_tensor(channel_weights, dtype=error.dtype, device=error.device)
        if weights.shape != (3,):
            raise ValueError("channel_weights must have shape (3,)")
        error = error * weights
    return error.mean()


@torch.no_grad()
def physical_metrics(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("metrics require [N,3] prediction and target tensors")
    error = prediction - target
    rmse = torch.sqrt(error.square().mean(dim=0))
    mae = error.abs().mean(dim=0)
    target_centered = target - target.mean(dim=0)
    r2 = 1.0 - error.square().sum(dim=0) / torch.clamp(target_centered.square().sum(dim=0), min=1e-8)
    force_dot = (prediction[:, :2] * target[:, :2]).sum(dim=1)
    force_norm = torch.linalg.vector_norm(prediction[:, :2], dim=1) * torch.linalg.vector_norm(target[:, :2], dim=1)
    valid = force_norm > 1e-6
    cosine = torch.where(valid, force_dot / torch.clamp(force_norm, min=1e-6), torch.ones_like(force_dot))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "force_cosine_mean": cosine.mean(),
    }

