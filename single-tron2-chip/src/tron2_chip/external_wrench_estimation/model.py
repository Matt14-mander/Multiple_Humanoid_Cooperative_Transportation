"""Configurable causal recurrent model."""

from __future__ import annotations

import torch
from torch import nn


class ExternalWrenchGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.1, output_size: int = 6):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, output_size))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.gru(history)
        return self.head(sequence[:, -1])

