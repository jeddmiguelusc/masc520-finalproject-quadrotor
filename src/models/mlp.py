"""M1 / M2 backbone: an MLP on the current sample only (no history)."""
from __future__ import annotations

import torch
from torch import nn


class MLPBackbone(nn.Module):
    """Normalised inputs at the current sample -> n_out residual outputs. Default ~27k params."""
    uses_history = False

    def __init__(self, n_in: int, n_out: int, hidden=(128, 128, 64)):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x (B, W, F): only the last step is used
        return self.net(x[:, -1, :])
