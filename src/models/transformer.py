"""M4 backbone (optional): small transformer encoder over the 20-sample window.

2 layers, 4 heads. Width d=40 (not 64) so it matches the ~25k-parameter budget of the other
models; at d=64 it would be ~70k params and an unfair comparison. Learned positional
embedding; causal attention mask; reads the last token.
"""
from __future__ import annotations

import torch
from torch import nn


class TransformerBackbone(nn.Module):
    uses_history = True

    def __init__(self, n_in: int, n_out: int, d_model: int = 40, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 80, window: int = 20):
        super().__init__()
        self.embed = nn.Linear(n_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_out)
        self.window = window
        self.register_buffer("mask", torch.triu(torch.full((window, window), float("-inf")), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x (B, W, F), W <= window
        W = x.shape[1]
        h = self.embed(x) + self.pos[:, -W:, :]
        h = self.encoder(h, mask=self.mask[:W, :W])
        return self.head(self.norm(h[:, -1, :]))
