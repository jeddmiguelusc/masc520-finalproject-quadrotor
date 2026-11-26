"""M3 backbone: causal temporal convolution network (TCN) over a 20-sample (50 ms) window.

Dilated causal 1-D convolutions (dilations 1, 2, 4, 8, kernel 3) give a receptive field of
31 samples >= the 20-sample window, so the output at the last step can see the whole window
and never sees the future. Separate force and moment heads read the last time step.
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fn
from torch import nn


class CausalConv(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

    def forward(self, x):                                   # (B, C, T)
        return self.conv(Fn.pad(x, (self.pad, 0)))


class TCNBackbone(nn.Module):
    uses_history = True

    def __init__(self, n_in: int, n_out: int, channels: int = 40, kernel: int = 3,
                 dilations=(1, 2, 4, 8), head_hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()   # no parameters either way
        self.inp = CausalConv(n_in, channels, kernel, 1)
        self.blocks = nn.ModuleList([CausalConv(channels, channels, kernel, d) for d in dilations[1:]])
        self.act = nn.SiLU()
        if n_out % 2:
            raise ValueError("n_out must split evenly into force and moment heads")
        half = n_out // 2
        def head():
            return nn.Sequential(nn.Linear(channels, head_hidden), nn.SiLU(), nn.Linear(head_hidden, half))
        self.force_head, self.moment_head = head(), head()
        self._half = half

    def sequence_features(self, x: torch.Tensor) -> torch.Tensor:   # (B, W, F) -> (B, C, W)
        h = self.drop(self.act(self.inp(x.transpose(1, 2))))
        for blk in self.blocks:
            h = h + self.drop(self.act(blk(h)))             # residual
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x (B, W, F)
        z = self.sequence_features(x)[:, :, -1]
        f, m = self.force_head(z), self.moment_head(z)
        if self._half == 3:
            return torch.cat([f, m], 1)
        # with NLL heads each head returns (mean3, logvar3): reorder to (mean6, logvar6)
        return torch.cat([f[:, :3], m[:, :3], f[:, 3:], m[:, 3:]], 1)
