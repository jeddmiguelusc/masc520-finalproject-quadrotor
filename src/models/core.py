"""ForceModel: physics prior (optional) + normalised neural residual, with an optional Gaussian
NLL head. All models see raw 18-feature windows and return force [N] and moment [N m].

    prediction = prior(x_now) + out_mean + out_std * net(normalise(x_window))

The network is trained on the standardised residual r = (y - prior - out_mean) / out_std, so
forces (~N) and moments (~0.01 N m) carry equal weight in the loss. Normalisers and the prior
are fitted on the training split only and stored as buffers (saved in checkpoints).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.models.physics import PhysicsPrior

LOGVAR_MIN, LOGVAR_MAX = -12.0, 6.0


class ForceModel(nn.Module):
    def __init__(self, backbone: nn.Module | None, n_in: int, n_out: int = 6, window: int = 1,
                 prior: bool = False, nll: bool = False):
        super().__init__()
        self.backbone, self.window, self.nll, self.n_out = backbone, window, nll, n_out
        self.prior = PhysicsPrior() if prior else None
        self.register_buffer("x_mean", torch.zeros(n_in))
        self.register_buffer("x_std", torch.ones(n_in))
        self.register_buffer("out_mean", torch.zeros(n_out))
        self.register_buffer("out_std", torch.ones(n_out))

    # ---------------------------------------------------------------- fitting on the train split
    @torch.no_grad()
    def fit_normalisers(self, X_rows: np.ndarray, X_last: np.ndarray, Y_last: np.ndarray) -> dict:
        """X_rows: every training row (input stats). X_last/Y_last: inputs/targets at scored samples."""
        info = {}
        if self.prior is not None:
            info["prior_r2"] = self.prior.fit(X_last, Y_last)
            info["prior_coefs"] = self.prior.coefficients()
            resid = Y_last - self.prior(torch.from_numpy(X_last)).numpy()
        else:
            resid = Y_last
        # statistics in float64: float32 sums over ~1.3M rows drift (vbat mean was off by 0.4)
        X_rows = np.asarray(X_rows, np.float64); resid = np.asarray(resid, np.float64)
        self.x_mean.copy_(torch.from_numpy(X_rows.mean(0).astype(np.float32)))
        self.x_std.copy_(torch.from_numpy(np.maximum(X_rows.std(0), 1e-6).astype(np.float32)))
        self.out_mean.copy_(torch.from_numpy(resid.mean(0).astype(np.float32)))
        self.out_std.copy_(torch.from_numpy(np.maximum(resid.std(0), 1e-9).astype(np.float32)))
        return info

    # ---------------------------------------------------------------- forward
    def base(self, x: torch.Tensor) -> torch.Tensor:
        b = self.out_mean.expand(x.shape[0], -1)
        if self.prior is not None:
            b = b + self.prior(x[:, -1, :])
        return b

    def raw_net(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Standardised residual mean (B,6) and log-variance (B,6) or None."""
        if self.backbone is None:
            return torch.zeros(x.shape[0], self.n_out, device=x.device), None
        z = self.backbone((x[:, -self.window:, :] - self.x_mean) / self.x_std)
        if self.nll:
            return z[:, :self.n_out], z[:, self.n_out:].clamp(LOGVAR_MIN, LOGVAR_MAX)
        return z, None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """x (B, W, 18) raw -> mean (B,6) physical units, variance (B,6) physical units or None."""
        mu, logvar = self.raw_net(x)
        mean = self.base(x) + self.out_std * mu
        var = None if logvar is None else torch.exp(logvar) * self.out_std ** 2
        return mean, var

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.raw_net(x)
        r = (y - self.base(x)) / self.out_std
        if logvar is None:
            return ((mu - r) ** 2).mean()
        return 0.5 * (logvar + (mu - r) ** 2 * torch.exp(-logvar)).mean()   # Gaussian NLL (no const)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
