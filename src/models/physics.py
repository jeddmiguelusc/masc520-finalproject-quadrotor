"""P0b and the prior inside the hybrids: thrust/torque proportional to omega^2 plus linear rotor drag.

Body frame FLU. Motors: 1 back-right, 2 front-right, 3 back-left, 4 front-left (Readme).
With w_i^2 the squared motor speeds and v the body velocity:

    F_x = a0 + a1 v_x + a2 v_x S_w          F_y = b0 + b1 v_y + b2 v_y S_w
    F_z = c0 + c1 sum(w_i^2) + c2 v_z
    M_x = d0 + d1 (w3^2 + w4^2 - w1^2 - w2^2)      left minus right   -> roll
    M_y = e0 + e1 (w1^2 + w3^2 - w2^2 - w4^2)      back minus front   -> pitch
    M_z = f0 + f1 (w1^2 + w4^2 - w2^2 - w3^2)      one diagonal minus the other -> yaw (sign fitted)

S_w = sum(w_i) (rotor drag grows with rotor speed). 15 coefficients, each output fitted by
ordinary least squares on the TRAINING split only, then frozen. Intercepts absorb constant
offsets such as a centre-of-mass offset. No arm length, prop data or spin directions needed.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.models.physics_np import TERM_NAMES, _design, fit_coefs  # shared with the numpy P0b baseline


class PhysicsPrior(nn.Module):
    n_coefs = sum(len(t) for t in TERM_NAMES)

    def __init__(self):
        super().__init__()
        for k, names in enumerate(TERM_NAMES):
            self.register_buffer(f"coef{k}", torch.zeros(len(names), dtype=torch.float32))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, X: np.ndarray, Y: np.ndarray) -> dict:
        """Least squares per output (float64, columns scaled for conditioning). Returns R^2 per output."""
        coefs, r2 = fit_coefs(X, Y)
        for k, c in enumerate(coefs):
            getattr(self, f"coef{k}").copy_(torch.from_numpy(c.astype(np.float32)))
        self.fitted.fill_(True)
        return r2

    def coefficients(self) -> dict:
        return {f"out{k}:{n}": float(c) for k, names in enumerate(TERM_NAMES)
                for n, c in zip(names, getattr(self, f"coef{k}").tolist())}

    def forward(self, x_last: torch.Tensor) -> torch.Tensor:
        """x_last (B, 18) raw inputs at the current sample -> (B, 6) force [N] and moment [N m]."""
        # float32 throughout (MPS has no float64); relative error ~1e-7, far below the noise.
        D = _design(x_last.float(), torch)
        return torch.stack([A @ getattr(self, f"coef{k}") for k, A in enumerate(D)], 1)
