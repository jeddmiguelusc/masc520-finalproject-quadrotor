"""Torch-free part of the P0b physics prior: design matrices and least-squares fit.

src/models/physics.py (the torch module inside the hybrids) and src/eval/baselines.py (numpy P0b)
both use these, so the baseline and the hybrids' prior are the same model.
"""
from __future__ import annotations

import numpy as np

from src.data.features import IDX

_V = [IDX["vel_x"], IDX["vel_y"], IDX["vel_z"]]
_M = [IDX["mot1"], IDX["mot2"], IDX["mot3"], IDX["mot4"]]
TERM_NAMES = [
    ["1", "v_x", "v_x*sum_w"], ["1", "v_y", "v_y*sum_w"], ["1", "sum_w2", "v_z"],
    ["1", "roll_w2"], ["1", "pitch_w2"], ["1", "yaw_w2"],
]


def _design(x, lib):
    """Per-output design matrices from raw inputs x (N, 18). lib = np or torch."""
    v = x[:, _V]; w = x[:, _M]; w2 = w * w
    one = lib.ones_like(v[:, 0])
    sw, s2 = w.sum(1), w2.sum(1)
    st = lib.stack
    return [
        st([one, v[:, 0], v[:, 0] * sw], 1),
        st([one, v[:, 1], v[:, 1] * sw], 1),
        st([one, s2, v[:, 2]], 1),
        st([one, w2[:, 2] + w2[:, 3] - w2[:, 0] - w2[:, 1]], 1),
        st([one, w2[:, 0] + w2[:, 2] - w2[:, 1] - w2[:, 3]], 1),
        st([one, w2[:, 0] + w2[:, 3] - w2[:, 1] - w2[:, 2]], 1),
    ]


def fit_coefs(X: np.ndarray, Y: np.ndarray) -> tuple[list[np.ndarray], dict]:
    """Least squares per output (float64, columns scaled for conditioning) -> (coefs, R^2 per output)."""
    D = _design(np.asarray(X, np.float64), np)
    coefs, r2 = [], {}
    for k, A in enumerate(D):
        scale = np.abs(A).max(0); scale[scale == 0] = 1
        c, *_ = np.linalg.lstsq(A / scale, np.asarray(Y[:, k], np.float64), rcond=None)
        c = c / scale
        coefs.append(c)
        pred = A @ c
        r2[k] = float(1 - np.sum((Y[:, k] - pred) ** 2) / np.sum((Y[:, k] - Y[:, k].mean()) ** 2))
    return coefs, r2


def predict(coefs: list[np.ndarray], X: np.ndarray) -> np.ndarray:
    D = _design(np.asarray(X, np.float64), np)
    return np.stack([A @ c for A, c in zip(D, coefs)], 1)
