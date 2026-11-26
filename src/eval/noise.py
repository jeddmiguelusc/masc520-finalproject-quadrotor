"""Target noise floor: how much of F and M is sample-to-sample noise no model can predict.

Two estimators per target component, computed inside contiguous pieces only:
  diff  sigma^2 = mean((y_t - y_{t-1})^2) / 2        exact for white noise on a smooth signal
  hp9   sigma^2 = mean(r^2) / (1 - 1/9),  r = y - centred 9-sample moving average (22.5 ms)

Both assume the noise is white at 400 Hz. The targets come from filtered signals (CP2: ang_acc
is ~0.78-0.98 x the raw derivative, i.e. low-passed), and filtered noise is correlated, which both
estimators under-count. So the result is an estimate of the white part of the noise, a LOWER
reference for achievable RMSE, not a hard bound. Reported in the paper's per-component RMS form.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from src.eval.metrics import _from_sums

K_HP = 9


def _piece_sums(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    d = np.diff(Y, axis=0)
    sse_diff = (d ** 2).sum(0) / 2
    if len(Y) >= K_HP:
        ker = np.ones(K_HP) / K_HP
        ma = np.stack([np.convolve(Y[:, j], ker, mode="valid") for j in range(Y.shape[1])], 1)
        h = K_HP // 2
        r = Y[h:len(Y) - h] - ma
        sse_hp = (r ** 2).sum(0) / (1 - 1 / K_HP)
        n_hp = len(r)
    else:
        sse_hp, n_hp = np.zeros(Y.shape[1]), 0
    return sse_diff, sse_hp, len(d), n_hp


def noise_floor(pieces: Sequence[tuple[str, np.ndarray, np.ndarray]]) -> dict:
    tot = {"diff": [np.zeros(6), 0], "hp9": [np.zeros(6), 0]}
    for _, _, Y in pieces:
        Y = np.asarray(Y, np.float64)
        if len(Y) < 2:
            continue
        sd, sh, nd, nh = _piece_sums(Y)
        tot["diff"][0] += sd; tot["diff"][1] += nd
        tot["hp9"][0] += sh; tot["hp9"][1] += nh
    out = {}
    for k, (sse, n) in tot.items():
        m = {kk: float(v) for kk, v in _from_sums(sse, n).items()}
        m["per_axis"] = np.sqrt(sse / n).tolist()
        m["n"] = int(n)
        out[k] = m
    return out


def noise_floor_by_speed(pieces, bins) -> dict:
    """Noise floor per speed bin: each piece is cut where |v| leaves the bin (vel = X[:, 0:3])."""
    from src.eval.metrics import speed_bin_labels
    out = {}
    for lab, lo, hi in zip(speed_bin_labels(bins), bins[:-1], bins[1:]):
        sub = []
        for pid, X, Y in pieces:
            sp = np.linalg.norm(X[:, 0:3], axis=1)
            keep = (sp >= lo) & (sp < hi)
            d = np.diff(np.r_[0, keep.astype(int), 0])
            sub += [(pid, X[a:b], Y[a:b]) for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]) if b - a >= K_HP]
        out[lab] = noise_floor(sub) if sub else {}
    return out
