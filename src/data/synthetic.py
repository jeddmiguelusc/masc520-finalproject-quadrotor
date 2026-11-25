"""Synthetic stand-in for the NeuroBEM data, for tests and timing only. Never used for results.

Inputs have realistic ranges; targets = a known omega^2 + linear-drag physics part, plus a
nonlinear part that depends on the last ~25 ms of history (so temporal models have something
to learn), plus noise.
"""
from __future__ import annotations

import numpy as np

from src.data.features import N_FEATURES

TRUE_COEFS = {          # used by tests to check the physics least-squares fit recovers them
    "kT": 1.5e-6, "drag_xy": -0.25, "drag_z": -0.10,
    "k_roll": 2.0e-7, "k_pitch": 1.8e-7, "k_yaw": 2.5e-8,
}


def _smooth_walk(rng, n, scale, smooth=200):
    w = np.cumsum(rng.normal(size=n + smooth))
    k = np.ones(smooth) / smooth
    s = np.convolve(w, k, mode="valid")[:n]
    s = (s - s.mean()) / (s.std() + 1e-9)
    return scale * s


def physics_targets(X: np.ndarray, c=TRUE_COEFS) -> np.ndarray:
    v = X[:, 0:3]; w2 = X[:, 9:13] ** 2
    s = w2.sum(1)
    F = np.stack([c["drag_xy"] * v[:, 0], c["drag_xy"] * v[:, 1], c["kT"] * s + c["drag_z"] * v[:, 2]], 1)
    roll = w2[:, 2] + w2[:, 3] - w2[:, 0] - w2[:, 1]
    pitch = w2[:, 0] + w2[:, 2] - w2[:, 1] - w2[:, 3]
    yaw = w2[:, 0] + w2[:, 3] - w2[:, 1] - w2[:, 2]
    M = np.stack([c["k_roll"] * roll, c["k_pitch"] * pitch, c["k_yaw"] * yaw], 1)
    return np.concatenate([F, M], 1)


def make_segment(rng: np.random.Generator, n: int, history_effect: float = 1.0,
                 noise: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((n, N_FEATURES), np.float32)
    for i in range(3):
        X[:, i] = _smooth_walk(rng, n, 4.0)                 # vel ~ +-10 m/s
        X[:, 3 + i] = _smooth_walk(rng, n, 1.0, 50)          # ang vel ~ +-3 rad/s
    tilt = _smooth_walk(rng, n, 0.3)
    X[:, 6] = 9.81 * np.sin(tilt); X[:, 8] = -9.81 * np.cos(tilt)
    for m in range(4):
        X[:, 9 + m] = 1150 + _smooth_walk(rng, n, 200.0, 40)
        X[:, 13 + m] = np.gradient(X[:, 9 + m]) * 400.0
    X[:, 17] = 165 - 15 * np.linspace(0, 1, n) + rng.normal(0, 0.2, n)
    Y = physics_targets(X)
    # history-dependent nonlinear part: lagged, saturating response to velocity and rates
    lag = 10
    vlag = np.vstack([np.repeat(X[:1, 0:3], lag, 0), X[:-lag, 0:3]])
    wlag = np.vstack([np.repeat(X[:1, 3:6], lag, 0), X[:-lag, 3:6]])
    Y[:, 0:3] += history_effect * 0.4 * np.tanh(vlag / 4.0) * np.abs(X[:, 0:3]) ** 0.5
    Y[:, 3:6] += history_effect * 0.01 * np.tanh(wlag)
    Y[:, 0:3] += rng.normal(0, noise, (n, 3))
    Y[:, 3:6] += rng.normal(0, noise * 0.02, (n, 3))
    return X.astype(np.float32), Y.astype(np.float32)


def make_segments(n_segments: int = 8, length: int = 2000, seed: int = 0, **kw):
    rng = np.random.default_rng(seed)
    return [make_segment(rng, int(length * rng.uniform(0.7, 1.3)), **kw) for _ in range(n_segments)]
