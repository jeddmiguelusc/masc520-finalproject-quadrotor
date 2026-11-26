"""Non-network baselines (numpy only), fitted on the training scored samples.

    P0a  constant: the training mean of each target
    P0b  physics:  omega^2 + linear rotor drag, 15 coefficients (src/models/physics_np.py, the SAME
                   model as the prior inside M2/M3)
    B1   ridge regression on degree-2 polynomial features of the 18 standardised inputs at the
         current sample (18 linear + 171 products/squares = 189 features + intercept). Ridge alpha
         picked from ALPHAS on the split's own validation part (7 values: the tuning budget).
"""
from __future__ import annotations

import numpy as np

from src.models import physics_np

ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
# B2 needs far less shrinkage (its features are nearly collinear lags; a differentiator needs large
# opposite-sign weights): same 7-value budget, shifted grid. On the B1 grid it picked the edge, 1e-4.
ALPHAS_B2 = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
CHUNK = 100_000


class ConstantModel:
    name = "P0a"

    def fit(self, X, Y):
        self.mean = np.asarray(Y, np.float64).mean(0)
        return self

    def predict(self, X):
        return np.broadcast_to(self.mean, (len(X), 6)).copy()


class PhysicsModel:
    name = "P0b"

    def fit(self, X, Y):
        self.coefs, self.r2 = physics_np.fit_coefs(X, Y)
        return self

    def predict(self, X):
        return physics_np.predict(self.coefs, X)


def poly2(Z: np.ndarray) -> np.ndarray:
    """[Z, Z_i*Z_j for i<=j] for standardised Z (N, d)."""
    i, j = np.triu_indices(Z.shape[1])
    return np.concatenate([Z, Z[:, i] * Z[:, j]], 1)


class RidgePoly2:
    name = "B1"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def _feats(self, X):
        return poly2((np.asarray(X, np.float64) - self.mu) / self.sd)

    def _stats(self, X, Y):
        """Chunked X'X, X'y of centred polynomial features (memory ~ CHUNK x 189)."""
        p = None
        for a in range(0, len(X), CHUNK):
            P = self._feats(X[a:a + CHUNK]); y = np.asarray(Y[a:a + CHUNK], np.float64)
            if p is None:
                n, s, sy, PtP, Pty = 0, np.zeros(P.shape[1]), np.zeros(6), np.zeros((P.shape[1],) * 2), np.zeros((P.shape[1], 6))
                p = True
            n += len(P); s += P.sum(0); sy += y.sum(0); PtP += P.T @ P; Pty += P.T @ y
        return n, s, sy, PtP, Pty

    def fit_stats(self, X, Y):
        """Normalise with train stats, accumulate sufficient statistics once (re-used across alphas)."""
        X = np.asarray(X, np.float64)
        self.mu, self.sd = X.mean(0), np.maximum(X.std(0), 1e-6)
        n, s, sy, PtP, Pty = self._stats(X, Y)
        pm, ym = s / n, sy / n
        self._C = PtP / n - np.outer(pm, pm)                # feature covariance
        self._c = Pty / n - np.outer(pm, ym)                # feature-target covariance
        self._pm, self._ym = pm, ym
        self._fsd = np.sqrt(np.maximum(np.diag(self._C), 1e-12))   # penalise on standardised features
        return self

    def set_alpha(self, alpha: float):
        self.alpha = alpha
        D = self._fsd
        A = self._C / np.outer(D, D) + alpha * np.eye(len(D))
        w = np.linalg.solve(A, self._c / D[:, None]) / D[:, None]
        self.W, self.b = w, self._ym - self._pm @ w
        return self

    def fit(self, X, Y):
        return self.fit_stats(X, Y).set_alpha(self.alpha)

    def predict(self, X):
        out = np.empty((len(X), 6))
        for a in range(0, len(X), CHUNK):
            out[a:a + CHUNK] = self._feats(X[a:a + CHUNK]) @ self.W + self.b
        return out


# ---------------------------------------------------------------- B2: kinematic differencing (CP4)
def kinematic_features(rows: dict, idx: np.ndarray, lags: int = 20) -> np.ndarray:
    """Window features for B2 at scored samples rows['ws'].ends[idx]: body velocity and body rates at
    lags 0..lags-1 (2 x 3 x lags), gravity in body, w x v and w x (J w) at the current sample.
    No motor speeds, dmot or vbat: B2 can only differentiate the measured motion."""
    from src.data.features import IDX
    from src.data.targets import INERTIA
    ws = rows["ws"]
    e = ws.ends[idx]
    X = ws.X
    v0, w0 = IDX["vel_x"], IDX["ang_vel_x"]
    cols = [X[e - k, v0:v0 + 3] for k in range(lags)] + [X[e - k, w0:w0 + 3] for k in range(lags)]
    v, w = X[e, v0:v0 + 3].astype(np.float64), X[e, w0:w0 + 3].astype(np.float64)
    cols += [X[e, IDX["g_b_x"]:IDX["g_b_x"] + 3], np.cross(w, v), np.cross(w, w @ INERTIA.T)]
    return np.concatenate([np.asarray(c, np.float64) for c in cols], 1)


class KinematicFIR(RidgePoly2):
    """B2: ridge regression on the 20-sample history of body velocity and body rates only (129 features).

    Not a dynamics model: it shows how much of each target can be recovered by (learned, linear)
    numerical differentiation of the inputs, because F = m(dv_b/dt + w x v_b - g_b) and
    M = J dw/dt + w x Jw hold exactly. Any windowed model (M3, M4) has this path available.
    Same alpha budget as B1 (7 values, ALPHAS_B2, split's own validation part).
    """
    name = "B2"

    def _feats_rows(self, rows, idx):
        return (kinematic_features(rows, idx) - self.mu) / self.sd

    def fit_stats_rows(self, rows):
        n_all = len(rows["ws"].ends)
        first = kinematic_features(rows, np.arange(min(n_all, CHUNK)))
        # input standardisation from a pass over all rows (float64)
        s = np.zeros(first.shape[1]); s2 = np.zeros(first.shape[1])
        for a in range(0, n_all, CHUNK):
            F = kinematic_features(rows, np.arange(a, min(a + CHUNK, n_all)))
            s += F.sum(0); s2 += (F ** 2).sum(0)
        self.mu = s / n_all
        self.sd = np.sqrt(np.maximum(s2 / n_all - self.mu ** 2, 1e-12))
        d = first.shape[1]
        n, sp, sy, PtP, Pty = 0, np.zeros(d), np.zeros(6), np.zeros((d, d)), np.zeros((d, 6))
        Y = rows["Y"]
        for a in range(0, n_all, CHUNK):
            ii = np.arange(a, min(a + CHUNK, n_all))
            P = self._feats_rows(rows, ii); y = np.asarray(Y[ii], np.float64)
            n += len(P); sp += P.sum(0); sy += y.sum(0); PtP += P.T @ P; Pty += P.T @ y
        pm, ym = sp / n, sy / n
        self._C = PtP / n - np.outer(pm, pm)
        self._c = Pty / n - np.outer(pm, ym)
        self._pm, self._ym = pm, ym
        self._fsd = np.sqrt(np.maximum(np.diag(self._C), 1e-12))
        return self

    def predict_rows(self, rows):
        n_all = len(rows["ws"].ends)
        out = np.empty((n_all, 6))
        for a in range(0, n_all, CHUNK):
            ii = np.arange(a, min(a + CHUNK, n_all))
            out[ii] = self._feats_rows(rows, ii) @ self.W + self.b
        return out
