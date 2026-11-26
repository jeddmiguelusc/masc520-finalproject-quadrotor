"""Error metrics matching NeuroBEM Table II, flight-level bootstrap CIs, speed bins.

Paper definition (checked against Table II: BEM Fxy 0.803, Fz 1.265 -> F 0.982;
NeuroBEM 0.204, 0.504 -> 0.335; BEM Mxy 0.090, Mz 0.017 -> M 0.074):
    F   = sqrt( mean over samples AND the 3 axes of e^2 )     (RMS per component, not vector norm)
    Fxy = sqrt( mean over samples and x,y of e^2 )
    Fz  = sqrt( mean over samples of e_z^2 )
and the same for M. So F = sqrt((2 Fxy^2 + Fz^2) / 3). The 3-vector-norm RMSE that
src/train.py prints is sqrt(3) times larger (reported here as F_vecnorm for reference only).

Everything is computed from per-flight sums of squared errors, so a bootstrap over flights
(resample flights with replacement, pool their sums) is cheap and exact.

    stats = flight_sse(y_true, y_pred, flight_ids)        # FlightSSE
    table(stats)                                          # point estimates
    bootstrap_ci(stats)                                   # 95% CI per metric
    paired_diff_ci(stats_a, stats_b)                      # CI of metric(a) - metric(b), same resamples
    by_speed(y_true, y_pred, speed)                       # table per speed bin
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MASS, G = 0.772, 9.81
HOVER_N = MASS * G                                    # 7.573 N
SPEED_BINS = [0.0, 2.0, 5.0, 8.0, 11.0, 14.0, np.inf]
METRICS = ["F", "Fxy", "Fz", "M", "Mxy", "Mz", "F_pct_hover", "F_vecnorm", "M_vecnorm"]


def _from_sums(sse: np.ndarray, n: np.ndarray | float) -> dict:
    """sse: (..., 6) summed squared errors; n: (...) sample counts -> metrics (... shaped)."""
    sse = np.asarray(sse, np.float64)
    n = np.asarray(n, np.float64)
    mse = sse / n[..., None]
    out = {
        "F": np.sqrt(mse[..., 0:3].mean(-1)), "Fxy": np.sqrt(mse[..., 0:2].mean(-1)), "Fz": np.sqrt(mse[..., 2]),
        "M": np.sqrt(mse[..., 3:6].mean(-1)), "Mxy": np.sqrt(mse[..., 3:5].mean(-1)), "Mz": np.sqrt(mse[..., 5]),
        "F_vecnorm": np.sqrt(mse[..., 0:3].sum(-1)), "M_vecnorm": np.sqrt(mse[..., 3:6].sum(-1)),
    }
    out["F_pct_hover"] = 100 * out["F"] / HOVER_N
    return out


def rmse_table(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    e = np.asarray(y_pred, np.float64) - np.asarray(y_true, np.float64)
    return {k: float(v) for k, v in _from_sums((e ** 2).sum(0), len(e)).items()}


@dataclass
class FlightSSE:
    flights: np.ndarray          # (K,) flight ids
    sse: np.ndarray              # (K, 6) summed squared error per flight and component
    n: np.ndarray                # (K,) samples per flight


def flight_sse(y_true, y_pred, flight_ids) -> FlightSSE:
    e2 = (np.asarray(y_pred, np.float64) - np.asarray(y_true, np.float64)) ** 2
    fl, inv = np.unique(np.asarray(flight_ids), return_inverse=True)
    sse = np.zeros((len(fl), 6))
    np.add.at(sse, inv, e2)
    return FlightSSE(fl, sse, np.bincount(inv, minlength=len(fl)).astype(np.float64))


def table(s: FlightSSE) -> dict:
    return {k: float(v) for k, v in _from_sums(s.sse.sum(0), s.n.sum()).items()}


def _resample_idx(k: int, b: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, k, size=(b, k))


def _boot(s: FlightSSE, idx: np.ndarray) -> dict:
    return _from_sums(s.sse[idx].sum(1), s.n[idx].sum(1))


def bootstrap_ci(s: FlightSSE, b: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """{metric: [point, lo, hi]} - percentile bootstrap over flights."""
    reps = _boot(s, _resample_idx(len(s.flights), b, seed))
    point = table(s)
    q = [(1 - level) / 2 * 100, (1 + level) / 2 * 100]
    return {k: [point[k], *np.percentile(reps[k], q).tolist()] for k in METRICS}


def paired_diff_ci(a: FlightSSE, b_: FlightSSE, b: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """{metric: [diff, lo, hi, p_a_not_better]} for metric(a) - metric(b), the SAME flight resamples
    for both models. Negative diff = a is better. p_a_not_better = share of resamples with diff >= 0."""
    if not np.array_equal(a.flights, b_.flights) or not np.array_equal(a.n, b_.n):
        raise ValueError("paired comparison needs the same flights and samples")
    idx = _resample_idx(len(a.flights), b, seed)
    ra, rb = _boot(a, idx), _boot(b_, idx)
    pa, pb = table(a), table(b_)
    q = [(1 - level) / 2 * 100, (1 + level) / 2 * 100]
    out = {}
    for k in METRICS:
        d = ra[k] - rb[k]
        out[k] = [pa[k] - pb[k], *np.percentile(d, q).tolist(), float(np.mean(d >= 0))]
    return out


def speed_bin_labels(bins=SPEED_BINS) -> list[str]:
    return [f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}" for lo, hi in zip(bins[:-1], bins[1:])]


def by_speed(y_true, y_pred, speed, bins=SPEED_BINS) -> dict:
    """{bin label: {"n": samples, metrics...}} for samples whose |v| falls in each bin."""
    speed = np.asarray(speed)
    out = {}
    for lab, lo, hi in zip(speed_bin_labels(bins), bins[:-1], bins[1:]):
        m = (speed >= lo) & (speed < hi)
        out[lab] = {"n": int(m.sum()), **(rmse_table(y_true[m], y_pred[m]) if m.any() else {})}
    return out


def flight_of_piece(piece_id: str) -> str:
    """'2021-02-03-13-43-38_seg_4#0' -> '2021-02-03-13-43-38'."""
    return piece_id.split("#")[0].rsplit("_seg_", 1)[0]
