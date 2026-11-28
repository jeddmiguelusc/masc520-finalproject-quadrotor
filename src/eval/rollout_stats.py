"""One failure policy and flight-bootstrap helpers for rollout errors (eval_rollout.py,
eval_rollout_horizons.py, evaluate_test.py).

Failure policy (review fix, Nov 28): a rollout error that is non-finite (NaN / Inf) or larger than
DIVERGED_M counts as diverged and is set to DIVERGED_M for EVERY statistic - RMS, median,
percentiles, bootstrap replicates and paired differences alike - and the number of diverged
rollouts is reported next to each summary. A model that blows up on hard starts can therefore never
look better than it is in one view and worse in another, and no NaN can reach an interval.

Bootstrap: percentile bootstrap over flights. Each subset (e.g. starts above 5 m/s) resamples only
the flights that have starts in that subset, so no replicate is empty.
"""
from __future__ import annotations

import numpy as np

DIVERGED_M = 10.0          # [m] position error treated as a diverged rollout (and its capped value)


def diverged(err: np.ndarray, cap: float = DIVERGED_M) -> np.ndarray:
    """Boolean mask: non-finite or above cap."""
    err = np.asarray(err, dtype=np.float64)
    return ~np.isfinite(err) | (err > cap)


def capped(err: np.ndarray, cap: float = DIVERGED_M) -> np.ndarray:
    """Errors with every diverged entry set to cap (NaN, +Inf and > cap all -> cap)."""
    err = np.asarray(err, dtype=np.float64)
    return np.where(diverged(err, cap), cap, err)


def flight_groups(flights: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Unique flights (sorted) and the row indices of each."""
    flights = np.asarray(flights)
    fl = np.unique(flights)
    return fl, [np.flatnonzero(flights == f) for f in fl]


def boot_idx(n_flights: int, b: int = 2000, seed: int = 0) -> np.ndarray:
    """(b, n_flights) flight indices drawn with replacement."""
    return np.random.default_rng(seed).integers(0, n_flights, size=(b, n_flights))


def boot_rows(groups: list[np.ndarray], idx: np.ndarray):
    """Yield the row indices of each bootstrap replicate."""
    for row in idx:
        yield np.concatenate([groups[k] for k in row])


def summarise_errors(err: np.ndarray, flights: np.ndarray, b: int = 2000, seed: int = 0,
                     cap: float = DIVERGED_M) -> dict:
    """RMS, median, p25/p75/p90, mean and flight-bootstrap 95% CIs of the RMS and the median, plus the
    diverged count, for one set of errors (one horizon, one subset). The bootstrap resamples only the
    flights present in `flights`."""
    raw = np.asarray(err, dtype=np.float64)
    div = diverged(raw, cap)
    e = capped(raw, cap)
    fl, groups = flight_groups(flights)
    idx = boot_idx(len(fl), b, seed)
    rms_r, med_r = [], []
    for rows in boot_rows(groups, idx):
        x = e[rows]
        rms_r.append(np.sqrt(np.mean(x ** 2))); med_r.append(np.median(x))
    return {"n": int(len(e)), "n_flights": int(len(fl)), "n_diverged": int(div.sum()),
            "rms": float(np.sqrt(np.mean(e ** 2))),
            "rms_ci95": np.percentile(rms_r, [2.5, 97.5]).tolist(),
            "median": float(np.median(e)), "median_ci95": np.percentile(med_r, [2.5, 97.5]).tolist(),
            "p25": float(np.percentile(e, 25)), "p75": float(np.percentile(e, 75)),
            "p90": float(np.percentile(e, 90)), "mean": float(np.mean(e))}


def paired_diff(a: np.ndarray, b_: np.ndarray, flights: np.ndarray, stat: str = "median",
                b: int = 2000, seed: int = 0, cap: float = DIVERGED_M) -> dict:
    """stat(a) - stat(b) on the same start points, with a paired flight bootstrap (the same flight
    draws for both sides). stat = "median" or "rms". Negative = a better. p = share of replicates
    with a not better. Diverged errors are capped on both sides first, so the result is always finite."""
    na, nb = int(diverged(a, cap).sum()), int(diverged(b_, cap).sum())
    a, b_ = capped(a, cap), capped(b_, cap)
    if a.shape != b_.shape:
        raise ValueError("paired comparison needs the same start points")
    f = (lambda x: np.median(x)) if stat == "median" else (lambda x: np.sqrt(np.mean(x ** 2)))
    fl, groups = flight_groups(flights)
    reps = np.array([f(a[r]) - f(b_[r]) for r in boot_rows(groups, boot_idx(len(fl), b, seed))])
    return {"stat": stat, "diff": float(f(a) - f(b_)), "ci95": np.percentile(reps, [2.5, 97.5]).tolist(),
            "p": float((reps >= 0).mean()), "win_rate_a": float((a < b_).mean()),
            "n_diverged_a": na, "n_diverged_b": nb}
