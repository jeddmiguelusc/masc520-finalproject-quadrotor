"""Review fix (Nov 28): one failure policy for rollout errors (src/eval/rollout_stats.py). No dataset needed."""
from __future__ import annotations

import numpy as np

from src.eval.rollout_stats import DIVERGED_M, boot_idx, capped, diverged, paired_diff, summarise_errors


def _toy(seed=0, n_flights=6, per=50):
    rng = np.random.default_rng(seed)
    flights = np.repeat([f"f{k}" for k in range(n_flights)], per)
    err = rng.gamma(2.0, 0.005, size=len(flights))
    return err, flights


def test_capping_rules():
    x = np.array([0.01, np.nan, np.inf, -np.inf, 50.0, DIVERGED_M])
    assert diverged(x).tolist() == [False, True, True, True, True, False]
    assert capped(x).tolist() == [0.01, DIVERGED_M, DIVERGED_M, DIVERGED_M, DIVERGED_M, DIVERGED_M]


def test_diverged_rollouts_are_penalised_in_every_statistic():
    err, flights = _toy()
    clean = summarise_errors(err, flights)
    bad = err.copy(); bad[:5] = np.nan; bad[5:10] = np.inf; bad[10:15] = 1e9
    s = summarise_errors(bad, flights)
    assert s["n_diverged"] == 15 and clean["n_diverged"] == 0
    for k in ("rms", "median", "p75", "p90", "mean"):
        assert np.isfinite(s[k]) and s[k] >= clean[k]
    assert s["rms"] > 10 * clean["rms"]                       # a blow-up cannot hide in the headline
    assert all(np.isfinite(s["rms_ci95"])) and all(np.isfinite(s["median_ci95"]))


def test_paired_comparison_agrees_with_summary_and_stays_finite():
    err, flights = _toy()
    bad = err.copy(); bad[:30] = np.inf                       # model a blows up on 30 starts
    for stat in ("rms", "median"):
        d = paired_diff(bad, err, flights, stat=stat)
        assert np.isfinite(d["diff"]) and all(np.isfinite(d["ci95"]))
        assert d["diff"] > 0 and d["n_diverged_a"] == 30 and d["n_diverged_b"] == 0
    # both sides fully diverged: difference is exactly 0, never NaN (old code: inf - inf = NaN)
    allbad = np.full_like(err, np.inf)
    d = paired_diff(allbad, allbad.copy(), flights)
    assert d["diff"] == 0.0 and d["ci95"] == [0.0, 0.0]
    s = summarise_errors(bad, flights)
    assert s["rms"] > summarise_errors(err, flights)["rms"]   # headline and paired view rank the same way


def test_subset_bootstrap_uses_only_flights_in_the_subset():
    err, flights = _toy()
    m = np.isin(flights, ["f0", "f1"])                        # a subset present in 2 of 6 flights
    s = summarise_errors(err[m], flights[m])
    assert s["n_flights"] == 2 and s["rms_ci95"][0] > 0       # no empty replicate pulls the CI to 0


def test_matches_previous_horizon_bootstrap_on_finite_errors():
    """Same flight draws as the pre-fix eval_rollout_horizons.py, so validation numbers do not move."""
    err, flights = _toy(3)
    fl = np.unique(flights)
    groups = [np.where(flights == f)[0] for f in fl]
    boot = np.random.default_rng(0).integers(0, len(fl), size=(2000, len(fl)))
    reps = np.array([np.median(err[np.concatenate([groups[k] for k in row])]) for row in boot])
    old = np.percentile(reps, [2.5, 97.5])
    assert np.allclose(summarise_errors(err, flights)["median_ci95"], old)
    assert np.array_equal(boot_idx(len(fl)), boot)
