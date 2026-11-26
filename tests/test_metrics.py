"""CP3 tests: metric definition, bootstrap/paired CIs, baselines, noise estimator (synthetic only)."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval import metrics as M
from src.eval.baselines import ConstantModel, PhysicsModel, RidgePoly2, poly2
from src.eval.noise import noise_floor


def test_paper_table2_definition_reproduces_published_totals():
    # F = sqrt((2 Fxy^2 + Fz^2)/3): BEM, NeuroBEM rows of Table II
    for fxy, fz, f in [(0.803, 1.265, 0.982), (0.204, 0.504, 0.335)]:
        assert abs(np.sqrt((2 * fxy ** 2 + fz ** 2) / 3) - f) < 0.002
    for mxy, mz, m in [(0.090, 0.017, 0.074), (0.014, 0.004, 0.012)]:
        assert abs(np.sqrt((2 * mxy ** 2 + mz ** 2) / 3) - m) < 0.001


def test_rmse_table_components():
    y = np.zeros((4, 6)); p = np.zeros((4, 6))
    p[:, 0] = 1.0; p[:, 2] = 2.0; p[:, 5] = 0.3
    t = M.rmse_table(y, p)
    assert np.isclose(t["Fxy"], np.sqrt(0.5)) and np.isclose(t["Fz"], 2.0)
    assert np.isclose(t["F"], np.sqrt(5 / 3)) and np.isclose(t["F_vecnorm"], np.sqrt(5))
    assert np.isclose(t["M"], 0.3 / np.sqrt(3)) and np.isclose(t["Mz"], 0.3)
    assert np.isclose(t["F_pct_hover"], 100 * t["F"] / (0.772 * 9.81))


def test_flight_sse_pooling_equals_direct():
    rng = np.random.default_rng(0)
    y, p = rng.normal(size=(500, 6)), rng.normal(size=(500, 6))
    fl = rng.choice(["a", "b", "c", "d"], 500)
    assert all(np.isclose(M.table(M.flight_sse(y, p, fl))[k], v) for k, v in M.rmse_table(y, p).items())


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(2000, 6))
    fl = np.repeat(np.arange(20), 100)
    p = y + rng.normal(size=y.shape) * (1 + fl[:, None] / 10)       # flights differ in error
    s = M.flight_sse(y, p, fl)
    c1, c2 = M.bootstrap_ci(s, b=500, seed=3), M.bootstrap_ci(s, b=500, seed=3)
    assert c1 == c2
    for k in ("F", "M"):
        pt, lo, hi = c1[k]
        assert lo < pt < hi


def test_paired_diff():
    rng = np.random.default_rng(2)
    y = rng.normal(size=(2000, 6)); fl = np.repeat(np.arange(20), 100)
    good = M.flight_sse(y, y + 0.1 * rng.normal(size=y.shape), fl)
    bad = M.flight_sse(y, y + 0.5 * rng.normal(size=y.shape), fl)
    d = M.paired_diff_ci(good, bad, b=500)
    assert d["F"][0] < 0 and d["F"][2] < 0 and d["F"][3] == 0.0      # CI below zero, never not better
    same = M.paired_diff_ci(good, good, b=200)
    assert same["F"][:3] == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        M.paired_diff_ci(good, M.flight_sse(y[:1000], y[:1000], fl[:1000]))


def test_by_speed_bins_cover_all_samples():
    rng = np.random.default_rng(3)
    y, p, sp = rng.normal(size=(300, 6)), rng.normal(size=(300, 6)), rng.uniform(0, 18, 300)
    t = M.by_speed(y, p, sp)
    assert sum(v["n"] for v in t.values()) == 300 and list(t) == ["0-2", "2-5", "5-8", "8-11", "11-14", ">14"]


def test_flight_of_piece():
    assert M.flight_of_piece("2021-02-03-13-43-38_seg_4#1") == "2021-02-03-13-43-38"


def test_constant_and_ridge_baselines():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(5000, 18)) * 3 + 1
    Y = np.stack([X[:, 0] * X[:, 1], X[:, 2] ** 2, X[:, 3], 2 + 0 * X[:, 0], X[:, 4] - X[:, 5], X[:, 6] * 0.1], 1)
    assert np.allclose(ConstantModel().fit(X, Y).predict(X[:3]), Y.mean(0))
    r = RidgePoly2(1e-8).fit(X, Y)
    assert np.abs(r.predict(X) - Y).max() < 1e-5              # degree-2 targets recovered exactly
    assert poly2(np.ones((2, 18))).shape == (2, 18 + 171)
    r_strong = RidgePoly2().fit_stats(X, Y).set_alpha(1e4)
    assert np.abs(r_strong.predict(X) - Y).mean() > np.abs(r.predict(X) - Y).mean()


def test_numpy_p0b_matches_torch_prior():
    torch = pytest.importorskip("torch")
    from src.models.physics import PhysicsPrior
    rng = np.random.default_rng(5)
    X = rng.normal(size=(3000, 18)); X[:, 9:13] = rng.uniform(800, 2500, (3000, 4))
    Y = rng.normal(size=(3000, 6)) + X[:, 9:10] ** 2 * 1e-6
    p_np = PhysicsModel().fit(X, Y)
    p_t = PhysicsPrior(); p_t.fit(X, Y)
    out = p_t(torch.from_numpy(X).float()).numpy()
    assert np.allclose(out, p_np.predict(X), rtol=1e-4, atol=1e-4)


def test_noise_floor_recovers_white_noise():
    rng = np.random.default_rng(6)
    t = np.arange(4000) * 0.0025
    sd = np.array([0.05, 0.05, 0.1, 0.002, 0.002, 0.001])
    # smooth signal 10x the noise: the diff estimator also counts the signal's slope, so keep it slow
    sig = np.stack([np.sin(2 * np.pi * f * t) for f in (0.5, 1, 1.5, 0.7, 0.3, 2)], 1) * 10 * sd
    Y = sig + rng.normal(size=sig.shape) * sd
    nf = noise_floor([("p", np.zeros((4000, 18)), Y)])
    for est in ("diff", "hp9"):
        assert np.allclose(nf[est]["per_axis"], sd, rtol=0.05)
