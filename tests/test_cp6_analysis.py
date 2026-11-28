"""CP6: statistics in scripts/eval_uncertainty.py and scripts/error_analysis.py on synthetic arrays
(numpy / scipy only: no data, no checkpoints, no torch)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load("eval_uncertainty")
E = _load("error_analysis")


def _synthetic(n=60_000, k=3, scale=1.0, seed=0):
    """Ensemble whose total variance is exactly the error variance (calibrated when scale=1)."""
    rng = np.random.default_rng(seed)
    speed = rng.uniform(0, 12, n)
    flights = np.array([f"f{i:02d}" for i in rng.integers(0, 12, n)])
    sd_true = 0.1 + 0.05 * speed[:, None] * np.ones((1, 6))
    Y = rng.normal(size=(n, 6))
    mean = Y + rng.normal(size=(n, 6)) * sd_true
    alea = 0.8 * sd_true ** 2 * scale
    epi = 0.2 * sd_true ** 2 * scale
    ens = {"mean": mean, "var": alea + epi, "aleatoric": alea, "epistemic": epi, "member_preds": []}
    return Y, ens, speed, flights


def test_gauss_nll_value_and_mixture_of_identical_members():
    e, v = np.array([0.0, 1.0]), np.array([1.0, 4.0])
    np.testing.assert_allclose(U.gauss_nll(e, v), 0.5 * (np.log(2 * np.pi) + np.log(v) + e ** 2 / v))
    y = np.array([0.3, -1.0]); mu = np.array([0.0, 0.5])
    np.testing.assert_allclose(U.mixture_nll(y, [mu, mu, mu], [v, v, v]), U.gauss_nll(mu - y, v), rtol=1e-12)


def test_calibrated_ensemble_has_nominal_coverage():
    Y, ens, speed, flights = _synthetic()
    g = U.group_stats(Y, ens, speed, flights, slice(0, 3), U.FlightBoot(flights, b=200))
    s = g["subsets"]["all"]
    assert abs(s["coverage"]["2sigma"][0] - U.NOMINAL["2sigma"]) < 0.005
    assert abs(s["coverage"]["90"][0] - 0.90) < 0.005
    assert s["coverage"]["2sigma"][1] <= s["coverage"]["2sigma"][0] <= s["coverage"]["2sigma"][2]
    assert g["calibration_curve"]["all"]["miscalibration_area"] < 0.005
    assert abs(s["rmse_over_sigma_rms"] - 1) < 0.02
    assert abs(s["epistemic_share_of_var"][0] - 0.2) < 1e-9
    assert g["sigma_ratio_gt5_over_le5"][0] > 1 and g["spearman_speed_sigma"] > 0.99
    bins = list(g["by_speed"].values())
    assert all(b["mean_sigma"] < c["mean_sigma"] for b, c in zip(bins, bins[1:]))   # sigma rises with speed


def test_overconfident_ensemble_undercovers():
    Y, ens, speed, flights = _synthetic(scale=0.25)            # sigma half the true error sd
    g = U.group_stats(Y, ens, speed, flights, slice(3, 6), U.FlightBoot(flights, b=200))
    s = g["subsets"]["all"]
    assert s["coverage"]["2sigma"][0] < 0.75                    # ~ P(|z| <= 1) = 0.683
    assert g["calibration_curve"]["all"]["max_shortfall"] > 0.2
    assert abs(s["rmse_over_sigma_rms"] - 2) < 0.05


def test_flight_boot_point_equals_direct_ratio():
    rng = np.random.default_rng(1)
    fl = np.array(list("aabbbcdd"))
    w = rng.uniform(size=8)
    b = U.FlightBoot(fl, b=50)
    m = np.ones(8, bool)
    pt = b.ci(b.per_flight(w, m), b.per_flight(np.ones(8), m))[0]
    assert pt == pytest.approx(w.mean())


def test_error_analysis_cells_match_manual_rmse():
    rng = np.random.default_rng(2)
    n = 5000
    Y = rng.normal(size=(n, 6))
    flights = np.array([f"fl{i}" for i in rng.integers(0, 4, n)])
    speed = rng.uniform(0, 10, n)
    traj_of = {"fl0": "circle", "fl1": "circle", "fl2": "lemniscate", "fl3": "lemniscate"}
    P1, P2 = Y + 0.1 * rng.normal(size=(n, 6)), Y + 0.3 * rng.normal(size=(n, 6))
    res = E.analyse(Y, {"A": [P1, P2], "B": [P1]}, flights, speed, traj_of)
    m = (np.array([traj_of[f] for f in flights]) == "circle") & (speed > 5)
    want = [np.sqrt(((P[m] - Y[m]) ** 2)[:, 0:3].mean()) for P in (P1, P2)]
    cell = res["tables"]["type_x_coarse"]["A"][("circle", ">5")]
    assert cell["F"] == pytest.approx(np.mean(want))
    assert cell["F_sd"] == pytest.approx(np.std(want, ddof=1))
    assert res["tables"]["type_x_coarse"]["B"][("circle", ">5")]["F_sd"] is None
    assert res["sizes"]["type_x_coarse"][("circle", ">5")]["n"] == int(m.sum())
    assert res["sizes"]["type"][("circle",)]["n_flights"] == 2
    assert sum(res["share_F_sse"]["A"].values()) == pytest.approx(1.0)
    # leave-one-flight-out equals the pooled RMSE on the remaining flights
    keep = flights != "fl0"
    lo = E.lofo(Y, [P1], flights, "fl0")
    assert lo["F"] == pytest.approx(np.sqrt(((P1[keep] - Y[keep]) ** 2)[:, 0:3].mean()))


def test_bin_labels_follow_metrics_bins():
    from src.eval import metrics as Mx
    sp = np.array([0.0, 1.99, 2.0, 4.9, 5.0, 13.9, 14.0, 30.0])
    lab = E.bin_label(sp, Mx.SPEED_BINS)
    assert lab.tolist() == ["0-2", "0-2", "2-5", "2-5", "5-8", "11-14", ">14", ">14"]
    assert E.bin_label(np.array([4.99, 5.0, 5.01]), E.COARSE_BINS).tolist() == ["<=5", "<=5", ">5"]


def test_combine_np_matches_ensemble_formula():
    from src.eval.predcache import combine_np
    rng = np.random.default_rng(3)
    mus = [rng.normal(size=(10, 6)) for _ in range(3)]
    vs = [rng.uniform(0.1, 1, size=(10, 6)) for _ in range(3)]
    c = combine_np(mus, vs)
    mix_var = np.mean([v + m ** 2 for m, v in zip(mus, vs)], 0) - np.mean(mus, 0) ** 2
    np.testing.assert_allclose(c["var"], mix_var, rtol=1e-10)
    assert np.all(combine_np(mus, [None] * 3)["aleatoric"] == 0)
