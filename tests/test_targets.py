"""CP2 target tests: body-frame force/torque definitions and the acceleration semantics.

    pytest tests/test_targets.py -v        -> results/target_tests.json

Synthetic tests check the formulas; data tests check that the CSV columns mean what
src/data/targets.py assumes. Data tests use every 4th non-server-testset segment (as CP1 did)
and skip if the dataset is not downloaded.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data import frames as F
from src.data.load import REPO, RAW_DIR
from src.data.targets import (INERTIA, MASS, body_force, body_torque, build_targets, contact_mask,
                              trim_contact)

OUT = REPO / "results" / "target_tests.json"
HAVE_DATA = (RAW_DIR / "processed_data").exists()


def _record(key: str, value) -> None:
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d[key] = value
    d["_source"] = "tests/test_targets.py"
    OUT.write_text(json.dumps(d, indent=1, default=float))


def _frame(n=50, **cols) -> pd.DataFrame:
    base = {c: np.zeros(n) for c in ["t", "ang_acc_x", "ang_acc_y", "ang_acc_z", "ang_vel_x", "ang_vel_y",
                                      "ang_vel_z", "qx", "qy", "qz", "acc_x", "acc_y", "acc_z", "vel_x",
                                      "vel_y", "vel_z", "mot1", "mot2", "mot3", "mot4"]}
    base["qw"] = np.ones(n); base["t"] = np.arange(n) * 0.0025
    base.update({k: np.broadcast_to(np.asarray(v, float), (n,)).copy() for k, v in cols.items()})
    df = pd.DataFrame(base)
    df["segment_id"] = "s"
    return df


# ---------------------------------------------------------------- formulas (synthetic)
def test_constants():
    assert MASS == 0.772
    assert np.allclose(INERTIA, np.diag([0.0025, 0.0021, 0.0043]))


def test_force_is_mass_times_specific_force():
    df = _frame(acc_x=1.0, acc_y=-2.0, acc_z=9.81)
    assert np.allclose(body_force(df), MASS * np.array([1.0, -2.0, 9.81]))


def test_torque_euler_equation():
    rng = np.random.default_rng(0)
    w, dw = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
    df = _frame(**{f"ang_vel_{a}": w[:, i] for i, a in enumerate("xyz")},
                **{f"ang_acc_{a}": dw[:, i] for i, a in enumerate("xyz")})
    J = INERTIA
    expect = np.array([J @ dw[k] + np.cross(w[k], J @ w[k]) for k in range(50)])
    assert np.allclose(body_torque(df), expect)
    # pure spin about a principal axis, no ang acc -> no torque; about two axes -> gyroscopic term
    assert np.allclose(body_torque(_frame(ang_vel_z=5.0)), 0)
    M = body_torque(_frame(ang_vel_x=2.0, ang_vel_y=3.0))[0]
    assert np.allclose(M, [0, 0, 2 * 3 * (0.0021 - 0.0025)])


def test_targets_shape_dtype():
    Y = build_targets(_frame(acc_z=9.81))
    assert Y.shape == (50, 6) and Y.dtype == np.float32 and np.isfinite(Y).all()


def test_trim_contact_cuts_and_splits():
    n = 100
    mot = np.full(n, 1500.0); mot[:10] = 300; mot[50:55] = 300; mot[95:] = 300
    df = _frame(n, mot1=mot, mot2=mot, mot3=mot, mot4=mot)
    pieces = trim_contact(df)
    assert [(p.index[0], p.index[-1]) for p in pieces] == [(10, 49), (55, 94)]
    assert not any(contact_mask(p).any() for p in pieces)
    # moving fast with idle motors (free fall / flip) is NOT contact
    assert not contact_mask(_frame(mot1=300, mot2=300, mot3=300, mot4=300, vel_x=3.0)).any()


# ---------------------------------------------------------------- semantics (real data)
@pytest.fixture(scope="module")
def data():
    if not HAVE_DATA:
        pytest.skip("dataset not downloaded")
    from src.data.load import load_all, segment_index
    idx = segment_index()
    srv = set(idx.segment_id[idx.in_server_testset])
    df = load_all()
    keep = [s for s in df.segment_id.cat.categories if s not in srv][::4]
    d = df[df.segment_id.isin(keep)].reset_index(drop=True)
    d["segment_id"] = d["segment_id"].cat.remove_unused_categories()
    return d


def test_hover_force_and_torque(data):
    """Near hover (no contact): F ~ (0, 0, m g) in FLU body frame, M ~ 0."""
    m = F.hover_mask(data) & ~contact_mask(data)
    Y = build_targets(data[m])
    Fm, Mm = np.median(Y[:, :3], 0), np.median(Y[:, 3:], 0)
    _record("hover", {"n": int(m.sum()), "F_median_N": Fm.tolist(), "M_median_Nm": Mm.tolist(),
                      "m_g_N": MASS * F.G})
    assert abs(Fm[2] - MASS * F.G) / (MASS * F.G) < 0.01        # + z = up (thrust) in FLU
    assert np.abs(Fm[:2]).max() < 0.1
    assert np.abs(Mm).max() < 0.005


def test_ang_acc_is_body_rate_derivative(data):
    """ang_acc == d(ang_vel)/dt per segment, best at zero lag -> Euler's equation needs no extra term."""
    w, dw = F.vec(data, "ang_vel"), F.vec(data, "ang_acc")
    g = F.segment_gradient(data, w)
    m = F.interior_mask(data) & np.isfinite(g).all(1)
    res = {}
    for i, ax in enumerate("xyz"):
        a, b = dw[m, i], g[m, i]
        lags = {lag: float(np.corrcoef(a, np.roll(g[:, i], lag)[m])[0, 1]) for lag in (-2, -1, 0, 1, 2)}
        res[ax] = {"corr_lag0": lags[0], "best_lag": max(lags, key=lags.get),
                   "slope_col_on_grad": float(a @ b / (b @ b)),
                   "rms_col": float(np.sqrt(np.mean(a ** 2))), "rms_diff": float(np.sqrt(np.mean((a - b) ** 2)))}
    _record("ang_acc_vs_gradient", res)
    # yaw rate changes are small (rms ~1.6 rad/s^2), so finite-difference noise caps corr_z near 0.94
    for ax in "xyz":
        assert res[ax]["corr_lag0"] > 0.90 and res[ax]["best_lag"] == 0


def test_thrust_matches_rotor_model_in_flight(data):
    """Fz = m*acc_z tracks k*sum(w^2) (CP1 hover k) away from the ground."""
    k = json.loads((REPO / "results" / "hover_thrust_fit.json").read_text())["k_through_origin_N_per_rad2s2"]
    m = ~contact_mask(data)
    Fz = build_targets(data[m])[:, 2]
    w2 = (data.loc[m, ["mot1", "mot2", "mot3", "mot4"]].to_numpy() ** 2).sum(1)
    r = Fz - k * w2
    corr = float(np.corrcoef(Fz, w2)[0, 1])
    _record("thrust_vs_rotor", {"n": int(m.sum()), "corr": corr, "resid_median_N": float(np.median(r)),
                                "resid_rms_N": float(np.sqrt(np.mean(r ** 2)))})
    assert corr > 0.95 and abs(np.median(r)) < 0.2


def test_torque_sign_structure(data):
    """Torque fitted on the four w_i^2 gives three different balanced +-1 sign patterns (roll, pitch,
    yaw of an X quad), so M has the right axes and is not, e.g., permuted or in the world frame."""
    m = ~contact_mask(data) & F.interior_mask(data)
    Y = build_targets(data[m])
    A = np.c_[data.loc[m, ["mot1", "mot2", "mot3", "mot4"]].to_numpy() ** 2, np.ones(int(m.sum()))]
    pats, r2 = [], []
    for j in range(3):
        c, *_ = np.linalg.lstsq(A, Y[:, 3 + j], rcond=None)
        pats.append(np.sign(c[:4]).astype(int))
        r = Y[:, 3 + j] - A @ c
        r2.append(float(1 - r.var() / Y[:, 3 + j].var()))
    _record("torque_vs_rotor_speeds", {"sign_patterns_MxMyMz": [p.tolist() for p in pats], "static_fit_r2": r2})
    P = np.array(pats)
    assert (P.sum(1) == 0).all()                       # two +, two - each
    assert np.allclose(P @ P.T, 4 * np.eye(3))         # mutually orthogonal patterns
