"""CP4: src/sim/rollout.py - integrators, frames, differentiability, and a real-data P0b rollout.

Synthetic tests need only torch. The real-data tests (one A/val flight, 0.1 s = 40 steps) need the
dataset (or its cache) and are skipped without it.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.frames import G, MASS  # noqa: E402
from src.data.load import CACHE_DIR, RAW_DIR, REPO  # noqa: E402
from src.sim import rollout as R  # noqa: E402

N_STEPS = 40                                      # 0.1 s at 400 Hz


# ---------------------------------------------------------------- synthetic episodes
def make_episode(B=2, T=R.WINDOW + N_STEPS, v0=(0.0, 0.0, 0.0), w0=(0.0, 0.0, 0.0), dtype=torch.float64,
                 dt=R.DT):
    """Level attitude, constant recorded state (only the start sample matters for a full rollout)."""
    feats = torch.zeros(B, T, 18, dtype=dtype)
    feats[:, :, R.V] = torch.tensor(v0, dtype=dtype)
    feats[:, :, R.W] = torch.tensor(w0, dtype=dtype)
    feats[:, :, R.GB] = torch.tensor([0.0, 0.0, -G], dtype=dtype)
    feats[:, :, 9:13] = 1500.0
    q = torch.zeros(B, T, 4, dtype=dtype); q[..., 3] = 1.0
    return R.Episodes(feats, q, torch.zeros(B, T, 3, dtype=dtype), None, R.WINDOW - 1, dt)


class ConstBodyForce:
    """Force fn: constant body force/torque (B, 6)."""

    def __init__(self, fm):
        self.fm = torch.as_tensor(fm, dtype=torch.float64)

    def __call__(self, win, k, tau):
        return self.fm.expand(win.shape[0], 6).to(win.dtype)


@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_hover_stays_put(integrator):
    ep = make_episode()
    out = R.rollout(ConstBodyForce([0, 0, MASS * G, 0, 0, 0]), ep, N_STEPS, integrator, "full")
    assert out["pos_err"].max() < 1e-12 and out["vel_err"].max() < 1e-12 and out["att_err"].max() < 1e-6


def test_constant_acceleration_rk4_exact_euler_first_order():
    """Level, no rotation, F = m (1, 0, g): x = t^2 / 2. RK4 is exact for this ODE;
    semi-implicit Euler overshoots by dt * t / 2."""
    ep = make_episode()
    f = ConstBodyForce([MASS * 1.0, 0, MASS * G, 0, 0, 0])
    t = N_STEPS * R.DT
    rk = R.rollout(f, ep, N_STEPS, "rk4", "full")
    eu = R.rollout(f, ep, N_STEPS, "euler", "full")
    assert rk["pos"][0, -1, 0].item() == pytest.approx(0.5 * t * t, abs=1e-12)
    assert eu["pos"][0, -1, 0].item() == pytest.approx(0.5 * t * t + 0.5 * R.DT * t, abs=1e-12)
    assert rk["vel"][0, -1, 0].item() == pytest.approx(t, abs=1e-12)


def test_torque_free_spin_matches_analytic_attitude():
    """Spin about the principal z axis at 3 rad/s, zero torque, hover thrust along z: w constant,
    q(t) = exp(w t / 2). Both integrators keep w; attitude error tiny."""
    ep = make_episode(w0=(0.0, 0.0, 3.0))
    f = ConstBodyForce([0, 0, MASS * G, 0, 0, 0])
    q_true = R.quat_exp(torch.tensor([[0.0, 0.0, 3.0 * N_STEPS * R.DT / 2]], dtype=torch.float64))
    for integ, tol in (("rk4", 1e-10), ("euler", 1e-10)):
        out = R.rollout(f, ep, N_STEPS, integ, "full")
        assert torch.allclose(out["w"][0, -1], torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64), atol=1e-12)
        assert R.quat_angle(out["q"][0, -1:], q_true).item() < tol, integ


def test_convergence_order_on_coupled_motion():
    """Tumbling body (w not on a principal axis -> w x Jw torque-free precession) with body velocity
    (w x v coupling). Halving dt: RK4 error / 16, semi-implicit Euler error / 2."""
    t_end = 0.2
    f = ConstBodyForce([0.3, -0.2, MASS * G * 1.1, 0.002, -0.001, 0.0005])

    def final(integ, dt):
        n = int(round(t_end / dt))
        ep = make_episode(B=1, T=R.WINDOW + n, v0=(2.0, -1.0, 0.5), w0=(4.0, -3.0, 2.0), dt=dt)
        o = R.rollout(f, ep, n, integ, "full")
        return torch.cat([o["pos"][0, -1], o["vel"][0, -1], o["w"][0, -1]])

    ref = final("rk4", 0.0025 / 32)
    for integ, ratio in (("rk4", 16.0), ("euler", 2.0)):
        e1 = (final(integ, 0.0025) - ref).norm().item()
        e2 = (final(integ, 0.00125) - ref).norm().item()
        assert e1 / e2 == pytest.approx(ratio, rel=0.25), (integ, e1, e2)


def test_translational_mode_follows_recorded_attitude_and_ignores_torque():
    ep = make_episode(w0=(0.0, 0.0, 1.0))
    a = R.rollout(ConstBodyForce([0, 0, MASS * G, 0, 0, 0]), ep, N_STEPS, "rk4", "translational")
    b = R.rollout(ConstBodyForce([0, 0, MASS * G, 0.5, 0.5, 0.5]), ep, N_STEPS, "rk4", "translational")
    assert torch.equal(a["q"], ep.q[:, ep.H:]) and torch.allclose(a["pos"], b["pos"])


def test_gradcheck_float64():
    """End-to-end derivative of a short 6-DOF rollout through a linear force model, both integrators."""
    torch.manual_seed(0)
    Wlin = torch.randn(6, 18, dtype=torch.float64) * 1e-3

    def run(Wm, v0, integ):
        ep = make_episode(B=1, T=R.WINDOW + 3, v0=(1.0, 0.5, -0.2), w0=(0.5, -0.3, 0.2))
        base = torch.tensor([0, 0, MASS * G, 0, 0, 0], dtype=torch.float64)
        f = lambda win, k, tau: base + win[:, -1, :] @ Wm.T  # noqa: E731
        o = R.rollout(f, ep, 3, integ, "full", v0=v0)
        return torch.cat([o["pos"][0, -1], o["q"][0, -1], o["w"][0, -1]])

    v0 = torch.tensor([[1.0, 0.5, -0.2]], dtype=torch.float64, requires_grad=True)
    Wm = Wlin.clone().requires_grad_(True)
    for integ in ("euler", "rk4"):
        assert torch.autograd.gradcheck(lambda a, b: run(a, b, integ), (Wm, v0), eps=1e-6, atol=1e-6)


def test_quat_exp_gradient_finite_at_zero_rate():
    r = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    R.quat_exp(r).sum().backward()
    assert torch.isfinite(r.grad).all()


def test_gradient_reaches_forcemodel_parameters():
    """A small TCN ForceModel (with prior) on a synthetic episode: loss on the final position has
    finite, non-zero gradients for every backbone parameter in both modes and both integrators."""
    from src.models import build_model
    torch.manual_seed(0)
    m = build_model({"type": "tcn", "prior": True, "window": 20, "channels": 8, "head_hidden": 8})
    m.prior.fitted.fill_(True)
    m.prior.coef2.copy_(torch.tensor([0.0, MASS * G / (4 * 1500.0 ** 2), 0.0]))   # hover thrust at 1500 rad/s
    ep = make_episode(B=3, v0=(1.0, 0.0, 0.0), dtype=torch.float32)
    for mode in ("translational", "full"):
        for integ in ("euler", "rk4"):
            m.zero_grad()
            out = R.rollout(R.ModelForce(m), ep, 10, integ, mode)
            (out["pos"][:, -1] ** 2).sum().backward()
            g = [p.grad for p in m.parameters()]
            assert all(x is not None and torch.isfinite(x).all() for x in g), (mode, integ)
            assert sum(float(x.abs().sum()) for x in g) > 0, (mode, integ)


# ---------------------------------------------------------------- real data: one validation flight
def _have_data():
    import pandas as pd
    return (RAW_DIR / "processed_data").exists() or (CACHE_DIR / f"processed_all_pandas{pd.__version__}.pkl").exists()


@pytest.fixture(scope="module")
def val_piece():
    if not _have_data():
        pytest.skip("dataset not downloaded")
    from src.data.dataset import frame_pieces, read_splits, segment_ids
    from src.data.load import load_all
    splits = read_splits()
    ids = sorted(segment_ids("A", "val", splits))
    df = load_all()
    seg = df[df["segment_id"].astype(str) == ids[0]]           # first A/val segment (deterministic)
    pid, piece = max(frame_pieces(seg), key=lambda x: len(x[1]))
    return pid, piece


@pytest.fixture(scope="module")
def p0b():
    path = REPO / "results" / "val_baselines.json"
    if not path.exists():
        pytest.skip("results/val_baselines.json missing - run scripts/eval_baselines.py")
    return R.physics_model_from_coefs(json.loads(path.read_text())["splits"]["A"]["P0b_coefs"])


def _episode(piece, n=8):
    starts = np.linspace(R.WINDOW - 1, len(piece) - N_STEPS - 2, n).astype(int).tolist()
    return R.Episodes.from_piece(piece, starts, N_STEPS)


def test_oracle_rollout_tracks_recorded_trajectory(val_piece):
    """Driving the integrator with the recorded F and M must reproduce the recorded motion
    (checks frames, gravity sign, quaternion kinematics). Remaining drift = filter mismatch between
    acc / ang_acc and vel / ang_vel in the data (~0.2 m/s^2, CP1 test 2)."""
    ep = _episode(val_piece[1])
    for integ in ("euler", "rk4"):
        o = R.rollout(R.RecordedForce(ep), ep, N_STEPS, integ, "full")
        assert o["pos_err"][:, -1].max() < 0.01, integ                 # < 1 cm after 0.1 s
        assert o["vel_err"][:, -1].median() < 0.05, integ
        assert math.degrees(o["att_err"][:, -1].max()) < 1.0, integ
        assert o["w_err"][:, -1].median() < 0.1, integ


@pytest.mark.parametrize("mode", ["translational", "full"])
@pytest.mark.parametrize("integrator", ["euler", "rk4"])
def test_p0b_rollout_one_val_flight_0p1s(val_piece, p0b, mode, integrator):
    """P0b rolled out for 0.1 s from 8 start points on one A/val flight: finite, starts on the
    truth, drifts more than the oracle but stays physically small; RK4 and Euler agree at 2.5 ms."""
    ep = _episode(val_piece[1])
    with torch.no_grad():
        o = R.rollout(R.ModelForce(p0b), ep, N_STEPS, integrator, mode)
        oracle = R.rollout(R.RecordedForce(ep), ep, N_STEPS, integrator, mode)
        other = R.rollout(R.ModelForce(p0b), ep, N_STEPS, "euler" if integrator == "rk4" else "rk4", mode)
    for k in ("pos", "vel", "q", "w"):
        assert torch.isfinite(o[k]).all()
    assert o["pos_err"][:, 0].max() < 1e-6 and o["vel_err"][:, 0].max() < 1e-5
    assert o["pos_err"][:, -1].max() < 0.05                            # < 5 cm after 0.1 s
    assert o["vel_err"][:, -1].median() < 0.5
    assert o["vel_err"][:, -1].mean() > oracle["vel_err"][:, -1].mean()
    if mode == "full":
        assert math.degrees(o["att_err"][:, -1].median()) < 10
    assert (o["pos"] - other["pos"]).norm(dim=-1).max() < 2e-3          # integrator choice: < 2 mm
    print(f"\n[{val_piece[0]} {mode} {integrator}] P0b at 0.1 s: pos {o['pos_err'][:, -1].median():.4f} m, "
          f"vel {o['vel_err'][:, -1].median():.4f} m/s, att {math.degrees(o['att_err'][:, -1].median()):.3f} deg "
          f"| oracle pos {oracle['pos_err'][:, -1].median():.4f} m, vel {oracle['vel_err'][:, -1].median():.4f} m/s")


def test_p0b_rollout_is_differentiable_on_real_data(val_piece, p0b):
    """Gradient of the 0.1 s position error w.r.t. the initial velocity is finite and non-zero."""
    ep = _episode(val_piece[1], n=2)
    v0 = ep.feats[:, ep.H, R.V].clone().requires_grad_(True)
    o = R.rollout(R.ModelForce(p0b), ep, N_STEPS, "rk4", "full", v0=v0)
    o["pos_err"][:, -1].sum().backward()
    assert torch.isfinite(v0.grad).all() and v0.grad.abs().sum() > 0


def test_model_window_matches_training_windows(val_piece, p0b):
    """At step 0 the rollout's window equals the recorded window, so the first force equals the
    model applied to the dataset window (the history is the recorded data)."""
    from src.models import build_model
    torch.manual_seed(0)
    m = build_model({"type": "tcn", "prior": True, "window": 20, "channels": 8, "head_hidden": 8}).eval()
    ep = _episode(val_piece[1], n=3)
    seen = []
    R.rollout(lambda win, k, tau: (seen.append(win) or m(win)[0]), ep, 1, "euler", "full")
    assert torch.allclose(seen[0], ep.feats[:, :R.WINDOW], atol=1e-4)


# ---------------------------------------------------------------- RK4 stage windows (review fix, Nov 28)
def test_stage_window_interpolates_history_to_stage_time():
    """Rows that are linear in time stay linear and 1-dt apart for every tau; tau = 1 is the plain shift."""
    B, F, W = 2, 18, R.WINDOW
    t = torch.arange(W, dtype=torch.float64)                       # t_{k-W+1} .. t_k in units of dt
    rows = [torch.full((B, F), float(x), dtype=torch.float64) for x in t]
    for tau in (0.5, 1.0):
        last = torch.full((B, F), float(t[-1]) + tau, dtype=torch.float64)
        win = R.stage_window(rows, last, tau)
        assert win.shape == (B, W, F)
        assert torch.allclose(win[:, :, 0], (t + tau).expand(B, W))
        assert torch.allclose(win.diff(dim=1), torch.ones(B, W - 1, F, dtype=torch.float64))
    win1 = R.stage_window(rows, last, 1.0)
    assert torch.equal(win1, torch.stack(rows[1:] + [last], 1))


class WindowSpy:
    """Constant body force that records every input window it is given."""

    def __init__(self, fm):
        self.fm = torch.as_tensor(fm, dtype=torch.float64)
        self.wins = []

    def __call__(self, win, k, tau):
        self.wins.append((tau, win.detach().clone()))
        return self.fm.expand(win.shape[0], 6).to(win.dtype)


@pytest.mark.parametrize("mode", ["full", "translational"])
def test_rk4_stage_windows_are_uniformly_spaced(mode):
    """Constant acceleration a along body x with a recorded velocity ramp: every window the model sees,
    at every RK4 stage, has velocity steps of exactly a*dt (the old code gave a*dt/2 at tau = 0.5)."""
    a = 2.0
    ep = make_episode(B=1)
    T = ep.feats.shape[1]
    ep.feats[:, :, R.V.start if isinstance(R.V, slice) else R.V[0]] = a * (torch.arange(T, dtype=torch.float64) - ep.H) * R.DT
    spy = WindowSpy([MASS * a, 0, MASS * G, 0, 0, 0])
    R.rollout(spy, ep, 5, "rk4", mode)
    taus = {tau for tau, _ in spy.wins}
    assert taus == {0.0, 0.5, 1.0}
    vx = R.V.start if isinstance(R.V, slice) else R.V[0]
    for tau, win in spy.wins:
        dv = win[0, :, vx].diff()
        assert torch.allclose(dv, torch.full_like(dv, a * R.DT), atol=1e-12), (tau, dv)
