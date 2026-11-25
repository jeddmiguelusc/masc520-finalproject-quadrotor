"""Model + training tests on synthetic data only (no NeuroBEM data needed). Run:
    pytest tests/test_models.py -v
"""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch

from src.data.features import N_FEATURES
from src.data.synthetic import TRUE_COEFS, make_segments, physics_targets
from src.data.windows import WindowedData
from src.models import build_model
from src.models.core import n_params
from src.models.ensemble import combine
from src.models.physics import PhysicsPrior
from src.train import load_config, load_split, run

CONFIGS = {
    "p0a": {"type": "constant"},
    "p0b": {"type": "physics"},
    "m1": {"type": "mlp", "prior": False},
    "m2": {"type": "mlp", "prior": True},
    "m3": {"type": "tcn", "prior": True, "window": 20},
    "m3_nll": {"type": "tcn", "prior": True, "window": 20, "nll": True},
    "m4": {"type": "transformer", "prior": True, "window": 20},
}
NEURAL = ["m1", "m2", "m3", "m3_nll", "m4"]


def _cfg(tmp_path, mcfg, **train):
    return load_config(None, model=mcfg, seed=0,
                       data={"source": "synthetic", "n_segments": 6, "segment_length": 800, "max_window": 20},
                       train={"max_epochs": 2, "batch_size": 256, "device": "cpu", **train},
                       log={"runs_csv": str(tmp_path / "runs.csv"), "ckpt_dir": str(tmp_path / "ckpt")})


# ---------------------------------------------------------------- shapes and size
@pytest.mark.parametrize("name", list(CONFIGS))
def test_forward_shapes(name):
    m = build_model(CONFIGS[name])
    mean, var = m(torch.randn(7, 20, N_FEATURES))
    assert mean.shape == (7, 6)
    assert (var is None) == (not CONFIGS[name].get("nll", False))
    if var is not None:
        assert var.shape == (7, 6) and (var > 0).all()


@pytest.mark.parametrize("name", NEURAL)
def test_param_budget_about_25k(name):
    p = n_params(build_model(CONFIGS[name]))
    assert 18_000 <= p <= 32_000, p           # equal-capacity rule: every network ~25k


def test_physics_has_15_coefficients_and_no_trainable_params():
    m = build_model(CONFIGS["p0b"])
    assert m.prior.n_coefs == 15 and n_params(m) == 0


# ---------------------------------------------------------------- physics prior
def test_physics_least_squares_recovers_true_coefficients():
    rng = np.random.default_rng(0)
    X = np.concatenate([s[0] for s in make_segments(4, 3000, seed=1)])
    Y = physics_targets(X) + rng.normal(0, 1e-4, (len(X), 6))
    p = PhysicsPrior()
    r2 = p.fit(X, Y)
    c = p.coefficients()
    assert np.isclose(c["out2:sum_w2"], TRUE_COEFS["kT"], rtol=1e-3)
    assert np.isclose(c["out0:v_x"], TRUE_COEFS["drag_xy"], rtol=1e-2)
    assert np.isclose(c["out3:roll_w2"], TRUE_COEFS["k_roll"], rtol=1e-3)
    assert np.isclose(c["out5:yaw_w2"], TRUE_COEFS["k_yaw"], rtol=1e-2)
    assert min(r2.values()) > 0.99
    pred = p(torch.from_numpy(X)).numpy()                       # float32 forward agrees with fit
    assert np.abs(pred - physics_targets(X)).max() < 1e-2


# ---------------------------------------------------------------- windows
def test_windows_never_cross_segments_and_end_at_the_target_sample():
    segs = [(np.full((30, N_FEATURES), k, np.float32), np.full((30, 6), k, np.float32)) for k in range(3)]
    segs.append((np.zeros((10, N_FEATURES), np.float32), np.zeros((10, 6), np.float32)))  # too short: no samples
    d = WindowedData(segs, max_window=20)
    assert len(d) == 3 * (30 - 19)
    x, y = d.gather(d.ends, 20)
    assert (x.amin(dim=(1, 2)) == x.amax(dim=(1, 2))).all()      # each window from one segment
    assert torch.equal(x[:, -1, 0], y[:, 0])                     # last input row = target row


def test_every_model_scores_the_same_samples():
    segs = make_segments(3, 500, seed=2)
    assert len(WindowedData(segs, 20)) == sum(len(s[0]) - 19 for s in segs)


def test_tcn_is_causal():
    bb = build_model(CONFIGS["m3"]).backbone
    x = torch.randn(2, 20, N_FEATURES)
    h0 = bb.sequence_features(x)
    x2 = x.clone(); x2[:, 12] += 5.0                              # perturb step 12 only
    h1 = bb.sequence_features(x2)
    assert torch.allclose(h0[:, :, :12], h1[:, :, :12])           # the past never sees the future
    assert not torch.allclose(h0[:, :, 12:], h1[:, :, 12:])


def test_tcn_last_output_sees_whole_window():
    bb = build_model(CONFIGS["m3"]).backbone
    x = torch.randn(2, 20, N_FEATURES)
    x2 = x.clone(); x2[:, 0] += 5.0                               # oldest step in the window
    assert not torch.allclose(bb(x), bb(x2))                      # receptive field 31 >= 20


# ---------------------------------------------------------------- training
@pytest.mark.parametrize("name", list(CONFIGS))
def test_two_epochs_train_and_log(tmp_path, name):
    out = run(_cfg(tmp_path, CONFIGS[name]), verbose=False)
    row = out["row"]
    assert row["epochs_run"] == (0 if name in ("p0a", "p0b") else 2)
    assert np.isfinite(row["best_val_loss"])
    rows = list(csv.DictReader((tmp_path / "runs.csv").open()))
    assert len(rows) == 1 and rows[0]["run"].endswith("_seed0")
    assert json.loads(rows[0]["config_json"])["model"]["type"] == CONFIGS[name]["type"]
    assert rows[0]["git_hash"] != ""


def test_training_reduces_loss_and_hybrid_beats_constant(tmp_path):
    p0a = run(_cfg(tmp_path / "a", CONFIGS["p0a"]), verbose=False)["final"]
    m3 = run(_cfg(tmp_path / "b", CONFIGS["m3"], max_epochs=6), verbose=False)
    hist = json.loads((m3["ckpt_dir"] / "history.json").read_text())["history"]
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert m3["final"]["rmse_F"] < 0.5 * p0a["rmse_F"]


def test_resume_continues_from_checkpoint(tmp_path):
    cfg = _cfg(tmp_path, CONFIGS["m2"], max_epochs=2)
    run(cfg, verbose=False)
    cfg["train"]["max_epochs"] = 4
    out = run(cfg, resume=True, verbose=False)
    hist = json.loads((out["ckpt_dir"] / "history.json").read_text())["history"]
    assert [h["epoch"] for h in hist] == [1, 2, 3, 4]
    assert out["row"]["epochs_run"] == 4


def test_same_seed_same_result(tmp_path):
    a = run(_cfg(tmp_path / "a", CONFIGS["m2"]), verbose=False)["final"]["loss"]
    b = run(_cfg(tmp_path / "b", CONFIGS["m2"]), verbose=False)["final"]["loss"]
    assert a == pytest.approx(b, rel=1e-6)


def test_train_script_refuses_test_split():
    with pytest.raises(PermissionError):
        load_split({"source": "synthetic"}, "test")


# ---------------------------------------------------------------- ensemble
def test_ensemble_mixture_variance():
    m = [torch.tensor([[0.0]]), torch.tensor([[2.0]])]
    v = [torch.tensor([[1.0]]), torch.tensor([[1.0]])]
    out = combine(m, v)
    assert out["mean"].item() == 1.0
    assert out["epistemic"].item() == 1.0 and out["aleatoric"].item() == 1.0
    assert out["var"].item() == 2.0
