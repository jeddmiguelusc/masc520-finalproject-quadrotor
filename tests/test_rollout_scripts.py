"""CP5/CP6: src/sim/pool.py, scripts/eval_rollout_horizons.py and scripts/finetune_rollout.py on synthetic
flights (tests/synthetic_flights.py) in a temporary repo layout. No dataset needed."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_flights import flights  # noqa: E402

from src.data.dataset import frame_pieces, piece_arrays  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.physics_np import fit_coefs  # noqa: E402
from src.sim import rollout as R  # noqa: E402
from src.sim.pool import EpisodePool  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- pool
def test_pool_respects_gaps_and_pieces():
    df = flights()
    pool = EpisodePool(frame_pieces(df), 200)
    t = np.concatenate([p["t"].to_numpy() for _, p in frame_pieces(df)])
    for g in pool.starts:                                  # every episode span is gap-free, same piece
        dts = np.diff(t[g - 19:g + 201])
        assert np.allclose(dts, R.DT, atol=1e-9)
    # the segment with a gap at sample 600 (n=1200) loses the starts whose span covers it
    pid = "2021-01-01-12-00-00_seg_1#0"
    k = pool.piece_ids.index(pid)
    assert (pool.owner == k).sum() == (600 - 19 - 200) + (600 - 19 - 200)
    sel = pool.select_per_flight(200)
    assert all((pool.flights[sel] == f).sum() == 200 for f in pool.flight_ids())
    sel2 = pool.select_per_flight(50, exclude=sel)
    assert len(np.intersect1d(sel, sel2)) == 0 and len(sel2) == 150


def test_pool_episodes_match_from_piece():
    df = flights(1, 1)
    (pid, piece), = frame_pieces(df)
    pool = EpisodePool([(pid, piece)], 40)
    k0 = [19, 500]
    ep_a = pool.episodes(np.searchsorted(pool.starts, k0), 40, relative_pos=False)
    ep_b = R.Episodes.from_piece(piece, k0, 40)
    for x, y in ((ep_a.feats, ep_b.feats), (ep_a.q, ep_b.q), (ep_a.pos, ep_b.pos), (ep_a.Y, ep_b.Y)):
        assert torch.allclose(x, y, atol=1e-5)
    ep_r = pool.episodes(np.searchsorted(pool.starts, k0), 40)
    assert ep_r.pos[:, ep_r.H].abs().max() == 0


def test_truncated_bptt_same_forward_shorter_gradient():
    df = flights(1, 1)
    pool = EpisodePool(frame_pieces(df), 20)
    ep = pool.episodes(np.arange(4), 20)
    torch.manual_seed(0)
    m = build_model({"type": "mlp", "prior": False, "hidden": [8]})
    m.out_mean.fill_(0.0); m.out_mean[2] = 7.5
    full = R.rollout(R.ModelForce(m), ep, 20, "euler", "translational")
    trunc = R.rollout(R.ModelForce(m), ep, 20, "euler", "translational", detach_every=5)
    assert torch.allclose(full["pos"], trunc["pos"])
    g = []
    for o in (full, trunc):
        m.zero_grad()
        o["pos"][:, -1].pow(2).sum().backward(retain_graph=True)
        g.append(torch.cat([p.grad.flatten() for p in m.parameters()]))
    assert torch.isfinite(g[1]).all() and not torch.allclose(g[0], g[1])
    with pytest.raises(ValueError):
        R.rollout(R.ModelForce(m), ep, 20, "euler", "translational", detach_every=0)


# ---------------------------------------------------------------- scripts end to end
@pytest.fixture()
def fake_repo(tmp_path, monkeypatch):
    df = flights(n_flights=4, segs=2, n=1400)
    fl = sorted(df["flight_id"].unique())
    seg = lambda f: sorted(df.loc[df.flight_id == f, "segment_id"].unique())  # noqa: E731
    splits = {"meta": {"sha256": "synthetic"},
              "A": {"train": seg(fl[0]) + seg(fl[1]), "val": seg(fl[2]) + seg(fl[3]), "test": []}}
    (tmp_path / "results").mkdir()
    train_pieces = frame_pieces(df[df.segment_id.isin(splits["A"]["train"])])
    arr = piece_arrays(train_pieces)
    X = np.concatenate([x for _, x, _ in arr]); Y = np.concatenate([y for _, _, y in arr])
    coefs, _ = fit_coefs(X, Y)
    (tmp_path / "results" / "val_baselines.json").write_text(
        json.dumps({"splits": {"A": {"P0b_coefs": [c.tolist() for c in coefs]}}}))
    torch.manual_seed(0)
    for run, mc in (("m2_hybrid_mlp_tuned_seed0", {"type": "mlp", "prior": True, "hidden": [16, 16]}),
                    ("m3_hybrid_tcn_seed0", {"type": "tcn", "prior": True, "window": 20, "channels": 8,
                                             "head_hidden": 8})):
        m = build_model(mc)
        m.fit_normalisers(X, X, Y)
        name = run.rsplit("_seed", 1)[0]
        cfg = {"name": name, "seed": 0, "model": mc, "data": {"split": "A", "max_window": 20, "source": "neurobem"},
               "train": {"weight_decay": 0.1, "eval_batch_size": 4096}}
        (tmp_path / "checkpoints" / run).mkdir(parents=True)
        torch.save({"model": m.state_dict(), "config": cfg}, tmp_path / "checkpoints" / run / "best.pt")
    mods = {n: _load(n) for n in ("eval_rollout_horizons", "finetune_rollout", "eval_rollout", "eval_models")}
    for mod in mods.values():
        monkeypatch.setattr(mod, "REPO", tmp_path)
        monkeypatch.setattr(mod, "load_all", lambda: df)
        monkeypatch.setattr(mod, "read_splits", lambda: splits)
    return tmp_path, mods


def test_eval_rollout_end_to_end(fake_repo):
    root, mods = fake_repo
    mods["eval_rollout_horizons"].main(["--runs", "m2_hybrid_mlp_tuned_seed0", "m3_hybrid_tcn_seed0", "missing_seed0",
                               "--no-ensembles", "--n-starts", "60", "--device", "cpu", "--batch", "50"])
    out = json.loads((root / "results" / "rollout_horizons_val.json").read_text())
    assert set(out["models"]) == {"oracle", "P0b", "m2_hybrid_mlp_tuned_seed0", "m3_hybrid_tcn_seed0"}
    assert out["missing"] == ["missing_seed0"] and out["n_starts"] == 120
    o = out["models"]["oracle"]["pos_err_m"]["all"]
    p = out["models"]["P0b"]["pos_err_m"]["all"]
    assert o["0.1"]["median"] < o["0.5"]["median"]                  # error grows with horizon
    assert o["0.5"]["median"] < p["0.5"]["median"]                  # oracle beats a fitted model
    s = p["0.25"]
    assert s["p25"] <= s["median"] <= s["p75"] and s["median_ci95"][0] <= s["median_ci95"][1]
    assert len(s["per_flight_median"]) == 2
    pairs = {(d["a"], d["b"], d["subset"], d["horizon_s"]) for d in out["paired"]}
    assert ("M3", "M2", "all", 0.25) in pairs and ("M2", "P0b", "all", 0.25) in pairs
    npz = np.load(root / "results" / "rollout_horizons_val_errors.npz", allow_pickle=True)
    assert npz["pos_err"].shape == (4, 120, 3)


def test_finetune_rollout_smoke(fake_repo):
    root, mods = fake_repo
    mods["finetune_rollout"].main(["--seed", "0", "--max-epochs", "2", "--batches-per-epoch", "3", "--batch", "16",
                                   "--batch-1step", "64", "--val-starts", "10", "--device", "cpu", "--tbptt", "5"])
    rep = json.loads((root / "results" / "finetune_rollout_seed0.json").read_text())
    assert len(rep["history"]) == 3 and rep["best_epoch"] in (0, 1, 2)
    assert rep["skipped_batches"] == 0 and rep["config"]["tbptt"] == 5
    ck = torch.load(root / "checkpoints" / "m2_hybrid_mlp_rollft_seed0" / "best.pt", weights_only=False)
    assert ck["config"]["name"] == "m2_hybrid_mlp_rollft" and ck["config"]["finetune"]["n_steps"] == 20
    m = build_model(ck["config"]["model"]); m.load_state_dict(ck["model"])     # loadable by eval scripts
    rows = (root / "results" / "runs.csv").read_text().splitlines()
    assert len(rows) == 2 and "m2_hybrid_mlp_rollft_seed0" in rows[1]
    # the fine-tuned run is picked up by eval_rollout under label M2roll, paired with M2
    mods["eval_rollout_horizons"].main(["--runs", "m2_hybrid_mlp_tuned_seed0", "m2_hybrid_mlp_rollft_seed0", "--no-ensembles",
                               "--no-baselines", "--n-starts", "20", "--device", "cpu"])
    out = json.loads((root / "results" / "rollout_horizons_val.json").read_text())
    assert out["models"]["m2_hybrid_mlp_rollft_seed0"]["label"] == "M2roll"
    assert any(d["a"] == "M2roll" and d["b"] == "M2" for d in out["paired"])


# ---------------------------------------------------------------- final-model registry (review fix, Nov 28)
REG = """
models:
  M2: {split: A, runs: [m2_hybrid_mlp_tuned_seed0]}
  M3: {split: A, runs: [m3_hybrid_tcn_seed0]}
ensembles:
  M3ens: {split: A, members: [m3_hybrid_tcn_seed0]}
headline: {A: [M2, M3]}
pairs_1step: {A: [[M3, M2, A_val, all]]}
pairs_rollout: [[M3, M2], [M3ens, M2]]
rollout_reference: M2
"""


def test_horizons_final_uses_registry_labels_and_oracle_pairs(fake_repo):
    root, mods = fake_repo
    (root / "reg.yaml").write_text(REG)
    mods["eval_rollout_horizons"].main(["--final", "--registry", str(root / "reg.yaml"), "--n-starts", "40",
                                        "--device", "cpu", "--batch", "50"])
    out = json.loads((root / "results" / "rollout_horizons_val.json").read_text())
    assert out["registry"]["labels"]["M3"] == ["m3_hybrid_tcn_seed0"]
    assert {m["label"] for m in out["models"].values()} == {"oracle", "P0b", "M2", "M3", "M3ens"}
    pairs = {(d["a"], d["b"], d["a_run"], d["b_run"]) for d in out["paired"] if d["subset"] == "all" and d["horizon_s"] == 0.25}
    assert ("M3", "M2", "m3_hybrid_tcn_seed0", "m2_hybrid_mlp_tuned_seed0") in pairs
    assert {(a, b) for a, b, _, _ in pairs} >= {("M2", "oracle"), ("M3", "oracle"), ("M3ens", "oracle"), ("P0b", "oracle")}
    ex = next(d for d in out["paired"] if d["a"] == "M3" and d["b"] == "oracle" and d["subset"] == "all")
    assert ex["diff_median"] > 0 and ex["n_diverged_a"] == 0      # nothing beats the recorded forces
    assert "failure_policy" in out


def test_eval_rollout_final_and_failure_policy(fake_repo):
    root, mods = fake_repo
    (root / "reg.yaml").write_text(REG)
    mods["eval_rollout"].main(["--final", "--registry", str(root / "reg.yaml"), "--stride", "100", "--steps", "40",
                               "--modes", "translational", "--integrators", "rk4", "--device", "cpu",
                               "--out", "results/val_rollout_final.json"])
    out = json.loads((root / "results" / "val_rollout_final.json").read_text())
    assert set(out["models"]) == {"oracle", "P0b", "M2", "M3", "M3ens"}
    s = out["models"]["M3"]["translational/rk4"]["per_seed"][0]["horizons"]["0.1s"]["pos"]
    assert s["all"]["n_diverged"] == 0 and s["all"]["n_flights"] == 2
    refs = {(d["a"], d["b"]) for d in out["paired_pos_rmse_final"]}
    assert ("M3", "M2") in refs and ("M3", "oracle") in refs and ("M2", "oracle") in refs
    assert all(np.isfinite(d["diff_rms_m"]) for d in out["paired_pos_rmse_final"])


def test_final_refuses_checkpoint_from_another_split(fake_repo):
    root, mods = fake_repo
    ck = root / "checkpoints" / "m3_hybrid_tcn_seed0" / "best.pt"
    d = torch.load(ck, weights_only=False); d["config"]["data"]["split"] = "A_r1"; torch.save(d, ck)
    (root / "reg.yaml").write_text(REG)
    with pytest.raises(ValueError, match="split A_r1"):
        mods["eval_rollout_horizons"].main(["--final", "--registry", str(root / "reg.yaml"), "--n-starts", "10",
                                            "--device", "cpu", "--no-baselines"])
    # without --final an ensemble member from another split is refused too (it used to be scored silently)
    with pytest.raises(ValueError, match="member"):
        mods["eval_rollout_horizons"].main(["--runs", "--ensemble", "E=m3_hybrid_tcn_seed0", "--n-starts", "10",
                                            "--device", "cpu", "--no-baselines"])


def test_eval_models_final_writes_run_names_and_refuses_label_clash(fake_repo):
    root, mods = fake_repo
    vb = json.loads((root / "results" / "val_baselines.json").read_text())
    vb["metric_definition"] = "test"; vb["splits"]["A"]["eval"] = {}
    (root / "results" / "val_baselines.json").write_text(json.dumps(vb))
    (root / "results" / "runs.csv").write_text("run,val_rmse_F_N\n")
    (root / "reg.yaml").write_text(REG)
    em = mods["eval_models"]
    em.main(["--final", "--registry", str(root / "reg.yaml"), "--device", "cpu",
             "--out", "results/vm.json", "--out-unc", "results/vu.json"])
    out = json.loads((root / "results" / "vm.json").read_text())
    assert out["runs"]["m3_hybrid_tcn_seed0"]["label"] == "M3" and out["ensembles"]["M3ens"]["label"] == "M3ens"
    p = next(d for d in out["paired"] if d["a"] == "M3" and d["b"] == "M2")
    assert p["a_run"] == "m3_hybrid_tcn_seed0" and p["b_run"] == "m2_hybrid_mlp_tuned_seed0"
    # two different checkpoints under one label and seed are refused (they used to overwrite each other)
    (root / "reg2.yaml").write_text(REG.replace("M2: {split: A, runs: [m2_hybrid_mlp_tuned_seed0]}",
                                                "M2: {split: A, runs: [m2_hybrid_mlp_tuned_seed0]}\n  M2b: {split: A, runs: [m3_hybrid_tcn_seed0]}"))
    with pytest.raises(ValueError, match="listed under both"):
        em.main(["--final", "--registry", str(root / "reg2.yaml"), "--device", "cpu", "--out", "results/vm2.json"])
