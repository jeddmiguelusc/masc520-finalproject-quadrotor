"""CP2 leakage tests for data/splits.json and src/data/dataset.py.

    pytest tests/test_leakage.py -v

Checks: parts disjoint by flight; every used segment assigned exactly once; shared test set
across Split A resamples with disjoint val folds; Split B speed rule; no forbidden inputs;
normaliser depends on train rows only; windows never cross a segment, a contact cut or a speed
cut; test data refused outside CP7. Skips if the dataset is not downloaded.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.data import frames as F
from src.data.dataset import (FORBIDDEN_SOURCE_COLUMNS, INPUTS, SPLITS_FILE, Normalizer, WindowSet,
                              frame_pieces, load_part, read_splits)
from src.data.features import build_features
from src.data.load import RAW_DIR
from src.data.targets import contact_mask

pytestmark = pytest.mark.skipif(not (RAW_DIR / "processed_data").exists(), reason="dataset not downloaded")
PARTS = ("train", "val", "test")
A_KEYS = ["A", "A_r1", "A_r2", "A_r3"]


@pytest.fixture(scope="module")
def df():
    from src.data.load import load_all
    return load_all()


@pytest.fixture(scope="module")
def idx():
    from src.data.load import segment_index
    return segment_index()


@pytest.fixture(scope="module")
def splits():
    return read_splits()


def keys(splits):
    return [k for k in splits if k != "meta"]


# ---------------------------------------------------------------- split file
def test_splits_file_is_reproducible(df, idx, splits):
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_splits", SPLITS_FILE.parents[1] / "scripts" / "make_splits.py")
    ms = importlib.util.module_from_spec(spec); spec.loader.exec_module(ms)
    rebuilt, _ = ms.build(df, idx)
    assert rebuilt["meta"]["sha256"] == splits["meta"]["sha256"]


def test_expected_keys(splits):
    assert set(keys(splits)) == {"A", "A_r1", "A_r2", "A_r3", "B", "B_s"}


def test_parts_disjoint_by_flight_and_segment(splits, idx):
    seg2flight = dict(zip(idx.segment_id, idx.flight_id))
    for k in keys(splits):
        for p, q in itertools.combinations(PARTS, 2):
            assert not set(splits[k][p]) & set(splits[k][q]), (k, p, q)
            fp = {seg2flight[s] for s in splits[k][p]}
            fq = {seg2flight[s] for s in splits[k][q]}
            assert not fp & fq, f"{k}: flight in both {p} and {q}: {fp & fq}"
        for p in PARTS:   # segment lists agree with the stored flight lists
            assert {seg2flight[s] for s in splits[k][p]} == set(splits[k]["flights"][p])


def test_every_used_segment_assigned_once(splits, idx):
    excluded = set(splits["meta"]["excluded_flights"])
    used = set(idx.segment_id[~idx.flight_id.isin(excluded)])
    for k in keys(splits):
        allsegs = [s for p in PARTS for s in splits[k][p]]
        assert len(allsegs) == len(set(allsegs)) and set(allsegs) == used, k
        assert not {f for p in PARTS for f in splits[k]["flights"][p]} & excluded


def test_split_a_proportions(splits, idx):
    n = len(set(idx.flight_id)) - len(splits["meta"]["excluded_flights"])
    for k in A_KEYS:
        fr = {p: len(splits[k]["flights"][p]) / n for p in PARTS}
        assert abs(fr["train"] - 0.70) < 0.04 and abs(fr["val"] - 0.20) < 0.04 and abs(fr["test"] - 0.10) < 0.02, fr


def test_a_resamples_share_test_and_have_disjoint_val(splits):
    tests = {tuple(splits[k]["test"]) for k in A_KEYS}
    assert len(tests) == 1
    for a, b in itertools.combinations(A_KEYS, 2):
        assert not set(splits[a]["val"]) & set(splits[b]["val"]), (a, b)
    assert splits["B_s"]["test"] == splits["A"]["test"]


def test_split_b_speed_rule(splits, df):
    vmax = splits["meta"]["b_max_speed"]
    speed = np.linalg.norm(F.vec(df, "vel"), axis=1)
    fmax = df.assign(s=speed).groupby("flight_id", observed=True).s.max()
    for p in ("train", "val"):
        assert (fmax[splits["B"]["flights"][p]] <= vmax).all()
    assert (fmax[splits["B"]["flights"]["test"]] > vmax).all()


# ---------------------------------------------------------------- inputs / targets
def test_inputs_whitelist_and_forbidden_columns_have_no_effect(df):
    assert INPUTS == ["vel_x", "vel_y", "vel_z", "ang_vel_x", "ang_vel_y", "ang_vel_z", "g_b_x", "g_b_y",
                      "g_b_z", "mot1", "mot2", "mot3", "mot4", "dmot1", "dmot2", "dmot3", "dmot4", "vbat"]
    seg = df[df.segment_id == df.segment_id.cat.categories[0]].copy()
    X0 = build_features(seg)
    rng = np.random.default_rng(0)
    for c in FORBIDDEN_SOURCE_COLUMNS:
        seg[c] = rng.normal(size=len(seg)) * 100
    assert X0.shape[1] == 18 and np.array_equal(X0, build_features(seg))


def test_no_contact_samples_in_any_piece(df, splits):
    sub = df[df.segment_id.astype(str).isin(set(splits["A"]["train"]) | set(splits["A"]["val"]))]
    pieces = frame_pieces(sub)
    assert sum(len(p) for _, p in pieces) == int((~contact_mask(sub)).sum())
    assert not any(contact_mask(p).any() for _, p in pieces)


def test_b_s_train_val_samples_are_slow(df, splits):
    for p in ("train", "val"):
        pcs = load_part("B_s", p, splits, df)
        assert max(float(np.linalg.norm(x[:, 0:3], axis=1).max()) for _, x, _ in pcs) <= 5.0 + 1e-5


# ---------------------------------------------------------------- normalisation
def test_normaliser_uses_train_rows_only(df, splits):
    tr = load_part("A", "train", splits, df)
    n1 = Normalizer.fit(tr)
    other = set(splits["A"]["val"]) | set(splits["A"]["test"])
    df2 = df.copy()
    m = df2.segment_id.astype(str).isin(other).to_numpy()
    for c in ["vel_x", "mot1", "dmot2", "vbat", "ang_vel_z"]:
        df2.loc[m, c] = df2.loc[m, c] * 100 + 1e3
    n2 = Normalizer.fit(load_part("A", "train", splits, df2))
    assert np.array_equal(n1.mean, n2.mean) and np.array_equal(n1.std, n2.std)
    X = np.concatenate([x for _, x, _ in tr]).astype(np.float64)
    assert np.allclose(n1.mean, X.mean(0)) and np.allclose(n1.std, np.maximum(X.std(0), 1e-6))


def test_normaliser_matches_model_normaliser(df, splits):
    torch = pytest.importorskip("torch")
    from src.models import build_model
    tr = load_part("A", "train", splits, df)[:20]
    norm = Normalizer.fit(tr)
    ws = WindowSet(tr, 20)
    model = build_model({"type": "mlp", "prior": False, "nll": False, "window": 1})
    model.fit_normalisers(np.concatenate([x for _, x, _ in tr]), ws.X[ws.ends], ws.Y[ws.ends])
    assert np.allclose(model.x_mean.numpy(), norm.mean, rtol=1e-4, atol=1e-4)
    assert np.allclose(model.x_std.numpy(), norm.std, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------- windows
def test_windows_stay_inside_one_piece(df, splits):
    for key, part in [("A", "val"), ("B_s", "val")]:
        sub = df[df.segment_id.astype(str).isin(set(splits[key][part]))]
        pieces = frame_pieces(sub, splits[key].get("filter", {}).get(part))
        arrs = [(pid, build_features(p), p["t"].to_numpy()[:, None].astype(np.float64)) for pid, p in pieces]
        ws = WindowSet([(pid, x, t) for pid, x, t in arrs], 20)   # Y slot carries t
        assert len(ws) == sum(max(0, len(x) - 19) for _, x, _ in arrs)
        lo, hi = ws.piece_bounds[ws.owner, 0], ws.piece_bounds[ws.owner, 1]
        assert ((ws.ends - 19 >= lo) & (ws.ends < hi)).all()
        tt = np.concatenate([t for _, _, t in arrs])[:, 0]
        span = tt[ws.ends] - tt[ws.ends - 19]
        assert np.allclose(span, 19 * 0.0025, atol=2e-4), (key, span.min(), span.max())


def test_window_gather_and_torch_view(df, splits):
    pcs = load_part("A", "val", splits, df)[:3]
    ws = WindowSet(pcs, 20, Normalizer.fit(pcs))
    x, y = ws.gather(0)
    assert x.shape == (20, 18) and y.shape == (6,)
    assert np.array_equal(y, pcs[0][2][19])
    torch = pytest.importorskip("torch")
    xb, yb = ws.torch_dataset()[5]
    assert tuple(xb.shape) == (20, 18) and tuple(yb.shape) == (6,)


# ---------------------------------------------------------------- test access
def test_test_part_refused(splits, df):
    with pytest.raises(PermissionError):
        load_part("A", "test", splits, df)


def test_train_py_uses_trimmed_pieces(splits):
    pytest.importorskip("torch")
    from src.train import load_split
    with pytest.raises(PermissionError):
        load_split({"source": "neurobem", "split": "A"}, "test")
    segs = load_split({"source": "neurobem", "split": "B", "max_window": 20}, "val")
    assert len(segs) == len(load_part("B", "val"))
