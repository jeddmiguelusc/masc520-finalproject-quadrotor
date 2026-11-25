"""Split-aware data access: whitelisted inputs, contact-trimmed targets, train-only
normalisation, 20-sample windows.

    from src.data.dataset import load_part, Normalizer, WindowSet
    train = load_part("A", "train")          # [(piece_id, X (T,18), Y (T,6)), ...]
    val   = load_part("A", "val")
    norm  = Normalizer.fit(train)            # statistics from TRAIN rows only
    ws    = WindowSet(train, window=20)      # numpy window index; ws.gather(i) -> (20,18), (6,)

Rules enforced here (tests/test_leakage.py checks each one):
  - Inputs are exactly INPUTS (18): vel, ang vel, gravity in body, motor speeds, dmot, vbat.
    acc / ang_acc (they define the targets), pos, raw quaternion and t never enter X.
  - Ground-contact samples are cut out of every segment (targets.trim_contact); a cut in the
    middle of a segment makes two pieces, and windows never cross piece boundaries.
  - A split key may carry a sample filter (e.g. {"max_speed": 5.0}); samples that fail it are
    cut the same way, so a window lies entirely inside the allowed region.
  - The test part of any split is refused unless allow_test=True (only scripts/evaluate_test.py,
    at CP7, passes it).
  - Normalizer.fit takes only training pieces; nothing about val/test can reach it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.data import frames as F
from src.data.features import FEATURES, build_features
from src.data.targets import TARGETS, build_targets, trim_contact

REPO = Path(__file__).resolve().parents[2]
SPLITS_FILE = REPO / "data" / "splits.json"
INPUTS = list(FEATURES)                      # the whitelist (order = model input order)
FORBIDDEN_SOURCE_COLUMNS = ["acc_x", "acc_y", "acc_z", "ang_acc_x", "ang_acc_y", "ang_acc_z",
                            "pos_x", "pos_y", "pos_z", "t"]
WINDOW = 20                                  # 20 samples = 50 ms at 400 Hz
assert len(INPUTS) == 18 and not set(INPUTS) & set(FORBIDDEN_SOURCE_COLUMNS)


# ---------------------------------------------------------------- splits file
def read_splits(path: Path | str = SPLITS_FILE) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run python scripts/make_splits.py")
    return json.loads(p.read_text())


def segment_ids(key: str, part: str, splits: dict | None = None, allow_test: bool = False) -> list[str]:
    if part not in ("train", "val", "test"):
        raise ValueError(part)
    if part == "test" and not allow_test:
        raise PermissionError("test data is evaluated once, at CP7 (scripts/evaluate_test.py)")
    splits = splits if splits is not None else read_splits()
    return list(splits[key][part])


def part_filter(key: str, part: str, splits: dict | None = None) -> dict:
    splits = splits if splits is not None else read_splits()
    return dict(splits[key].get("filter", {}).get(part, {}))


# ---------------------------------------------------------------- pieces
def _cut(piece: pd.DataFrame, keep: np.ndarray) -> list[pd.DataFrame]:
    d = np.diff(np.r_[0, keep.astype(int), 0])
    return [piece.iloc[a:b] for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0])]


def frame_pieces(df: pd.DataFrame, sample_filter: dict | None = None) -> list[tuple[str, pd.DataFrame]]:
    """Contact-trimmed (and optionally filtered) contiguous pieces of every segment in df.

    piece_id = "<segment_id>#<k>". Order: segment order of df, then time.
    """
    sample_filter = sample_filter or {}
    unknown = set(sample_filter) - {"max_speed"}
    if unknown:
        raise ValueError(f"unknown sample filter keys {unknown}")
    out = []
    for sid, g in df.groupby("segment_id", observed=True, sort=False):
        g = g.sort_values("t")
        pieces = trim_contact(g)
        if "max_speed" in sample_filter:
            vmax = float(sample_filter["max_speed"])
            pieces = [q for p in pieces
                      for q in _cut(p, np.linalg.norm(F.vec(p, "vel"), axis=1) <= vmax)]
        out += [(f"{sid}#{k}", p) for k, p in enumerate(pieces)]
    return out


def piece_arrays(pieces: Sequence[tuple[str, pd.DataFrame]]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    return [(pid, build_features(p), build_targets(p)) for pid, p in pieces]


def load_part(key: str, part: str, splits: dict | None = None, df: pd.DataFrame | None = None,
              allow_test: bool = False) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """[(piece_id, X (T,18) float32, Y (T,6) float32), ...] for one part of one split."""
    splits = splits if splits is not None else read_splits()
    ids = segment_ids(key, part, splits, allow_test)
    if df is None:
        from src.data.load import load_all
        df = load_all()
    sub = df[df["segment_id"].astype(str).isin(set(ids))]
    found = sub["segment_id"].astype(str).nunique()
    if found != len(ids):
        raise ValueError(f"{key}/{part}: {len(ids) - found} segment ids not found in the data")
    return piece_arrays(frame_pieces(sub, part_filter(key, part, splits)))


# ---------------------------------------------------------------- normalisation
class Normalizer:
    """Per-input mean/std from training rows only (same formula as models.core.fit_normalisers)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, np.float64)
        self.std = np.asarray(std, np.float64)

    @classmethod
    def fit(cls, train_pieces: Sequence[tuple[str, np.ndarray, np.ndarray]]) -> "Normalizer":
        X = np.concatenate([x for _, x, _ in train_pieces]).astype(np.float64)
        return cls(X.mean(0), np.maximum(X.std(0), 1e-6))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"inputs": INPUTS, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        if d["inputs"] != INPUTS:
            raise ValueError("normaliser was fitted on a different input list")
        return cls(d["mean"], d["std"])


# ---------------------------------------------------------------- windows
class WindowSet:
    """Index of `window`-sample windows that lie inside one piece (numpy; no copies).

    Window i covers rows ends[i]-window+1 .. ends[i] of the concatenated pieces; its target is
    Y[ends[i]]. Every sample with >= window-1 earlier samples in its own piece is an end.
    """

    def __init__(self, pieces: Sequence[tuple[str, np.ndarray, np.ndarray]], window: int = WINDOW,
                 normalizer: Normalizer | None = None):
        xs, ys, ends, owner = [], [], [], []
        off = 0
        for k, (_, X, Y) in enumerate(pieces):
            n = len(X)
            if n >= window:
                e = np.arange(window - 1, n) + off
                ends.append(e); owner.append(np.full(len(e), k))
            xs.append(X); ys.append(Y); off += n
        self.window = window
        self.piece_ids = [p for p, _, _ in pieces]
        X = np.concatenate(xs) if xs else np.zeros((0, len(INPUTS)), np.float32)
        self.X = normalizer.transform(X) if normalizer is not None else X.astype(np.float32)
        self.Y = np.concatenate(ys) if ys else np.zeros((0, len(TARGETS)), np.float32)
        self.ends = np.concatenate(ends) if ends else np.zeros(0, np.int64)
        self.owner = np.concatenate(owner) if owner else np.zeros(0, np.int64)
        starts = np.cumsum([0] + [len(x) for x in xs])
        self.piece_bounds = np.stack([starts[:-1], starts[1:]], 1)  # [start, end) per piece

    def __len__(self) -> int:
        return len(self.ends)

    def rows(self, i: int) -> np.ndarray:
        e = self.ends[i]
        return np.arange(e - self.window + 1, e + 1)

    def gather(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        e = self.ends[i]
        return self.X[e - self.window + 1:e + 1], self.Y[e]

    def torch_dataset(self):
        """torch Dataset view (imports torch lazily)."""
        import torch

        ws = self

        class _DS(torch.utils.data.Dataset):
            def __len__(self):
                return len(ws)

            def __getitem__(self, i):
                x, y = ws.gather(i)
                return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))

        return _DS()
