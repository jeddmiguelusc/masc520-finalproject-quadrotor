"""Rollout start points over a whole split part (CP5/CP6): one concatenated array per quantity,
plus the list of valid start samples. Shared by scripts/eval_rollout.py and
scripts/finetune_rollout.py.

    pool = EpisodePool(frame_pieces(df_val), n_steps=200)       # 0.5 s horizon at 400 Hz
    sel  = pool.select_per_flight(200)                          # >= 200 evenly spaced starts per flight
    ep   = pool.episodes(sel[:1024], n_steps=200, device="mps") # src.sim.rollout.Episodes

A start k0 is valid when its piece has WINDOW-1 recorded samples before it and n_steps after it,
with no time gap (|dt - 2.5 ms| > 25%) anywhere in that span. Pieces come from
src.data.dataset.frame_pieces (contact-trimmed, split filter applied), so an episode never
crosses a cut, a segment or a flight boundary. Pieces whose median sample time is not 2.5 ms are
skipped (none in the 94 flights used; the 164 Hz flight is excluded upstream).
"""
from __future__ import annotations

import numpy as np
import torch

from src.data.features import build_features
from src.data.targets import build_targets
from src.eval.metrics import flight_of_piece
from src.sim.rollout import DT, WINDOW, Episodes


class EpisodePool:
    def __init__(self, pieces, n_steps: int, window: int = WINDOW):
        self.n_steps, self.window, self.H = int(n_steps), int(window), int(window) - 1
        X, Y, Q, P, starts, flights, owners = [], [], [], [], [], [], []
        self.piece_ids, self.skipped = [], []
        off = 0
        for pid, piece in pieces:
            t = piece["t"].to_numpy()
            n = len(t)
            if n < self.H + self.n_steps + 1:
                continue
            dts = np.diff(t)
            if not np.isclose(np.median(dts), DT, rtol=1e-3):
                self.skipped.append(pid)
                continue
            bad = np.abs(dts - DT) > 0.25 * DT                     # bad[j]: gap between j and j+1
            cb = np.r_[0, np.cumsum(bad)]
            k0 = np.arange(self.H, n - self.n_steps)
            ok = (cb[k0 + self.n_steps] - cb[k0 - self.H]) == 0       # no gap in dts[k0-H : k0+n_steps]
            q = piece[["qx", "qy", "qz", "qw"]].to_numpy()
            X.append(build_features(piece)); Y.append(build_targets(piece))
            Q.append(q / np.linalg.norm(q, axis=1, keepdims=True))
            P.append(piece[["pos_x", "pos_y", "pos_z"]].to_numpy())
            k = len(self.piece_ids)
            self.piece_ids.append(pid)
            starts.append(off + k0[ok]); owners.append(np.full(int(ok.sum()), k))
            flights.append(np.full(int(ok.sum()), flight_of_piece(pid), dtype=object))
            off += n
        if not starts:
            raise ValueError(f"no piece is long enough for {self.n_steps}-step rollouts")
        self.X = np.concatenate(X).astype(np.float32)
        self.Y = np.concatenate(Y).astype(np.float32)
        self.Q = np.concatenate(Q).astype(np.float64)
        self.P = np.concatenate(P).astype(np.float64)
        self.starts = np.concatenate(starts).astype(np.int64)      # global row index of each valid k0
        self.flights = np.concatenate(flights).astype(str)
        self.owner = np.concatenate(owners)
        self._dev = {}

    def __len__(self) -> int:
        return len(self.starts)

    def flight_ids(self) -> list[str]:
        return sorted(set(self.flights.tolist()))

    def select_per_flight(self, n_per_flight: int, exclude: np.ndarray | None = None) -> np.ndarray:
        """Indices into self.starts: n_per_flight starts per flight, evenly spaced over that flight's
        valid starts (all of them if it has fewer). exclude: indices that must not be picked
        (finetune_rollout.py excludes eval_rollout's start points from its model-selection set).
        Deterministic."""
        excl = np.zeros(len(self.starts), bool)
        if exclude is not None:
            excl[np.asarray(exclude)] = True
        out = []
        for f in self.flight_ids():
            idx = np.where((self.flights == f) & ~excl)[0]
            if len(idx) <= n_per_flight:
                out.append(idx)
                continue
            pos = np.floor(np.arange(n_per_flight) * (len(idx) / n_per_flight)).astype(np.int64)
            out.append(idx[pos])
        return np.concatenate(out)

    def _tensors(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._dev:
            self._dev = {key: tuple(torch.as_tensor(a, dtype=dtype, device=device)
                                    for a in (self.X, self.Q, self.P, self.Y))}
        return self._dev[key]

    def episodes(self, sel: np.ndarray, n_steps: int | None = None, device="cpu",
                 dtype=torch.float32, relative_pos: bool = True) -> Episodes:
        """Episodes for the starts self.starts[sel]. relative_pos subtracts the start position
        (errors are unchanged; avoids float32 rounding of absolute positions of several metres)."""
        n = self.n_steps if n_steps is None else int(n_steps)
        if n > self.n_steps:
            raise ValueError(f"pool was built for {self.n_steps} steps, asked for {n}")
        X, Q, P, Y = self._tensors(device, dtype)
        g = torch.as_tensor(self.starts[np.asarray(sel)], device=device)
        idx = g[:, None] + torch.arange(-self.H, n + 1, device=device)[None, :]
        pos = P[idx]
        if relative_pos:
            pos = pos - pos[:, self.H:self.H + 1]
        return Episodes(X[idx], Q[idx], pos, Y[idx], self.H, DT)

    def start_speed(self, sel: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.X[self.starts[np.asarray(sel)], 0:3], axis=1)
