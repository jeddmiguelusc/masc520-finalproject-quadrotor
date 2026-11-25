"""Segment-safe sliding windows, gathered on the fly (no copies of the data per window).

Every model is trained and scored on the SAME set of samples: those with at least
`max_window - 1` earlier samples in their own segment. A window never crosses a segment
boundary, so it never mixes two flights or bridges a Vicon dropout.
"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
import torch


class WindowedData:
    def __init__(self, segments: Sequence[tuple[np.ndarray, np.ndarray]], max_window: int = 20,
                 device: str | torch.device = "cpu"):
        xs, ys, ends, seg_of_end = [], [], [], []
        offset = 0
        for k, (X, Y) in enumerate(segments):
            n = len(X)
            if n >= max_window:
                e = np.arange(max_window - 1, n) + offset
                ends.append(e)
                seg_of_end.append(np.full(len(e), k))
            xs.append(np.asarray(X, np.float32)); ys.append(np.asarray(Y, np.float32))
            offset += n
        if not ends:
            raise ValueError("no segment is as long as max_window")
        self.max_window = max_window
        self.device = torch.device(device)
        self.X = torch.from_numpy(np.concatenate(xs)).to(self.device)
        self.Y = torch.from_numpy(np.concatenate(ys)).to(self.device)
        self.ends = torch.from_numpy(np.concatenate(ends)).long().to(self.device)
        self.segment_index = np.concatenate(seg_of_end)          # which segment each sample is in

    def __len__(self) -> int:
        return len(self.ends)

    def gather(self, end_idx: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B, window, F) ending at each end index (inclusive), y (B, 6) at the end index."""
        if window > self.max_window:
            raise ValueError(f"window {window} > max_window {self.max_window}")
        offs = torch.arange(-window + 1, 1, device=self.device)
        idx = end_idx[:, None] + offs[None, :]
        return self.X[idx], self.Y[end_idx]

    def batches(self, batch_size: int, window: int, shuffle: bool = False,
                generator: torch.Generator | None = None) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        n = len(self.ends)
        order = torch.randperm(n, generator=generator).to(self.device) if shuffle \
            else torch.arange(n, device=self.device)
        for i in range(0, n, batch_size):
            yield self.gather(self.ends[order[i:i + batch_size]], window)

    def last_rows(self) -> tuple[np.ndarray, np.ndarray]:
        """Inputs and targets at every scored sample (for fitting priors / normalisers)."""
        e = self.ends.cpu()
        return self.X.cpu()[e].numpy(), self.Y.cpu()[e].numpy()
