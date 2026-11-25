"""Time one training epoch per model at real data scale, on random data (no real data used).

    python scripts/bench_epoch.py                       # cpu + mps if available, ~10-20 min
    python scripts/bench_epoch.py --devices cpu --quick # 1-minute check

Scale: --n-train rows (default 1.26M ~ 70% of the 1,798,481 NeuroBEM samples), 18 inputs,
6 outputs, 20-sample windows. Times a fixed number of optimiser steps after warm-up and
extrapolates to a full epoch, plus one full pass over a 20% "val" set. Writes
results/timing_bench.json with the machine, torch version and the estimates, including how many
epochs fit in an 8-hour overnight budget for a single model and a 3-member ensemble.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.features import N_FEATURES, N_TARGETS  # noqa: E402
from src.data.windows import WindowedData  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.core import n_params  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODELS = {
    "M2 hybrid MLP": {"type": "mlp", "prior": True, "hidden": [128, 128, 64]},
    "M3 hybrid TCN": {"type": "tcn", "prior": True, "window": 20},
    "M3 ensemble member (NLL)": {"type": "tcn", "prior": True, "window": 20, "nll": True},
    "M4 hybrid transformer": {"type": "transformer", "prior": True, "window": 20},
}


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def fake_segments(n_rows, seg_len, rng):
    segs, left = [], n_rows
    while left > 0:
        n = min(seg_len, left)
        X = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
        X[:, 9:13] = 1150 + 200 * X[:, 9:13]                 # motor speeds in a realistic range
        segs.append((X, rng.normal(size=(n, N_TARGETS)).astype(np.float32)))
        left -= n
    return segs


def bench_one(mcfg, train, val, dev, batch, steps, warmup):
    model = build_model(mcfg)
    Xs, Ys = train.X[:200_000].cpu().numpy(), train.Y[:200_000].cpu().numpy()
    model.fit_normalisers(Xs, Xs, Ys)              # values irrelevant for timing
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    win = model.window if model.window > 1 else 1
    it = train.batches(batch, win, shuffle=True, generator=torch.Generator().manual_seed(0))
    model.train()
    for i in range(warmup + steps):
        if i == warmup:
            sync(dev); t = time.perf_counter()
        xb, yb = next(it)
        loss = model.loss(xb, yb)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    sync(dev)
    per_step = (time.perf_counter() - t) / steps
    model.eval(); sync(dev); t = time.perf_counter()
    with torch.no_grad():
        for xb, yb in val.batches(8192, val.max_window):
            model.loss(xb, yb); model(xb)
    sync(dev)
    val_s = time.perf_counter() - t
    steps_per_epoch = int(np.ceil(len(train) / batch))
    epoch_s = per_step * steps_per_epoch + val_s
    return {"params": n_params(model), "batch_size": batch, "sec_per_step": per_step,
            "steps_per_epoch": steps_per_epoch, "val_pass_s": val_s, "epoch_s": epoch_s,
            "epochs_in_8h_single": int(8 * 3600 // epoch_s),
            "epochs_in_8h_per_member_x3_sequential": int(8 * 3600 // (3 * epoch_s))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="auto", help="comma list: cpu,mps,cuda or auto")
    ap.add_argument("--n-train", type=int, default=1_260_000)
    ap.add_argument("--batches", default="1024,4096")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="tiny run to check the script works")
    ap.add_argument("--out", default=str(REPO / "results" / "timing_bench.json"))
    a = ap.parse_args(argv)
    if a.quick:
        a.n_train, a.steps, a.warmup = 60_000, 5, 2
    if a.devices == "auto":
        devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) \
            + (["cuda"] if torch.cuda.is_available() else [])
    else:
        devs = a.devices.split(",")
    batches = [int(b) for b in a.batches.split(",")]
    rng = np.random.default_rng(0)
    seg_len = 7300                                     # ~ mean NeuroBEM segment (1.8M rows / 247)
    train_segs = fake_segments(a.n_train, seg_len, rng)
    val_segs = fake_segments(int(a.n_train * 0.2 / 0.7), seg_len, rng)
    out = {"note": "random data at real scale; timing only, no real data used",
           "machine": {"platform": platform.platform(), "processor": platform.processor(),
                       "python": platform.python_version(), "torch": torch.__version__,
                       "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads()},
           "n_train_rows": a.n_train, "steps_timed": a.steps, "results": []}
    for d in devs:
        dev = torch.device(d)
        train, val = WindowedData(train_segs, 20, dev), WindowedData(val_segs, 20, dev)
        for name, mcfg in MODELS.items():
            for b in batches:
                try:
                    r = bench_one(mcfg, train, val, dev, b, a.steps, a.warmup)
                    r.update(model=name, device=d)
                    print(f"{d:4s} {name:28s} batch {b:5d}: {r['epoch_s']/60:6.1f} min/epoch, "
                          f"{r['epochs_in_8h_single']:4d} epochs in 8 h  ({r['params']} params)", flush=True)
                except Exception as e:                  # e.g. an op missing on MPS
                    r = {"model": name, "device": d, "batch_size": b, "error": repr(e)}
                    print(f"{d:4s} {name:28s} batch {b:5d}: ERROR {e!r}", flush=True)
                out["results"].append(r)
        del train, val
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
