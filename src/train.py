"""Train one model from a YAML config; log one row to results/runs.csv.

    python -m src.train --config configs/m3_hybrid_tcn.yaml
    python -m src.train --config configs/m3_hybrid_tcn.yaml --seed 1 --resume
    python -m src.train --config configs/m2_hybrid_mlp.yaml --set train.lr=3e-4 model.hidden=[64,64]

- Loads ONLY the train and val parts of a split (the test part is refused here: test runs
  once, at CP7, through scripts/evaluate_test.py).
- Prior + normalisers fitted on train only. Early stopping on val loss.
- Checkpoints every epoch to checkpoints/<run>/last.pt (+ best.pt); --resume continues a run
  that was cut off (Colab disconnect, laptop sleep). train.max_hours caps wall time.
- runs.csv row: git hash + dirty flag, full config, params, epochs, best val loss, val RMSEs,
  wall time, stop reason. Reported numbers come from the CP3 evaluation scripts, not this row.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.windows import WindowedData
from src.models import build_model
from src.models.core import n_params

REPO = Path(__file__).resolve().parents[1]
RUNS_FIELDS = ["timestamp_utc", "run", "name", "seed", "git_hash", "git_dirty", "model", "prior", "nll",
               "params", "device", "n_train", "n_val", "epochs_run", "best_epoch", "best_val_loss",
               "val_rmse_F_N", "val_rmse_M_Nm", "wall_time_s", "stop_reason", "config_json"]

DEFAULTS = {
    "seed": 0,
    "model": {"type": "mlp", "prior": False, "nll": False, "window": 1},
    "data": {"source": "synthetic", "split": "A", "max_window": 20},
    "train": {"batch_size": 1024, "lr": 1e-3, "weight_decay": 0.0, "max_epochs": 100, "patience": 8,
              "min_delta": 1e-4, "lr_patience": 3, "lr_factor": 0.5, "grad_clip": 1.0,
              "max_hours": None, "device": "auto", "eval_batch_size": 8192},
    "log": {"runs_csv": "results/runs.csv", "ckpt_dir": "checkpoints"},
}


# ---------------------------------------------------------------- config helpers
def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | None = None, **direct) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        cfg = deep_update(cfg, yaml.safe_load(Path(path).read_text()))
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        d = cfg
        *parents, last = key.split(".")
        for p in parents:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(val)
    return deep_update(cfg, direct)


def git_info() -> tuple[str, bool]:
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True,
                                    stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                             cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip())
        return h, dirty
    except Exception:
        return "unknown", True


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# ---------------------------------------------------------------- data
def load_split(dcfg: dict, part: str):
    """List of (X, Y) segments for one part ('train' or 'val') of a split.

    NeuroBEM: data/splits.json (scripts/make_splits.py) maps split keys to segment-id lists:
        {"A": {"train": [...], "val": [...], "test": [...]}, "A_r1".."A_r3", "B", "B_s"}
    Each segment is contact-trimmed into pieces (windows never cross a cut).
    data.resample = k > 0 selects key "A_rk".
    """
    if part not in ("train", "val"):
        raise PermissionError("train.py never loads test data (test is evaluated once, at CP7)")
    if dcfg["source"] == "synthetic":
        from src.data.synthetic import make_segments
        n = dcfg.get("n_segments", 10)
        segs = make_segments(n, dcfg.get("segment_length", 2000), seed=dcfg.get("data_seed", 123))
        n_val = max(1, n // 5)
        return segs[n_val:] if part == "train" else segs[:n_val]
    if dcfg["source"] == "neurobem":
        # contact-trimmed pieces + any split sample filter (src/data/dataset.py, CP2)
        from src.data.dataset import load_part, read_splits
        splits_path = REPO / dcfg.get("splits_file", "data/splits.json")
        key = dcfg["split"] if dcfg.get("resample") in (None, 0) else f"{dcfg['split']}_r{dcfg['resample']}"
        return [(X, Y) for _, X, Y in load_part(key, part, read_splits(splits_path))]
    raise ValueError(f"unknown data.source {dcfg['source']!r}")


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, data: WindowedData, batch_size: int) -> dict:
    model.eval()
    tot, n, se = 0.0, 0, torch.zeros(6, dtype=torch.float64)
    for xb, yb in data.batches(batch_size, data.max_window):
        tot += model.loss(xb, yb).item() * len(xb)
        mean, _ = model(xb)
        se += ((mean - yb) ** 2).double().sum(0).cpu()
        n += len(xb)
    mse = se / n
    return {"loss": tot / n,
            # RMSE of the 3-vector error: sqrt(mean ||e||^2). CP3 metrics.py is the reporting authority.
            "rmse_F": float(math.sqrt(mse[:3].sum())), "rmse_M": float(math.sqrt(mse[3:].sum())),
            "rmse_per_axis": [float(math.sqrt(v)) for v in mse]}


# ---------------------------------------------------------------- main loop
def run(cfg: dict, resume: bool = False, verbose: bool = True) -> dict:
    t0 = time.time()
    seed = int(cfg["seed"])
    set_seed(seed)
    tc, mc, dc, lc = cfg["train"], cfg["model"], cfg["data"], cfg["log"]
    run_name = f"{cfg.get('name', mc['type'])}_seed{seed}"
    ckpt_dir = Path(lc["ckpt_dir"]) / run_name
    if not ckpt_dir.is_absolute():
        ckpt_dir = REPO / ckpt_dir
    device = pick_device(tc["device"])

    train_segs, val_segs = load_split(dc, "train"), load_split(dc, "val")
    W = int(dc["max_window"])
    if int(mc.get("window", 1)) > W:
        raise ValueError("model.window exceeds data.max_window")
    train = WindowedData(train_segs, W, device)
    val = WindowedData(val_segs, W, device)

    model = build_model(mc)
    X_rows = np.concatenate([s[0] for s in train_segs])
    X_last, Y_last = train.last_rows()
    fit_info = model.fit_normalisers(X_rows, X_last, Y_last)
    model.to(device)
    params = n_params(model)
    trainable = params > 0
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"]) if trainable else None
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=tc["lr_factor"], patience=tc["lr_patience"]) \
        if trainable else None
    gen = torch.Generator().manual_seed(seed)
    state = {"epoch": 0, "best": float("inf"), "best_epoch": 0, "bad": 0, "history": [], "wall_s": 0.0}

    last_pt = ckpt_dir / "last.pt"
    if resume and last_pt.exists():
        ck = torch.load(last_pt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if opt:
            opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"]); torch.set_rng_state(ck["torch_rng"])
        state = ck["state"]
        if verbose:
            print(f"resumed {run_name} at epoch {state['epoch']} (best {state['best']:.4f})")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wall_before = state.get("wall_s", 0.0)       # time spent in earlier (cut-off) sessions

    stop = "max_epochs"
    max_epochs = int(tc["max_epochs"]) if trainable else 0
    while state["epoch"] < max_epochs:
        ep_t = time.time()
        model.train()
        tr_loss, tr_n = 0.0, 0
        for xb, yb in train.batches(tc["batch_size"], model.window if model.window > 1 else 1,
                                    shuffle=True, generator=gen):
            loss = model.loss(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tc["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            tr_loss += loss.item() * len(xb); tr_n += len(xb)
        ev = evaluate(model, val, tc["eval_batch_size"])
        sched.step(ev["loss"])
        state["epoch"] += 1
        state["history"].append({"epoch": state["epoch"], "train_loss": tr_loss / tr_n, "val_loss": ev["loss"],
                                 "val_rmse_F": ev["rmse_F"], "val_rmse_M": ev["rmse_M"],
                                 "lr": opt.param_groups[0]["lr"], "sec": time.time() - ep_t})
        state["wall_s"] = wall_before + (time.time() - t0)
        improved = ev["loss"] < state["best"] - tc["min_delta"]
        if improved:
            state.update(best=ev["loss"], best_epoch=state["epoch"], bad=0)
            torch.save({"model": model.state_dict(), "config": cfg}, ckpt_dir / "best.pt")
        else:
            state["bad"] += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "gen": gen.get_state(), "torch_rng": torch.get_rng_state(), "state": state,
                    "config": cfg}, last_pt)
        if verbose:
            h = state["history"][-1]
            print(f"[{run_name}] ep {h['epoch']:3d} train {h['train_loss']:.4f} val {h['val_loss']:.4f} "
                  f"F {h['val_rmse_F']:.3f} N  M {h['val_rmse_M']:.4f} N·m  {h['sec']:.1f}s"
                  + ("  *" if improved else ""))
        if state["bad"] >= tc["patience"]:
            stop = "early_stop"; break
        if tc.get("max_hours") and state["wall_s"] / 3600 > float(tc["max_hours"]):
            stop = "time_budget"; break

    if trainable and (ckpt_dir / "best.pt").exists():
        model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)["model"])
    else:                                     # constant / physics: nothing to train
        torch.save({"model": model.state_dict(), "config": cfg}, ckpt_dir / "best.pt")
        stop = "no_trainable_params"
    final = evaluate(model, val, tc["eval_batch_size"])
    wall = wall_before + (time.time() - t0)
    gh, dirty = git_info()
    row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run": run_name,
           "name": cfg.get("name", ""), "seed": seed, "git_hash": gh, "git_dirty": dirty, "model": mc["type"],
           "prior": bool(mc.get("prior", False)) or mc["type"] == "physics", "nll": bool(mc.get("nll", False)),
           "params": params, "device": str(device), "n_train": len(train), "n_val": len(val),
           "epochs_run": state["epoch"], "best_epoch": state["best_epoch"],
           "best_val_loss": round(final["loss"], 6), "val_rmse_F_N": round(final["rmse_F"], 5),
           "val_rmse_M_Nm": round(final["rmse_M"], 6), "wall_time_s": round(wall, 1), "stop_reason": stop,
           "config_json": json.dumps(cfg, sort_keys=True)}
    runs_csv = Path(lc["runs_csv"])
    if not runs_csv.is_absolute():
        runs_csv = REPO / runs_csv
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not runs_csv.exists()
    with runs_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUNS_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    (ckpt_dir / "history.json").write_text(json.dumps({"fit_info": fit_info, **state}, indent=1, default=str))
    if verbose:
        print(f"[{run_name}] done: {stop}, best epoch {state['best_epoch']}, val F {final['rmse_F']:.4f} N, "
              f"M {final['rmse_M']:.5f} N·m, {wall:.0f}s -> {runs_csv.name}")
    return {"row": row, "final": final, "fit_info": fit_info, "model": model, "ckpt_dir": ckpt_dir}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.lr=3e-4")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    if a.seed is not None:
        cfg["seed"] = a.seed
    run(cfg, resume=a.resume)


if __name__ == "__main__":
    main()
