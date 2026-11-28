"""CP6 stretch (H5): fine-tune M2 with a multi-step rollout loss.

    python scripts/finetune_rollout.py --seed 0                       # from checkpoints/m2_hybrid_mlp_tuned_seed0
    python scripts/finetune_rollout.py --seed 0 --tbptt 5             # truncated BPTT (detach every 5 steps)
    python scripts/finetune_rollout.py --seed 0 --max-epochs 1 --batches-per-epoch 5   # smoke run
    -> checkpoints/m2_hybrid_mlp_rollft_seed<k>/{best,last}.pt, history.json
    -> results/finetune_rollout_seed<k>.json   per-epoch log (rollout + one-step metrics)
    -> one row in results/runs.csv (val_rmse_* = one-step validation, same definition as train.py)

Method
  - Initial weights: the M2 checkpoint (--init, default m2_hybrid_mlp_tuned_seed<seed>). The P0b
    prior and the input/output normalisers stay frozen (buffers fitted on A/train); only the MLP
    residual is trained.
  - Training episodes: random start points on A/train (src/sim/pool.py: 19 recorded samples before,
    --n-steps after, no gap, one contact-trimmed piece). Each start is rolled out --n-steps (20 =
    50 ms) with semi-implicit Euler in --mode (translational: recorded attitude and rates, so the
    loss only sees the force prediction).
  - Loss = L_roll + lambda * L_1step
        L_roll  = mean over steps 1..N and the batch of |dp|^2 / s_p^2 + |dv_world|^2 / s_v^2
                  (s_p, s_v = RMS position / velocity error of the INITIAL model on a fixed train batch,
                   so both terms start near 1)
        L_1step = the original standardised one-step loss (ForceModel.loss) on random train windows;
                  keeps the torque head (unseen by a translational rollout) and one-step accuracy in place.
    Gradients: full BPTT through the N steps, or truncated (--tbptt k: state detached every k steps).
    AdamW (weight decay from the M2 config), gradient-norm clipping (--grad-clip), non-finite
    batches skipped and counted.
  - Model selection: after every epoch, median position error at --select-horizon (0.25 s) of RK4
    translational rollouts on A/val start points (--val-starts per flight), chosen to EXCLUDE the
    start points scripts/eval_rollout_horizons.py reports on (same flights, different starts). Best epoch
    kept (epoch 0 = the unmodified M2 counts, so the fine-tune can only be kept if it helps);
    early stop after --patience epochs without improvement.
Validation only; test data never loaded.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import frame_pieces, part_filter, piece_arrays, read_splits, segment_ids  # noqa: E402
from src.data.load import REPO, load_all  # noqa: E402
from src.data.windows import WindowedData  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.core import n_params  # noqa: E402
from src.sim import rollout as R  # noqa: E402
from src.sim.pool import EpisodePool  # noqa: E402
from src.train import RUNS_FIELDS, evaluate, git_info, pick_device, set_seed  # noqa: E402

EVAL_STARTS_PER_FLIGHT = 200          # scripts/eval_rollout_horizons.py default; excluded from selection
EVAL_MAX_STEPS = 200                  # 0.5 s, the longest eval_rollout horizon


def rollout_errors(out: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared position and world-velocity errors (B, N) for steps 1..N (step 0 is the true state)."""
    dp = out["pos"][:, 1:] - out["pos_true"][:, 1:]
    vw = (R.rot(out["q"][:, 1:]) @ out["vel"][:, 1:].unsqueeze(-1)).squeeze(-1)
    vw_true = (R.rot(out["q_true"][:, 1:]) @ out["vel_true"][:, 1:].unsqueeze(-1)).squeeze(-1)
    return (dp ** 2).sum(-1), ((vw - vw_true) ** 2).sum(-1)


@torch.no_grad()
def val_rollout_median(model, pool, sel, n_steps, mode, device, batch=2048) -> dict:
    model.eval()
    errs = []
    for a in range(0, len(sel), batch):
        ep = pool.episodes(sel[a:a + batch], n_steps, device=device)
        o = R.rollout(R.ModelForce(model), ep, n_steps, "rk4", mode)
        errs.append(o["pos_err"][:, -1].cpu().double().numpy())
    e = np.concatenate(errs)
    e = np.where(np.isfinite(e), e, np.inf)
    return {"median": float(np.median(e)), "p25": float(np.percentile(e, 25)), "p75": float(np.percentile(e, 75)),
            "n": int(len(e)), "n_nonfinite": int(np.isinf(e).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="checkpoint dir name (default m2_hybrid_mlp_tuned_seed<seed>)")
    ap.add_argument("--name", default="m2_hybrid_mlp_rollft")
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--tbptt", type=int, default=0, help="detach every k steps (0 = full BPTT)")
    ap.add_argument("--mode", default="translational", choices=["translational", "full"])
    ap.add_argument("--batch", type=int, default=512, help="rollout episodes per step")
    ap.add_argument("--batch-1step", type=int, default=1024)
    ap.add_argument("--lambda-1step", type=float, default=1.0)
    ap.add_argument("--batches-per-epoch", type=int, default=200)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-delta", type=float, default=1e-5, help="m, on the selection median")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=None, help="default: the M2 config's")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-starts", type=int, default=50, help="selection start points per val flight")
    ap.add_argument("--select-horizon", type=float, default=0.25)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--split", default="A")
    a = ap.parse_args(argv)
    t0 = time.time()
    set_seed(a.seed)
    device = pick_device(a.device)
    init = a.init or f"m2_hybrid_mlp_tuned_seed{a.seed}"
    ck = torch.load(REPO / "checkpoints" / init / "best.pt", map_location=device, weights_only=False)
    cfg0 = ck["config"]
    if cfg0["data"]["split"] != a.split:
        raise ValueError(f"{init} was trained on split {cfg0['data']['split']}")
    model = build_model(cfg0["model"]); model.load_state_dict(ck["model"]); model.to(device)
    wd = float(cfg0["train"]["weight_decay"]) if a.weight_decay is None else a.weight_decay
    sel_steps = int(round(a.select_horizon / R.DT))

    # ---- data (train + val only)
    splits = read_splits()
    df = load_all()

    def pieces(part):
        ids = set(segment_ids(a.split, part, splits))
        return frame_pieces(df[df["segment_id"].astype(str).isin(ids)], part_filter(a.split, part, splits))
    tr_pieces, va_pieces = pieces("train"), pieces("val")
    tr_pool = EpisodePool(tr_pieces, a.n_steps)
    va_pool = EpisodePool(va_pieces, max(EVAL_MAX_STEPS, sel_steps))
    va_sel = va_pool.select_per_flight(a.val_starts, exclude=va_pool.select_per_flight(EVAL_STARTS_PER_FLIGHT))
    W = int(cfg0["data"]["max_window"])
    tr_1 = WindowedData([(X, Y) for _, X, Y in piece_arrays(tr_pieces)], W, device)
    va_1 = WindowedData([(X, Y) for _, X, Y in piece_arrays(va_pieces)], W, device)
    print(f"{init} -> {a.name}_seed{a.seed} on {device}; train starts {len(tr_pool)}, "
          f"selection starts {len(va_sel)} ({len(va_pool.flight_ids())} val flights, eval starts excluded)")

    # ---- loss scales from the initial model on a fixed train batch
    gen = np.random.default_rng(a.seed)
    with torch.no_grad():
        model.eval()
        ep = tr_pool.episodes(gen.integers(0, len(tr_pool), 4 * a.batch), a.n_steps, device=device)
        p2, v2 = rollout_errors(R.rollout(R.ModelForce(model), ep, a.n_steps, "euler", a.mode))
        s_p, s_v = float(p2.mean().sqrt()), float(v2.mean().sqrt())
    print(f"loss scales (initial model, {a.n_steps}-step Euler): s_p {s_p * 100:.3f} cm, s_v {s_v:.4f} m/s")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=wd)
    tgen = torch.Generator().manual_seed(a.seed)
    run = f"{a.name}_seed{a.seed}"
    ckpt_dir = REPO / "checkpoints" / run
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg0)
    cfg["name"] = a.name
    cfg["seed"] = a.seed
    cfg["finetune"] = {"init": init, "n_steps": a.n_steps, "integrator": "euler", "mode": a.mode, "tbptt": a.tbptt,
                       "batch": a.batch, "batch_1step": a.batch_1step, "lambda_1step": a.lambda_1step,
                       "batches_per_epoch": a.batches_per_epoch, "max_epochs": a.max_epochs, "patience": a.patience,
                       "lr": a.lr, "weight_decay": wd, "grad_clip": a.grad_clip, "s_p": s_p, "s_v": s_v,
                       "select": f"median pos err at {a.select_horizon} s, RK4 {a.mode}, {len(va_sel)} A/val starts"}

    def epoch_eval(ep_no, extra):
        r = val_rollout_median(model, va_pool, va_sel, sel_steps, a.mode, device)
        one = evaluate(model, va_1, int(cfg0["train"]["eval_batch_size"]))
        h = {"epoch": ep_no, "sel_median_m": r["median"], "sel_p25_m": r["p25"], "sel_p75_m": r["p75"],
             "sel_nonfinite": r["n_nonfinite"], "val_1step_loss": one["loss"], "val_rmse_F_vecnorm": one["rmse_F"],
             "val_rmse_M_vecnorm": one["rmse_M"], **extra}
        print(f"[{run}] ep {ep_no:2d}  sel {a.select_horizon:g}s median {r['median'] * 100:.3f} cm "
              f"[{r['p25'] * 100:.3f}, {r['p75'] * 100:.3f}]  1-step F {one['rmse_F']:.4f} N M {one['rmse_M']:.5f} (vecnorm)"
              + "".join(f"  {k} {v:.4g}" for k, v in extra.items() if isinstance(v, float)))
        return h

    hist = [epoch_eval(0, {"train_loss": float("nan"), "secs": 0.0})]
    best, best_epoch, bad = hist[0]["sel_median_m"], 0, 0
    torch.save({"model": model.state_dict(), "config": cfg}, ckpt_dir / "best.pt")
    n_skipped = 0
    stop = "max_epochs"
    for epoch in range(1, a.max_epochs + 1):
        te = time.time()
        model.train()
        tl, tr_roll, tr_one, nb = 0.0, 0.0, 0.0, 0
        for _ in range(a.batches_per_epoch):
            sel = gen.integers(0, len(tr_pool), a.batch)
            ep = tr_pool.episodes(sel, a.n_steps, device=device)
            out = R.rollout(R.ModelForce(model), ep, a.n_steps, "euler", a.mode,
                            detach_every=(a.tbptt or None))
            p2, v2 = rollout_errors(out)
            l_roll = p2.mean() / s_p ** 2 + v2.mean() / s_v ** 2
            ends = tr_1.ends[torch.randint(0, len(tr_1), (a.batch_1step,), generator=tgen).to(tr_1.ends.device)]
            xb, yb = tr_1.gather(ends, model.window if model.window > 1 else 1)
            l_one = model.loss(xb, yb)
            loss = l_roll + a.lambda_1step * l_one
            if not torch.isfinite(loss):
                n_skipped += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if a.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()
            tl += loss.item(); tr_roll += l_roll.item(); tr_one += l_one.item(); nb += 1
        h = epoch_eval(epoch, {"train_loss": tl / max(nb, 1), "train_l_roll": tr_roll / max(nb, 1),
                               "train_l_1step": tr_one / max(nb, 1), "skipped_batches": n_skipped,
                               "secs": time.time() - te})
        hist.append(h)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "config": cfg, "history": hist},
                   ckpt_dir / "last.pt")
        if h["sel_median_m"] < best - a.min_delta:
            best, best_epoch, bad = h["sel_median_m"], epoch, 0
            torch.save({"model": model.state_dict(), "config": cfg}, ckpt_dir / "best.pt")
            print(f"   * new best")
        else:
            bad += 1
            if bad >= a.patience:
                stop = "early_stop"
                break

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)["model"])
    final = evaluate(model, va_1, int(cfg0["train"]["eval_batch_size"]))
    wall = time.time() - t0
    gh, dirty = git_info()
    row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run": run, "name": a.name,
           "seed": a.seed, "git_hash": gh, "git_dirty": dirty, "model": cfg["model"]["type"], "prior": True,
           "nll": bool(cfg["model"].get("nll", False)), "params": n_params(model), "device": str(device),
           "n_train": len(tr_pool), "n_val": len(va_sel), "epochs_run": len(hist) - 1, "best_epoch": best_epoch,
           "best_val_loss": round(final["loss"], 6), "val_rmse_F_N": round(final["rmse_F"], 5),
           "val_rmse_M_Nm": round(final["rmse_M"], 6), "wall_time_s": round(wall, 1), "stop_reason": stop,
           "config_json": json.dumps(cfg, sort_keys=True, default=str)}
    runs_csv = REPO / "results" / "runs.csv"
    new = not runs_csv.exists()
    with runs_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUNS_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    rep = {"source": "scripts/finetune_rollout.py", "run": run, "init": init, "git_hash": gh, "git_dirty": dirty,
           "splits_sha256": splits["meta"]["sha256"], "device": str(device), "config": cfg["finetune"],
           "n_train_starts": len(tr_pool), "n_select_starts": int(len(va_sel)), "best_epoch": best_epoch,
           "best_sel_median_m": best, "epoch0_sel_median_m": hist[0]["sel_median_m"], "stop_reason": stop,
           "skipped_batches": n_skipped, "final_val_1step": final, "history": hist, "wall_time_s": round(wall, 1),
           "note": "n_val in runs.csv = number of selection start points; best_val_loss / val_rmse_* = one-step "
                   "validation of the kept checkpoint (vector-norm RMSE, as train.py)."}
    (ckpt_dir / "history.json").write_text(json.dumps(rep, indent=1, default=str))
    out = REPO / "results" / f"finetune_rollout_seed{a.seed}.json"
    out.write_text(json.dumps(rep, indent=1, default=str))
    print(f"[{run}] done: {stop}, best epoch {best_epoch} (sel median {best * 100:.3f} cm vs epoch 0 "
          f"{hist[0]['sel_median_m'] * 100:.3f} cm), {wall:.0f} s -> {out.relative_to(REPO)}, runs.csv")


if __name__ == "__main__":
    main()
