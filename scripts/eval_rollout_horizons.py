"""CP5/CP6: rollout evaluation at 0.1 / 0.25 / 0.5 s with >= 200 start points per flight, median + IQR.

Complements scripts/eval_rollout.py (stride-200 starts, 0.1 / 0.25 s, RMS; both modes and integrators;
-> results/val_rollout.json). Translational mode first.

    python scripts/eval_rollout_horizons.py                                  # default model list (below)
    python scripts/eval_rollout_horizons.py --runs m2_hybrid_mlp_tuned_seed0 --no-ensembles --n-starts 50   # quick
    python scripts/eval_rollout_horizons.py --mode full --out results/rollout_horizons_val_full.json
    -> results/rollout_horizons_val.json         summaries, per-flight medians, seed summary, paired diffs
    -> results/rollout_horizons_val_errors.npz   per-start position / velocity errors (for figures)

Protocol
  - Validation flights of split A only (18 flights). >= --n-starts (200) start points per flight,
    evenly spaced over every valid start of that flight (src/sim/pool.py): 19 recorded samples
    before, 0.5 s (200 steps) after, no time gap, inside one contact-trimmed piece. The same start
    points are used for every model, so all comparisons are paired.
  - Each rollout starts on the true state and is driven by the recorded motor speeds, dmot, vbat
    (src/sim/rollout.py). Translational mode: attitude and body rates come from the data, only
    p and v are integrated, so only the force prediction matters. RK4 by default.
  - Position error |p_sim - p_true| [m] at 0.1, 0.25, 0.5 s (40, 100, 200 steps); also world-frame
    velocity error. Summaries: median and IQR (p25, p75) over all start points, plus mean, p90,
    a 95% flight-bootstrap CI of the median (2000 resamples, seed 0) and per-flight medians.
    Subsets by speed at the start: all, |v| <= 5, |v| > 5 m/s.
  - Reference rows: "oracle" (recorded F and M: integrator + filter floor) and P0b.
  - Paired diffs (a - b, seed-matched like scripts/eval_models.py): difference of medians with a
    paired flight bootstrap (same flights, same starts), p = share of resamples with a not better,
    and win = share of start points where a has the smaller error.
Validation only (test data never loaded). Model inputs and checkpoints as in eval_models.py.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import frame_pieces, part_filter, read_splits, segment_ids  # noqa: E402
from src.data.load import REPO, load_all  # noqa: E402
from src.models import build_model  # noqa: E402
from src.sim import rollout as R  # noqa: E402
from src.sim.pool import EpisodePool  # noqa: E402
from src.train import pick_device  # noqa: E402

HORIZONS_S = (0.1, 0.25, 0.5)
SEEDS = (0, 1, 2)
DEFAULT_RUNS = ([f"m1_mlp_tuned_seed{s}" for s in SEEDS] + [f"m2_hybrid_mlp_tuned_seed{s}" for s in SEEDS]
                + [f"m3_hybrid_tcn_seed{s}" for s in SEEDS] + [f"m3_hybrid_tcn_tuned_seed{s}" for s in SEEDS]
                + [f"m3_hybrid_tcn_act_tuned_seed{s}" for s in SEEDS]
                + [f"m3_hybrid_tcn_act_seed{s}" for s in SEEDS]
                + [f"m4_hybrid_transformer_seed{s}" for s in SEEDS]
                + [f"m2_hybrid_mlp_rollft_seed{s}" for s in SEEDS])
DEFAULT_ENSEMBLES = {"M3ens": [f"m3_ens_seed{s}" for s in SEEDS],
                     "M3ens_tuned": [f"m3_ens_tuned_seed{s}" for s in SEEDS]}
LABEL = {"m1_mlp": "M1", "m1_mlp_tuned": "M1", "m2_hybrid_mlp": "M2", "m2_hybrid_mlp_tuned": "M2",
         "m3_hybrid_tcn": "M3", "m3_hybrid_tcn_tuned": "M3tuned", "m3_hybrid_tcn_act_tuned": "M3act",
         "m3_hybrid_tcn_act": "M3act_default", "m4_hybrid_transformer": "M4", "m2_hybrid_mlp_rollft": "M2roll",
         "m3_ens": "M3nll", "m3_ens_tuned": "M3nll_tuned"}
# a - b; negative = a better
PAIRS = [("M2", "M1"), ("M2", "P0b"), ("M3", "M2"), ("M3tuned", "M2"), ("M3act", "M2"), ("M3act", "M3"),
         ("M3act_default", "M2"), ("M3act_default", "M3"), ("M4", "M3"), ("M3ens", "M3"), ("M3ens", "M2"), ("M2roll", "M2"), ("M3", "P0b")]
SUBSETS = {"all": (-np.inf, np.inf), "le5": (-np.inf, 5.0), "gt5": (5.0, np.inf)}


# ---------------------------------------------------------------- force sources
class MeanOf(torch.nn.Module):
    """Ensemble force = mean of the members' means (the mixture mean used in eval_models.py)."""

    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    def forward(self, x):
        return torch.stack([m(x)[0] for m in self.members]).mean(0), None


def load_run(run: str, device):
    ck = torch.load(REPO / "checkpoints" / run / "best.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    m = build_model(cfg["model"]); m.load_state_dict(ck["model"])
    return m.to(device).eval(), cfg


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- rollouts
@torch.no_grad()
def run_rollouts(force_fn, pool: EpisodePool, sel: np.ndarray, horizons: list[int], mode: str,
                 integrator: str, device, batch: int) -> dict:
    """Position / velocity error (n_sel, len(horizons)) for one force source."""
    n = max(horizons)
    pe, ve = [], []
    for a in range(0, len(sel), batch):
        ep = pool.episodes(sel[a:a + batch], n, device=device)
        f = force_fn(ep) if callable(force_fn) and getattr(force_fn, "needs_episode", False) else force_fn
        o = R.rollout(f, ep, n, integrator, mode)
        pe.append(o["pos_err"][:, horizons].cpu().double().numpy())
        ve.append(o["vel_err"][:, horizons].cpu().double().numpy())
    return {"pos": np.concatenate(pe), "vel": np.concatenate(ve)}


class _Oracle:
    needs_episode = True

    def __call__(self, ep):
        return R.RecordedForce(ep)


# ---------------------------------------------------------------- statistics
def _flight_index(flights: np.ndarray):
    fl = np.unique(flights)
    return fl, [np.where(flights == f)[0] for f in fl]


def _boot_idx(n_flights: int, b: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_flights, size=(b, n_flights))


def summarize(err: np.ndarray, flights: np.ndarray, boot: np.ndarray) -> dict:
    """err (n,) for one horizon."""
    fl, groups = _flight_index(flights)
    finite = np.isfinite(err)
    e = np.where(finite, err, np.inf)                           # a diverged rollout counts as worst
    reps = np.array([np.median(e[np.concatenate([groups[k] for k in row])]) for row in boot])
    return {"n": int(len(e)), "n_nonfinite": int((~finite).sum()),
            "median": float(np.median(e)), "p25": float(np.percentile(e, 25)), "p75": float(np.percentile(e, 75)),
            "iqr": float(np.percentile(e, 75) - np.percentile(e, 25)),
            "mean": float(np.mean(err[finite])) if finite.any() else None, "p90": float(np.percentile(e, 90)),
            "median_ci95": [float(x) for x in np.percentile(reps, [2.5, 97.5])],
            "per_flight_median": {str(f): float(np.median(e[g])) for f, g in zip(fl, groups)}}


def paired_median_diff(a: np.ndarray, b: np.ndarray, flights: np.ndarray, boot: np.ndarray) -> dict:
    a = np.where(np.isfinite(a), a, np.inf); b = np.where(np.isfinite(b), b, np.inf)
    _, groups = _flight_index(flights)
    d0 = float(np.median(a) - np.median(b))
    reps = []
    for row in boot:
        ix = np.concatenate([groups[k] for k in row])
        reps.append(np.median(a[ix]) - np.median(b[ix]))
    reps = np.array(reps)
    return {"diff_median": d0, "ci95": [float(x) for x in np.percentile(reps, [2.5, 97.5])],
            "p": float((reps >= 0).mean()), "win_rate_a": float((a < b).mean())}


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="checkpoint names (default: list in the script)")
    ap.add_argument("--ensemble", nargs="*", default=None, help="label=run1,run2,run3")
    ap.add_argument("--no-ensembles", action="store_true")
    ap.add_argument("--no-baselines", action="store_true", help="skip oracle and P0b")
    ap.add_argument("--n-starts", type=int, default=200, help="start points per validation flight")
    ap.add_argument("--horizons", type=float, nargs="*", default=list(HORIZONS_S))
    ap.add_argument("--mode", default="translational", choices=["translational", "full"])
    ap.add_argument("--integrator", default="rk4", choices=["rk4", "euler"])
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--split", default="A")
    ap.add_argument("--out", default="results/rollout_horizons_val.json")
    ap.add_argument("--errors-out", default="results/rollout_horizons_val_errors.npz")
    a = ap.parse_args(argv)
    t0 = time.time()
    device = pick_device(a.device)
    steps = [int(round(h / R.DT)) for h in a.horizons]
    n_max = max(steps)

    runs = a.runs if a.runs is not None else DEFAULT_RUNS
    if a.no_ensembles:
        ensembles = {}
    elif a.ensemble is not None:
        ensembles = {k: v.split(",") for k, v in (e.split("=", 1) for e in a.ensemble)}
    else:
        ensembles = dict(DEFAULT_ENSEMBLES)
    have = lambda r: (REPO / "checkpoints" / r / "best.pt").exists()  # noqa: E731
    missing = [r for r in runs if not have(r)]
    runs = [r for r in runs if have(r)]
    for k, mem in list(ensembles.items()):
        if not all(have(r) for r in mem):
            missing.append(f"{k} ({','.join(r for r in mem if not have(r))})"); ensembles.pop(k)
    for m in missing:
        print(f"skip (no checkpoint): {m}")

    splits = read_splits()
    df = load_all()
    ids = set(segment_ids(a.split, "val", splits))
    sub = df[df["segment_id"].astype(str).isin(ids)]
    pool = EpisodePool(frame_pieces(sub, part_filter(a.split, "val", splits)), n_max)
    sel = pool.select_per_flight(a.n_starts)
    flights = pool.flights[sel]
    speed0 = pool.start_speed(sel)
    per_flight = {f: int((flights == f).sum()) for f in pool.flight_ids()}
    short = {f: c for f, c in per_flight.items() if c < a.n_starts}
    print(f"{a.split}/val: {len(pool.flight_ids())} flights, {len(pool)} valid starts for {n_max} steps; "
          f"selected {len(sel)} ({min(per_flight.values())}-{max(per_flight.values())} per flight); "
          f"|v0|>5: {(speed0 > 5).sum()}; device {device}; {a.mode}, {a.integrator}")
    if short:
        print(f"  flights with fewer than {a.n_starts} valid starts (all used): {short}")
    if pool.skipped:
        print(f"  pieces skipped (sample time != 2.5 ms): {pool.skipped}")

    entries = {}                                          # name -> {label, seed, pos, vel, secs}

    def add(name, label, seed, force, extra=None):
        ts = time.time()
        e = run_rollouts(force, pool, sel, steps, a.mode, a.integrator, device, a.batch)
        e.update(label=label, seed=seed, secs=time.time() - ts, **(extra or {}))
        entries[name] = e
        med = "  ".join(f"{h:g}s {np.median(e['pos'][:, j]) * 100:.2f} cm" for j, h in enumerate(a.horizons))
        print(f"[{name:>32}] median pos err {med}  ({e['secs']:.1f} s)")

    if not a.no_baselines:
        add("oracle", "oracle", None, _Oracle())
        coefs = json.loads((REPO / "results" / "val_baselines.json").read_text())["splits"][a.split]["P0b_coefs"]
        add("P0b", "P0b", None, R.ModelForce(R.physics_model_from_coefs(coefs).to(device)))
    for run in runs:
        m, cfg = load_run(run, device)
        base = re.sub(r"_seed\d+$", "", cfg.get("name", run))
        if cfg["data"]["split"] != a.split:
            print(f"skip {run}: trained on split {cfg['data']['split']}")
            continue
        add(run, LABEL.get(base, base), int(cfg.get("seed", 0)), R.ModelForce(m),
            {"config_name": cfg.get("name")})
    for label, mem in ensembles.items():
        models = [load_run(r, device)[0] for r in mem]
        add(label, label, None, R.ModelForce(MeanOf(models)), {"members": mem})

    # ---- statistics
    boot = _boot_idx(len(np.unique(flights)), 2000, seed=0)
    masks = {k: (speed0 > lo) & (speed0 <= hi) for k, (lo, hi) in SUBSETS.items()}
    boots = {k: _boot_idx(len(np.unique(flights[m])), 2000, seed=0) for k, m in masks.items() if m.any()}
    out = {"source": "scripts/eval_rollout_horizons.py", "git_hash": git_hash(), "splits_sha256": splits["meta"]["sha256"],
           "split": a.split, "part": "val", "mode": a.mode, "integrator": a.integrator, "dt_s": R.DT,
           "horizons_s": a.horizons, "horizon_steps": steps, "n_starts_per_flight_target": a.n_starts,
           "n_starts": int(len(sel)), "starts_per_flight": per_flight, "n_valid_starts": int(len(pool)),
           "n_gt5": int(masks["gt5"].sum()), "device": str(device), "missing": missing,
           "definition": "pos error = |p_sim - p_true| [m]; vel error = world-frame |v_sim - v_true| [m/s]; "
                         "median / IQR over start points; CI = flight bootstrap 2000 of the pooled median",
           "models": {}, "paired": [], "seed_summary": {}}
    for name, e in entries.items():
        res = {"label": e["label"], "seed": e["seed"], "secs": round(e["secs"], 1),
               **({"members": e["members"]} if "members" in e else {}),
               **({"config_name": e["config_name"]} if "config_name" in e else {}), "pos_err_m": {}, "vel_err_mps": {}}
        for sub, msk in masks.items():
            if not msk.any():
                continue
            res["pos_err_m"][sub] = {f"{h:g}": summarize(e["pos"][msk, j], flights[msk], boots[sub])
                                     for j, h in enumerate(a.horizons)}
            res["vel_err_mps"][sub] = {f"{h:g}": {k: v for k, v in summarize(e["vel"][msk, j], flights[msk], boots[sub]).items()
                                                  if k in ("median", "p25", "p75", "iqr", "mean", "n_nonfinite")}
                                       for j, h in enumerate(a.horizons)}
        out["models"][name] = res

    # seed-matched pairs; seed-free entries (baselines, ensembles) pair with every seed
    by_label = defaultdict(dict)
    for name, e in entries.items():
        by_label[e["label"]][e["seed"]] = name
    for la, lb in PAIRS:
        if la not in by_label or lb not in by_label:
            continue
        seeds = sorted({s for s in list(by_label[la]) + list(by_label[lb]) if s is not None}) or [None]
        for s in seeds:
            na = by_label[la].get(s, by_label[la].get(None)); nb = by_label[lb].get(s, by_label[lb].get(None))
            if na is None or nb is None:
                continue
            for sub, msk in masks.items():
                if not msk.any():
                    continue
                for j, h in enumerate(a.horizons):
                    d = paired_median_diff(entries[na]["pos"][msk, j], entries[nb]["pos"][msk, j], flights[msk], boots[sub])
                    out["paired"].append({"a": la, "b": lb, "a_run": na, "b_run": nb, "seed": s, "subset": sub,
                                          "horizon_s": h, **d})
                    if sub == "all" and h == 0.25:
                        print(f"paired [{la} - {lb}, seed {s}, all, 0.25 s] median diff {d['diff_median'] * 100:+.2f} cm "
                              f"[{d['ci95'][0] * 100:+.2f}, {d['ci95'][1] * 100:+.2f}] p={d['p']:.3f} win {d['win_rate_a']:.2f}")
    for label, by_seed in by_label.items():
        names = [n for s, n in sorted(by_seed.items(), key=lambda x: (x[0] is None, x[0])) if s is not None]
        if len(names) < 2:
            continue
        ent = {"runs": names}
        for sub in masks:
            if not masks[sub].any():
                continue
            for h in a.horizons:
                v = np.array([out["models"][n]["pos_err_m"][sub][f"{h:g}"]["median"] for n in names])
                ent[f"{sub}|{h:g}"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "values": v.tolist()}
        out["seed_summary"][label] = ent
        print(f"seeds {label:>12}: median pos err 0.25 s {ent['all|0.25']['mean'] * 100:.2f} ± "
              f"{ent['all|0.25']['sd'] * 100:.2f} cm (n={len(names)})" if "all|0.25" in ent else "")
    out["wall_time_s"] = round(time.time() - t0, 1)

    (REPO / a.out).parent.mkdir(parents=True, exist_ok=True)
    (REPO / a.out).write_text(json.dumps(out, indent=1))
    np.savez_compressed(REPO / a.errors_out, names=np.array(list(entries)), horizons_s=np.array(a.horizons),
                        flights=flights, start_speed=speed0, start_rows=pool.starts[sel],
                        pos_err=np.stack([entries[n]["pos"] for n in entries]),
                        vel_err=np.stack([entries[n]["vel"] for n in entries]))
    print(f"wrote {a.out} and {a.errors_out} ({out['wall_time_s']:.0f} s)")


if __name__ == "__main__":
    main()
