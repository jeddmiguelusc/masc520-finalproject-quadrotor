"""CP6: one-step error analysis on A/val - RMSE by trajectory type x speed bin, and the worst flights.

    .venv/bin/python scripts/error_analysis.py                          # registry headline A + P0b, B1
    .venv/bin/python scripts/error_analysis.py --labels M1 M2 M3 M3ens M3act M4
    -> results/error_analysis_val_final.json
    -> results/error_analysis_val_final.md     (tables, ready to copy into the poster / report)
    (re-uses cache/val_predictions_final.npz from scripts/eval_uncertainty.py when present)

Metric: paper-definition RMSE (src/eval/metrics.py), F in N, M in N·m. Single models: computed per
seed, reported as mean (and sd) over the registry's seeds; ensembles and baselines are seed-free.
Trajectory type from Flights.txt (src/data/load.parse_flights). Speed = |v| of the scored sample.
Cells: fine bins (src/eval/metrics.SPEED_BINS) and coarse bins (<=5, >5 m/s, same rule as the le5 / gt5 subsets); a cell with fewer
than 2 flights is flagged thin=True (its RMSE is one flight's, not a type effect).

Worst flight: per label, flights ranked by flight RMSE (F) and by share of the pooled squared error;
for the reference label (default: the last headline label) also the pooled RMSE without each of the
top flights (leave-one-flight-out) and the rank of that flight under every other label.
Validation only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.load import REPO, parse_flights  # noqa: E402
from src.eval import metrics as Mx  # noqa: E402
from src.eval.predcache import PredCache, git_hash  # noqa: E402

COARSE_BINS = [0.0, 5.0, np.inf]   # "<=5" / ">5": same rule as the le5 / gt5 subsets everywhere else
EVAL = "A_val"


_BIN_ORDER = {lab: i for i, lab in enumerate(Mx.speed_bin_labels() + ["<=5", ">5"])}


def cell_order(k: tuple) -> tuple:
    """Sort cells by trajectory type, then by speed bin in numeric order (not lexical)."""
    return tuple((_BIN_ORDER.get(x, -1), x) for x in k)


def rmse_FM(e2_sum: np.ndarray, n: float) -> tuple[float, float]:
    """Paper F, M from summed squared errors (6,) over n samples."""
    mse = e2_sum / n
    return float(np.sqrt(mse[0:3].mean())), float(np.sqrt(mse[3:6].mean()))


def cell_sse(e2: np.ndarray, keys: list[np.ndarray]) -> dict:
    """{key tuple: (sse (6,), n)} for every combination of the key arrays present in the data."""
    stacked = np.stack([k.astype(str) for k in keys], 1)
    uniq, inv = np.unique(stacked, axis=0, return_inverse=True)
    inv = inv.ravel()
    sse = np.zeros((len(uniq), 6))
    np.add.at(sse, inv, e2)
    n = np.bincount(inv, minlength=len(uniq)).astype(float)
    return {tuple(u): (sse[i], n[i]) for i, u in enumerate(uniq)}


def bin_label(speed: np.ndarray, bins) -> np.ndarray:
    """Fine bins: [lo, hi) as in metrics.by_speed. Coarse bins: <= 5 and > 5 as in evaluate.SUBSETS."""
    if list(bins) == COARSE_BINS:
        return np.where(speed > 5.0, ">5", "<=5")
    labs = np.array(Mx.speed_bin_labels(bins))
    return labs[np.clip(np.digitize(speed, bins[1:-1], right=False), 0, len(labs) - 1)]


def summarise(per_seed: list[dict]) -> dict:
    """[{key: (F, M)}] over seeds -> {key: {"F", "F_sd", "M", "M_sd"}}."""
    out = {}
    for k in per_seed[0]:
        F = np.array([d[k][0] for d in per_seed]); M = np.array([d[k][1] for d in per_seed])
        out[k] = {"F": float(F.mean()), "M": float(M.mean()),
                  "F_sd": float(F.std(ddof=1)) if len(F) > 1 else None,
                  "M_sd": float(M.std(ddof=1)) if len(M) > 1 else None}
    return out


def analyse(Y, preds: dict, flights, speed, traj_of: dict) -> dict:
    """preds: {label: [P per seed]} -> tables (pure numpy; tested in tests/test_error_analysis.py)."""
    traj = np.array([traj_of.get(f, "unknown") for f in flights])
    fine, coarse = bin_label(speed, Mx.SPEED_BINS), bin_label(speed, COARSE_BINS)
    groupings = {"type_x_bin": [traj, fine], "type_x_coarse": [traj, coarse], "type": [traj],
                 "bin": [fine], "coarse": [coarse], "flight": [flights]}
    sizes = {}                                    # cell sizes (label-free)
    for g, keys in groupings.items():
        ks = np.stack([k.astype(str) for k in keys], 1)
        sizes[g] = {}
        for key in {tuple(r) for r in ks}:
            m = np.all(ks == np.array(key), axis=1)
            sizes[g][key] = {"n": int(m.sum()), "n_flights": int(len(np.unique(flights[m])))}
    tables = {g: {} for g in groupings}
    for lab, Ps in preds.items():
        for g, keys in groupings.items():
            per_seed = []
            for P in Ps:
                e2 = (P - Y) ** 2
                per_seed.append({k: rmse_FM(s, n) for k, (s, n) in cell_sse(e2, keys).items()})
            tables[g][lab] = summarise(per_seed)
    # total SSE share per flight (mean over seeds)
    share = {}
    for lab, Ps in preds.items():
        sh = []
        for P in Ps:
            e2 = ((P - Y) ** 2)[:, 0:3].sum(1)
            tot = e2.sum()
            fl, inv = np.unique(flights, return_inverse=True)
            sh.append(dict(zip(fl, np.bincount(inv, weights=e2) / tot)))
        share[lab] = {f: float(np.mean([s[f] for s in sh])) for f in sh[0]}
    return {"traj": traj, "tables": tables, "sizes": sizes, "share_F_sse": share}


def lofo(Y, Ps, flights, drop) -> dict:
    """Pooled F, M without flight `drop` (mean over seeds)."""
    m = flights != drop
    vals = [rmse_FM(((P[m] - Y[m]) ** 2).sum(0), m.sum()) for P in Ps]
    return {"F": float(np.mean([v[0] for v in vals])), "M": float(np.mean([v[1] for v in vals]))}


def fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="*", default=None, help="registry labels (default: headline A)")
    ap.add_argument("--baselines", nargs="*", default=["P0b", "B1"])
    ap.add_argument("--reference", default=None, help="label for leave-one-flight-out (default: last label)")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default="results/error_analysis_val_final.json")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    t0 = time.time()
    from src.train import pick_device
    pc = PredCache(pick_device(a.device))
    reg = pc.reg
    labels = a.labels or list(reg.headline["A"])
    for lab in labels:
        if reg.split_of(lab) != "A":
            raise ValueError(f"{lab}: error analysis is on split A models only (B_s models: H1 view elsewhere)")
    ref = a.reference or labels[-1]
    rows = pc.rows(EVAL)
    Y = rows["Y"].astype(np.float64)
    flights, speed = rows["flights"], rows["speed"]
    preds, runs_of = {}, {}
    for b in a.baselines:
        preds[b] = [pc.baseline("A", b, EVAL)]
        runs_of[b] = [f"baseline:A:{b}"]
    for lab in labels:
        if reg.is_ensemble(lab):
            preds[lab] = [pc.ensemble(lab, EVAL)["mean"]]
            runs_of[lab] = reg.ensemble_members()[lab]
        else:
            rr = reg.models[lab]["runs"]
            preds[lab] = [pc.run(r, EVAL)[0] for r in rr]
            runs_of[lab] = rr
    pc.save()
    fl_tab = parse_flights()
    traj_of = dict(zip(fl_tab.flight_id, fl_tab.trajectory_type))
    res = analyse(Y, preds, flights, speed, traj_of)
    T, S = res["tables"], res["sizes"]
    order = list(preds)

    # ---- consistency: pooled F per label vs results/val_models_final.json
    checks = []
    vm = json.loads((REPO / "results" / "val_models_final.json").read_text())
    for lab in labels:
        Fs = [rmse_FM(((P - Y) ** 2).sum(0), len(Y))[0] for P in preds[lab]]
        if reg.is_ensemble(lab):
            old = [vm["ensembles"][lab]["eval"][EVAL]["all"]["ci95"]["F"][0]]
        else:
            old = [vm["runs"][r]["eval"][EVAL]["all"]["ci95"]["F"][0] for r in runs_of[lab]]
        ok = bool(np.allclose(Fs, old, rtol=1e-4))
        checks.append({"label": lab, "F_here": Fs, "F_val_models_final": old, "match": ok})
        print(f"check {lab}: pooled F {np.round(Fs, 4).tolist()} vs val_models_final {np.round(old, 4).tolist()} "
              f"-> {'OK' if ok else 'MISMATCH'}")

    # ---- per-flight table and worst flights
    flist = sorted(np.unique(flights))
    flight_info = {}
    for f in flist:
        m = flights == f
        flight_info[f] = {"trajectory_type": traj_of.get(f, "unknown"), "n": int(m.sum()),
                          "max_speed": float(speed[m].max()), "median_speed": float(np.median(speed[m])),
                          "share_gt5": float((speed[m] > 5).mean())}
    per_flight = {lab: {f[0]: v for f, v in T["flight"][lab].items()} for lab in order}
    worst = {}
    for lab in order:
        rank = sorted(flist, key=lambda f: -per_flight[lab][f]["F"])
        worst[lab] = [{"flight": f, **flight_info[f], "F": per_flight[lab][f]["F"], "M": per_flight[lab][f]["M"],
                       "F_sd": per_flight[lab][f]["F_sd"], "share_of_F_sse": res["share_F_sse"][lab][f]}
                      for f in rank[:a.top]]
    rank_ref = sorted(flist, key=lambda f: -per_flight[ref][f]["F"])
    pooled_ref = summarise([{"x": rmse_FM(((P - Y) ** 2).sum(0), len(Y))} for P in preds[ref]])["x"]
    worst_ref = []
    for f in rank_ref[:a.top]:
        ranks = {lab: sorted(flist, key=lambda g: -per_flight[lab][g]["F"]).index(f) + 1 for lab in order}
        worst_ref.append({"flight": f, **flight_info[f], "F": per_flight[ref][f]["F"], "M": per_flight[ref][f]["M"],
                          "share_of_F_sse": res["share_F_sse"][ref][f], "pooled_without": lofo(Y, preds[ref], flights, f),
                          "F_rank_under": ranks, "F_under": {lab: per_flight[lab][f]["F"] for lab in order}})

    # ---- JSON
    def jt(tab, sizes):
        return {"|".join(k): {"n": sizes[k]["n"], "n_flights": sizes[k]["n_flights"], "thin": sizes[k]["n_flights"] < 2,
                              **{lab: tab[lab][k] for lab in order}} for k in sorted(sizes, key=cell_order)}
    out = {"source": "scripts/error_analysis.py", "git": git_hash(), "registry": reg.info(),
           "splits_sha256": pc.splits["meta"]["sha256"], "eval": EVAL, "labels": order, "runs": runs_of,
           "metric": "paper RMSE, F (N) and M (N·m); single models = mean (sd) over registry seeds",
           "checks": checks,
           "by_type_x_speed_bin": jt(T["type_x_bin"], S["type_x_bin"]),
           "by_type_x_coarse": jt(T["type_x_coarse"], S["type_x_coarse"]),
           "by_type": jt(T["type"], S["type"]), "by_speed_bin": jt(T["bin"], S["bin"]),
           "by_coarse": jt(T["coarse"], S["coarse"]),
           "per_flight": {f: {**flight_info[f], **{lab: per_flight[lab][f] for lab in order},
                              "share_of_F_sse": {lab: res["share_F_sse"][lab][f] for lab in order}} for f in flist},
           "worst_flights": worst, "reference": ref, "reference_pooled": pooled_ref, "worst_flights_reference": worst_ref}
    (REPO / a.out).write_text(json.dumps(out, indent=1))

    # ---- markdown
    md = [f"# Error analysis, {EVAL} (scripts/error_analysis.py, git {out['git']})", "",
          "Paper RMSE. Single models: mean over registry seeds. thin = fewer than 2 flights in the cell.", ""]
    for title, key in (("trajectory type x speed bin", "type_x_bin"), ("trajectory type x coarse speed", "type_x_coarse"),
                       ("trajectory type", "type"), ("speed bin", "bin")):
        tab = T[key]
        for q, nd, qname in (("F", 3, "Force F (N)"), ("M", 4, "Moment M (N·m)")):
            md += [f"## {qname} by {title}", "",
                   "| cell | n | flights | " + " | ".join(order) + " |",
                   "|---|---:|---:|" + "---:|" * len(order)]
            for k in sorted(S[key], key=cell_order):
                sz = S[key][k]
                md.append(f"| {' / '.join(k)}{' (thin)' if sz['n_flights'] < 2 else ''} | {sz['n']:,} | "
                          f"{sz['n_flights']} | " + " | ".join(fmt(tab[lab][k][q], nd) for lab in order) + " |")
            md.append("")
    md += [f"## Worst flights under {ref} (by flight F RMSE)", "",
           f"Pooled {ref}: F {pooled_ref['F']:.3f} N, M {pooled_ref['M']:.4f} N·m.", "",
           "| flight | type | max |v| | n | F | M | share of F SSE | pooled F without | rank under " + ", ".join(order) + " |",
           "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for w in worst_ref:
        md.append(f"| {w['flight']} | {w['trajectory_type']} | {w['max_speed']:.1f} | {w['n']:,} | {w['F']:.3f} | "
                  f"{w['M']:.4f} | {w['share_of_F_sse']:.1%} | {w['pooled_without']['F']:.3f} | "
                  + ", ".join(str(w["F_rank_under"][lab]) for lab in order) + " |")
    md.append("")
    (REPO / a.out).with_suffix(".md").write_text("\n".join(md))

    # ---- stdout
    print(f"\nF (N) by trajectory type x coarse speed (n flights):")
    print(f"{'cell':>34} {'n':>8} {'fl':>3} " + " ".join(f"{lab:>8}" for lab in order))
    for k in sorted(S["type_x_coarse"], key=cell_order):
        sz = S["type_x_coarse"][k]
        print(f"{' / '.join(k):>34} {sz['n']:>8} {sz['n_flights']:>3} "
              + " ".join(f"{T['type_x_coarse'][lab][k]['F']:>8.3f}" for lab in order) + ("  thin" if sz["n_flights"] < 2 else ""))
    print(f"\nWorst flights under {ref} (pooled F {pooled_ref['F']:.3f} N):")
    for w in worst_ref:
        print(f"  {w['flight']} {w['trajectory_type']:>20} max|v| {w['max_speed']:5.1f}  F {w['F']:.3f}  M {w['M']:.4f}  "
              f"share {w['share_of_F_sse']:.1%}  pooled F without {w['pooled_without']['F']:.3f}  ranks "
              + " ".join(f"{lab}:{w['F_rank_under'][lab]}" for lab in order))
    out["runtime_s"] = round(time.time() - t0, 1)
    (REPO / a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out} and {Path(a.out).with_suffix('.md')} ({out['runtime_s']} s)")
    if any(not c["match"] for c in checks):
        print("WARNING: pooled F does not match results/val_models_final.json - do not use these numbers yet")


if __name__ == "__main__":
    main()
