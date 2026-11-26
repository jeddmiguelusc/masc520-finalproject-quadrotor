"""CP3/CP4: score trained checkpoints on validation with the paper's metrics, CIs and speed bins.

    python scripts/eval_models.py                       # the Wed seed-0 runs (default list below)
    python scripts/eval_models.py --all                 # every checkpoint + the two H4 ensembles
    python scripts/eval_models.py --runs m3_hybrid_tcn_seed0 m2_hybrid_mlp_seed0
    python scripts/eval_models.py --runs ... --ensemble m3_ens=m3_ens_seed0,m3_ens_seed1,m3_ens_seed2
    -> results/val_models.json       point metrics, CIs, by-speed tables, paired diffs, seed summary
    -> results/val_uncertainty.json  (only with ensembles) H4: sigma vs speed, coverage, aleatoric/epistemic
    (needs results/val_baselines.json for paired comparisons with P0b / B1 / B2)

Each run's checkpoint (checkpoints/<run>/best.pt) carries its config; the split decides the
evaluation sets exactly as in scripts/eval_baselines.py:
  A   -> A/val
  B_s -> B_s/val (slow only) and A/val (all speeds; |v|>5 subset = speed-extrapolation view, H1)
Validation only. Sanity check: F_vecnorm here must match val_rmse_F_N in results/runs.csv.

Paired diffs are seed-matched (M2 seed k - M1 seed k); baselines are deterministic. The seed
summary gives mean / sd / min / max of each point metric over the seeds of one config.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_part, read_splits  # noqa: E402
from src.data.load import REPO, load_all  # noqa: E402
from src.eval import metrics as Mx  # noqa: E402
from src.eval.evaluate import paired, score, scored_rows, short  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.ensemble import combine  # noqa: E402
from src.train import pick_device  # noqa: E402

DEFAULT_RUNS = ["m1_mlp_seed0", "m2_hybrid_mlp_seed0", "m3_hybrid_tcn_seed0",
                "m1_mlp_Bs_seed0", "m2_hybrid_mlp_Bs_seed0"]
DEFAULT_ENSEMBLES = {"m3_ens": [f"m3_ens_seed{s}" for s in range(3)],
                     "m3_ens_Bs": [f"m3_ens_Bs_seed{s}" for s in range(3)]}
EVAL_SETS = {"A": [("A", "val")], "B_s": [("B_s", "val"), ("A", "val")]}
LABEL = {"m1_mlp": "M1", "m2_hybrid_mlp": "M2", "m3_hybrid_tcn": "M3", "m4_hybrid_transformer": "M4",
         "m1_mlp_Bs": "M1", "m2_hybrid_mlp_Bs": "M2", "m3_hybrid_tcn_Bs": "M3",
         "m3_ens": "M3nll", "m3_ens_Bs": "M3nll", "m3_hybrid_tcn_act": "M3act", "m3_hybrid_tcn_act_Bs": "M3act"}
# (split, eval set, a, b, subset): metric(a) - metric(b); negative = a better
PAIRS = [("A", "A_val", "M2", "M1", "all"), ("A", "A_val", "M3", "M2", "all"), ("A", "A_val", "M1", "B1", "all"),
         ("A", "A_val", "M2", "P0b", "all"), ("A", "A_val", "M3", "B1", "all"),
         ("A", "A_val", "M2", "M1", "gt5"), ("A", "A_val", "M3", "M2", "gt5"),
         ("A", "A_val", "M3", "B2", "all"), ("A", "A_val", "M3nll", "M3", "all"),
         ("A", "A_val", "M3ens", "M3", "all"), ("A", "A_val", "M3act", "M3", "all"),
         ("A", "A_val", "M3act", "M2", "all"), ("A", "A_val", "M3act", "B2", "all"),
         ("A", "A_val", "M4", "M3", "all"), ("B_s", "A_val", "M3act", "M2", "gt5"),
         ("B_s", "B_s_val", "M2", "M1", "all"),
         ("B_s", "A_val", "M2", "M1", "all"), ("B_s", "A_val", "M2", "M1", "gt5"),
         ("B_s", "A_val", "M2", "P0b", "gt5"), ("B_s", "A_val", "M1", "B1", "gt5"),
         ("B_s", "A_val", "M3", "M2", "gt5"), ("B_s", "A_val", "M3", "P0b", "gt5"),
         ("B_s", "A_val", "M3ens", "M3", "gt5")]
COVERAGE_Z = {"68": 1.0, "90": 1.6449, "95": 1.96}


@torch.no_grad()
def predict(model, rows: dict, window: int, device, batch: int = 8192) -> tuple[np.ndarray, np.ndarray | None]:
    """Mean (N,6) and, for NLL models, variance (N,6) at every scored sample."""
    ws = rows["ws"]
    X = torch.from_numpy(ws.X).to(device)
    offs = torch.arange(-window + 1, 1, device=device)
    ends = torch.from_numpy(ws.ends).long().to(device)
    means, vars_ = [], []
    for a in range(0, len(ends), batch):
        idx = ends[a:a + batch, None] + offs[None, :]
        mean, var = model(X[idx])
        means.append(mean.float().cpu().numpy())
        if var is not None:
            vars_.append(var.float().cpu().numpy())
    mean = np.concatenate(means).astype(np.float64)
    return mean, (np.concatenate(vars_).astype(np.float64) if vars_ else None)


def base_name(name: str) -> str:
    """Sweep runs (<config>_sw<k>, scripts/sweep_thu.sh) and tuned configs (<config>_tuned,
    scripts/sweep_summary.py) share their config's label."""
    return re.sub(r"(_sw\d+|_tuned)$", "", name)


def run_seed(run: str) -> int:
    m = re.search(r"_seed(\d+)$", run)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------- H4: uncertainty statistics
def _cov_counts(e: np.ndarray, sd: np.ndarray, flights: np.ndarray, fl: np.ndarray) -> dict:
    """Per-flight counts of |e| <= z*sigma (pooled over the given components)."""
    inv = np.searchsorted(fl, flights)
    out = {}
    for k, z in COVERAGE_Z.items():
        hit = (np.abs(e) <= z * sd).sum(1)
        out[k] = np.bincount(inv, weights=hit, minlength=len(fl))
    return out


def uncertainty_stats(rows: dict, ens: dict, b: int = 2000, seed: int = 0) -> dict:
    """sigma vs speed, coverage and aleatoric/epistemic split for force (F) and moment (M) separately.

    sigma_rms = sqrt(mean of predicted variance over samples and the 3 components) - same pooling as
    the paper RMSE, so sigma_rms and RMSE are directly comparable (calibrated ~ sigma_rms = RMSE).
    Coverage = share of (sample, component) pairs with |error| <= z*sigma, nominal 68.3/90/95%.
    CIs: percentile bootstrap over flights (same resampling as the RMSE CIs).
    """
    Y, mean, var = rows["Y"].astype(np.float64), ens["mean"], ens["var"]
    alea, epi = ens["aleatoric"], ens["epistemic"]
    speed, flights = rows["speed"], rows["flights"]
    fl = np.unique(flights)
    inv = np.searchsorted(fl, flights)
    idx = np.random.default_rng(seed).integers(0, len(fl), size=(b, len(fl)))
    out = {"definition": "sigma_rms = sqrt(mean predicted variance over samples and 3 components); "
                         "coverage = share of |e| <= z*sigma; CIs = flight bootstrap 2000",
           "nominal_coverage": {k: {"68": 0.6827, "90": 0.90, "95": 0.95}[k] for k in COVERAGE_Z}}
    masks = {"all": np.ones(len(speed), bool), "le5": speed <= 5.0, "gt5": speed > 5.0}
    for grp, sl in (("F", slice(0, 3)), ("M", slice(3, 6))):
        e, v = mean[:, sl] - Y[:, sl], var[:, sl]
        g = {"subsets": {}, "by_speed": {}}
        sums = {}
        for name, m in masks.items():
            if not m.any():
                continue
            fm = flights[m]
            n_f = np.bincount(inv[m], minlength=len(fl)).astype(np.float64) * 3
            var_f = np.bincount(inv[m], weights=v[m].sum(1), minlength=len(fl))
            se_f = np.bincount(inv[m], weights=(e[m] ** 2).sum(1), minlength=len(fl))
            al_f = np.bincount(inv[m], weights=alea[m][:, sl].sum(1), minlength=len(fl))
            ep_f = np.bincount(inv[m], weights=epi[m][:, sl].sum(1), minlength=len(fl))
            cov = _cov_counts(e[m], np.sqrt(v[m]), fm, fl)
            sums[name] = (n_f, var_f)

            def ci(num, den, f=lambda x: x):
                pt = f(num.sum() / den.sum())
                reps = f(num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12))
                return [float(pt), *np.percentile(reps, [2.5, 97.5]).tolist()]
            g["subsets"][name] = {
                "n": int(m.sum()),
                "sigma_rms": ci(var_f, n_f, np.sqrt), "rmse": ci(se_f, n_f, np.sqrt),
                "sigma_aleatoric_rms": ci(al_f, n_f, np.sqrt), "sigma_epistemic_rms": ci(ep_f, n_f, np.sqrt),
                "epistemic_share_of_var": float(ep_f.sum() / var_f.sum()),
                "coverage": {k: ci(c, n_f) for k, c in cov.items()},
                "mean_z2": float(((e[m] ** 2) / v[m]).mean()),
            }
        if "le5" in sums and "gt5" in sums:
            (n_lo, v_lo), (n_hi, v_hi) = sums["le5"], sums["gt5"]
            pt = np.sqrt((v_hi.sum() / n_hi.sum()) / (v_lo.sum() / n_lo.sum()))
            reps = np.sqrt((v_hi[idx].sum(1) / np.maximum(n_hi[idx].sum(1), 1)) /
                           (v_lo[idx].sum(1) / np.maximum(n_lo[idx].sum(1), 1)))
            g["sigma_ratio_gt5_over_le5"] = [float(pt), *np.nanpercentile(reps, [2.5, 97.5]).tolist()]
        sig = np.sqrt(v.mean(1))
        from scipy.stats import spearmanr
        g["spearman_speed_sigma"] = float(spearmanr(speed, sig).statistic)
        g["spearman_abs_error_sigma"] = float(spearmanr(np.sqrt((e ** 2).mean(1)), sig).statistic)
        for lab, lo, hi in zip(Mx.speed_bin_labels(), Mx.SPEED_BINS[:-1], Mx.SPEED_BINS[1:]):
            m = (speed >= lo) & (speed < hi)
            if not m.any():
                continue
            z = np.abs(e[m]) / np.sqrt(v[m])
            g["by_speed"][lab] = {"n": int(m.sum()), "rmse": float(np.sqrt((e[m] ** 2).mean())),
                                  "sigma_rms": float(np.sqrt(v[m].mean())),
                                  "sigma_aleatoric_rms": float(np.sqrt(alea[m][:, sl].mean())),
                                  "sigma_epistemic_rms": float(np.sqrt(epi[m][:, sl].mean())),
                                  **{f"coverage_{k}": float((z <= zz).mean()) for k, zz in COVERAGE_Z.items()}}
        out[grp] = g
    return out


# ---------------------------------------------------------------- main
def discover_runs() -> list[str]:
    """Every checkpoint with a known config; sweep runs (_sw<k>) are skipped (scripts/sweep_summary.py
    scores them), and where a *_tuned config exists its runs replace the untuned ones of that config,
    so each (label, split, seed) appears once."""
    found = {}
    for d in sorted((REPO / "checkpoints").iterdir()):
        cfg = re.sub(r"_seed\d+$", "", d.name)
        if not (d / "best.pt").exists() or re.search(r"_sw\d+$", cfg) or base_name(cfg) not in LABEL:
            continue
        found.setdefault(base_name(cfg), {}).setdefault(cfg, []).append(d.name)
    runs = []
    for base, by_cfg in found.items():
        runs += by_cfg.get(f"{base}_tuned", by_cfg.get(base, []))
    return sorted(runs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="every checkpoint with a known config + default ensembles")
    ap.add_argument("--ensemble", nargs="*", default=[], help="name=run1,run2,... (members must share a split)")
    ap.add_argument("--out", default="results/val_models.json")
    ap.add_argument("--out-unc", default="results/val_uncertainty.json")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    ensembles = {k: v.split(",") for k, v in (e.split("=", 1) for e in a.ensemble)}
    if a.all:
        runs = discover_runs()
        for k, v in DEFAULT_ENSEMBLES.items():
            tuned = [r.replace("_seed", "_tuned_seed") for r in v]
            for members in (tuned, v):            # prefer the tuned members when all exist
                if all((REPO / "checkpoints" / r / "best.pt").exists() for r in members):
                    ensembles.setdefault(k, members)
                    break
    else:
        runs = a.runs if a.runs is not None else DEFAULT_RUNS
    runs = list(dict.fromkeys(runs + [r for v in ensembles.values() for r in v]))
    device = pick_device(a.device)
    splits, df = read_splits(), load_all()
    base = json.loads((REPO / "results" / "val_baselines.json").read_text())
    runs_csv = {r["run"]: r for r in csv.DictReader((REPO / "results" / "runs.csv").open())}
    cache, preds = {}, {}
    out = {"source": "scripts/eval_models.py", "splits_sha256": splits["meta"]["sha256"],
           "metric_definition": base["metric_definition"], "device": str(device), "runs": {},
           "ensembles": {}, "paired": [], "seed_summary": {}}
    table = {}                         # (split, eval set, label, seed) -> score
    for run in runs:
        ck = torch.load(REPO / "checkpoints" / run / "best.pt", map_location=device, weights_only=False)
        cfg = ck["config"]
        key = cfg["data"]["split"]
        model = build_model(cfg["model"]); model.load_state_dict(ck["model"]); model.to(device).eval()
        label = LABEL.get(base_name(cfg.get("name", "")), cfg.get("name", run))
        seed = int(cfg.get("seed", run_seed(run)))
        res = {"split": key, "label": label, "seed": seed, "config_name": cfg.get("name"), "eval": {}}
        for k, p in EVAL_SETS[key]:
            name = f"{k}_{p}"
            if (k, p) not in cache:
                cache[(k, p)] = scored_rows(load_part(k, p, splits, df))
            rows = cache[(k, p)]
            P, V = predict(model, rows, int(cfg["data"]["max_window"]), device)
            preds[(run, name)] = (P, V)
            res["eval"][name] = sc = score(rows, P)
            table[(key, name, label, seed)] = sc
            print(f"[{run} -> {name}] {short(sc)}")
            if sc["gt5"]["n"]:
                print(f"{'':>{len(run) + len(name) + 6}} |v|>5: {short(sc, 'gt5')}")
            if name == f"{key}_val" and run in runs_csv:
                csv_f = float(runs_csv[run]["val_rmse_F_N"]); here = sc["all"]["ci95"]["F_vecnorm"][0]
                ok = abs(csv_f - here) < 1e-3
                res["check_vs_runs_csv"] = {"runs_csv_val_rmse_F_N": csv_f, "F_vecnorm_here": here, "match": ok}
                print(f"   check: runs.csv F {csv_f:.4f} vs F_vecnorm {here:.4f} {'OK' if ok else 'MISMATCH'}")
        out["runs"][run] = res

    # ---- ensembles (Gaussian mixture of the members; H4)
    unc = {"source": "scripts/eval_models.py", "splits_sha256": splits["meta"]["sha256"], "ensembles": {}}
    for ename, members in ensembles.items():
        key = out["runs"][members[0]]["split"]
        e_res = {"split": key, "members": members, "eval": {}}
        unc["ensembles"][ename] = {"split": key, "members": members, "eval": {}}
        for k, p in EVAL_SETS[key]:
            name = f"{k}_{p}"
            rows = cache[(k, p)]
            mv = [preds[(r, name)] for r in members]
            c = combine([torch.from_numpy(m) for m, _ in mv],
                        [None if v is None else torch.from_numpy(v) for _, v in mv])
            ens = {kk: vv.numpy() for kk, vv in c.items()}
            e_res["eval"][name] = sc = score(rows, ens["mean"])
            table[(key, name, "M3ens", 0)] = sc
            unc["ensembles"][ename]["eval"][name] = u = uncertainty_stats(rows, ens)
            print(f"[ensemble {ename} -> {name}] {short(sc)}")
            for grp in ("F", "M"):
                s = u[grp]["subsets"]
                line = "  ".join(f"{sub}: sigma {s[sub]['sigma_rms'][0]:.4g} rmse {s[sub]['rmse'][0]:.4g} "
                                 f"cov90 {s[sub]['coverage']['90'][0]:.3f}" for sub in ("le5", "gt5") if sub in s)
                ratio = u[grp].get("sigma_ratio_gt5_over_le5")
                print(f"   {grp}: {line}" + (f"  ratio gt5/le5 {ratio[0]:.2f} [{ratio[1]:.2f},{ratio[2]:.2f}]"
                                             if ratio else "") + f"  rho(speed,sigma) {u[grp]['spearman_speed_sigma']:.2f}")
        out["ensembles"][ename] = e_res

    # ---- paired diffs (seed-matched; baselines and ensembles are seed-free)
    seeds_present = sorted({s for (_, _, _, s) in table})
    for key, name, la, lb, sub in PAIRS:
        for s in seeds_present:
            def get(lab):
                if (key, name, lab, s) in table:
                    return table[(key, name, lab, s)]
                if lab == "M3ens" and (key, name, lab, 0) in table:
                    return table[(key, name, lab, 0)]
                return base["splits"][key]["eval"].get(name, {}).get(lab)
            A_, B_ = get(la), get(lb)
            if A_ is None or B_ is None or (la == "M3ens" and lb == "M3ens"):
                continue
            if s > 0 and not any((key, name, lab, s) in table for lab in (la, lb)):
                continue                  # both sides seed-free: already done at seed 0
            d = paired(A_, B_, sub)
            out["paired"].append({"split": key, "eval": name, "a": la, "b": lb, "subset": sub, "seed": s,
                                  "diff_ci95_p": d})
            print(f"paired [{key} -> {name}, {sub}, seed {s}] {la} - {lb}: F {d['F'][0]:+.3f} "
                  f"[{d['F'][1]:+.3f},{d['F'][2]:+.3f}] N (p={d['F'][3]:.3f})  M {d['M'][0]:+.4f} "
                  f"[{d['M'][1]:+.4f},{d['M'][2]:+.4f}] (p={d['M'][3]:.3f})")

    # ---- seed summary
    groups = defaultdict(list)
    for run, r in out["runs"].items():
        for name, sc in r["eval"].items():
            for sub in ("all", "gt5"):
                if sc[sub]["n"]:
                    groups[(r["config_name"], name, sub)].append((r["seed"], sc[sub]["ci95"]))
    for (cfg, name, sub), lst in sorted(groups.items()):
        ent = {"seeds": [s for s, _ in lst]}
        for met in ("F", "M", "Fxy", "Fz", "Mxy", "Mz"):
            v = np.array([c[met][0] for _, c in lst])
            ent[met] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
                        "min": float(v.min()), "max": float(v.max()), "values": v.tolist()}
        out["seed_summary"][f"{cfg}|{name}|{sub}"] = ent
        if len(lst) > 1:
            print(f"seeds {cfg:>20} {name:>8} {sub:>3}: F {ent['F']['mean']:.3f} ± {ent['F']['sd']:.3f} "
                  f"(range {ent['F']['min']:.3f}-{ent['F']['max']:.3f})  M {ent['M']['mean']:.4f} ± "
                  f"{ent['M']['sd']:.4f}  n={len(lst)}")

    (REPO / a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")
    if ensembles:
        (REPO / a.out_unc).write_text(json.dumps(unc, indent=1))
        print(f"wrote {a.out_unc}")


if __name__ == "__main__":
    main()
