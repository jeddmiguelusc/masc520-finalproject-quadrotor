"""CP6: uncertainty evaluation of the registry ensembles on validation (H4).

    .venv/bin/python scripts/eval_uncertainty.py                  # all ensembles in configs/final_models.yaml
    .venv/bin/python scripts/eval_uncertainty.py --ensembles M3ens M3ens_Bs
    -> results/uncertainty_val_final.json
    -> cache/val_predictions_final.npz (member predictions; git-ignored, reused by scripts/error_analysis.py)

For each ensemble (Gaussian mixture of NLL members: mean = mean of mu_k,
var = aleatoric (mean sigma_k^2) + epistemic (variance of mu_k), src/models/ensemble.py) and for
force (F, N) and moment (M, N·m) separately, on all / |v| <= 5 / |v| > 5 and per speed bin:

  sigma_rms, rmse        sqrt(mean over samples and the 3 components); calibrated ~ equal
  aleatoric / epistemic  sqrt(mean sigma_alea^2), sqrt(mean sigma_epi^2), epistemic share of variance
  mean_sigma             mean over samples and components of sigma (per speed bin: "mean sigma")
  nll_gauss              mean Gaussian NLL of y under N(mean, var), per component, nats (unit-dependent:
                         compare models within F or within M only, never F with M)
  nll_mixture            NLL under the actual K-member mixture (members' own Gaussians)
  nll_member_mean        mean over members of each member's own NLL (does ensembling help?)
  coverage               share of |e| <= z*sigma at 1 sigma, 2 sigma (nominal 0.9545), 90%, 95%
  calibration curve      observed vs nominal central-interval coverage at 20 levels; miscalibration
                         area = mean |observed - nominal| over 0.05..0.95

CIs: percentile flight bootstrap (2000 resamples, seed 0) - same resampling as every other script.
Consistency check: sigma_rms and 90% coverage on the full set must match results/val_uncertainty_final.json
(written by scripts/eval_models.py --final) - printed as OK / MISMATCH.
Validation only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.load import REPO  # noqa: E402
from src.eval import metrics as Mx  # noqa: E402
from src.eval.predcache import PredCache, git_hash  # noqa: E402

EVAL_SETS = {"A": ["A_val"], "B_s": ["A_val", "B_s_val"]}
CURVE_LEVELS = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)] + [0.99]
COVERAGE = {"1sigma": 1.0, "2sigma": 2.0, "90": float(norm.ppf(0.95)), "95": float(norm.ppf(0.975))}
NOMINAL = {"1sigma": float(2 * norm.cdf(1) - 1), "2sigma": float(2 * norm.cdf(2) - 1), "90": 0.90, "95": 0.95}
LOG2PI = float(np.log(2 * np.pi))
GROUPS = {"F": slice(0, 3), "M": slice(3, 6)}


def gauss_nll(e: np.ndarray, v: np.ndarray) -> np.ndarray:
    return 0.5 * (LOG2PI + np.log(v) + e ** 2 / v)


def mixture_nll(y: np.ndarray, mus: list[np.ndarray], vs: list[np.ndarray]) -> np.ndarray:
    """-log (1/K) sum_k N(y; mu_k, v_k), elementwise."""
    lp = np.stack([-gauss_nll(m - y, v) for m, v in zip(mus, vs)])
    return -(logsumexp(lp, axis=0) - np.log(len(mus)))


class FlightBoot:
    """Ratio-of-sums statistics with a flight bootstrap (num and den summed per flight)."""

    def __init__(self, flights: np.ndarray, b: int = 2000, seed: int = 0):
        self.fl, self.inv = np.unique(flights, return_inverse=True)
        self.idx = np.random.default_rng(seed).integers(0, len(self.fl), size=(b, len(self.fl)))

    def per_flight(self, w: np.ndarray, m: np.ndarray) -> np.ndarray:
        return np.bincount(self.inv[m], weights=w, minlength=len(self.fl))

    def ci(self, num_f: np.ndarray, den_f: np.ndarray, f=lambda x: x) -> list[float]:
        pt = f(num_f.sum() / den_f.sum())
        den = den_f[self.idx].sum(1)
        ok = den > 0
        reps = f(num_f[self.idx].sum(1)[ok] / den[ok])
        return [float(pt), *np.percentile(reps, [2.5, 97.5]).tolist()]


def group_stats(Y, ens, speed, flights, sl, boot: FlightBoot) -> dict:
    """All H4 statistics for one target group (F or M)."""
    y = Y[:, sl]
    mean, var = ens["mean"][:, sl], ens["var"][:, sl]
    alea, epi = ens["aleatoric"][:, sl], ens["epistemic"][:, sl]
    e = mean - y
    sd = np.sqrt(var)
    z = np.abs(e) / sd
    nll_g = gauss_nll(e, var)
    mp = ens.get("member_preds") or []
    have_mix = len(mp) > 1 and all(v is not None for _, v in mp)
    if have_mix:
        mus = [m[:, sl] for m, _ in mp]
        vs = [v[:, sl] for _, v in mp]
        nll_mix = mixture_nll(y, mus, vs)
        nll_mem = np.mean([gauss_nll(m - y, v) for m, v in zip(mus, vs)], axis=0)
    out = {"subsets": {}, "by_speed": {}, "calibration_curve": {}}
    masks = {"all": np.ones(len(speed), bool), "le5": speed <= 5.0, "gt5": speed > 5.0}
    for name, m in masks.items():
        if not m.any():
            continue
        n_f = boot.per_flight(np.full(m.sum(), 3.0), m)
        pf = lambda a: boot.per_flight(a[m].sum(1), m)            # noqa: E731
        s = {"n": int(m.sum()), "n_flights": int((n_f > 0).sum()),
             "sigma_rms": boot.ci(pf(var), n_f, np.sqrt),
             "rmse": boot.ci(pf(e ** 2), n_f, np.sqrt),
             "sigma_aleatoric_rms": boot.ci(pf(alea), n_f, np.sqrt),
             "sigma_epistemic_rms": boot.ci(pf(epi), n_f, np.sqrt),
             "epistemic_share_of_var": boot.ci(pf(epi), pf(var)),
             "mean_sigma": boot.ci(pf(sd), n_f),
             "mean_sigma_aleatoric": boot.ci(pf(np.sqrt(alea)), n_f),
             "mean_sigma_epistemic": boot.ci(pf(np.sqrt(epi)), n_f),
             "nll_gauss": boot.ci(pf(nll_g), n_f),
             "coverage": {k: boot.ci(pf((z <= zz).astype(float)), n_f) for k, zz in COVERAGE.items()},
             "mean_z2": float((z[m] ** 2).mean())}
        s["rmse_over_sigma_rms"] = s["rmse"][0] / s["sigma_rms"][0]
        if have_mix:
            s["nll_mixture"] = boot.ci(pf(nll_mix), n_f)
            s["nll_member_mean"] = boot.ci(pf(nll_mem), n_f)
        out["subsets"][name] = s
        obs = [float((z[m] <= norm.ppf((1 + p) / 2)).mean()) for p in CURVE_LEVELS]
        inner = [abs(o - p) for o, p in zip(obs, CURVE_LEVELS) if p <= 0.95]
        out["calibration_curve"][name] = {"nominal": CURVE_LEVELS, "observed": obs,
                                          "miscalibration_area": float(np.mean(inner)),
                                          "max_shortfall": float(max(p - o for o, p in zip(obs, CURVE_LEVELS)))}
    if masks["le5"].any() and masks["gt5"].any():
        lo, hi = masks["le5"], masks["gt5"]
        v_lo, n_lo = boot.per_flight(var[lo].sum(1), lo), boot.per_flight(np.full(lo.sum(), 3.0), lo)
        v_hi, n_hi = boot.per_flight(var[hi].sum(1), hi), boot.per_flight(np.full(hi.sum(), 3.0), hi)
        pt = np.sqrt((v_hi.sum() / n_hi.sum()) / (v_lo.sum() / n_lo.sum()))
        a, b_ = n_hi[boot.idx].sum(1), n_lo[boot.idx].sum(1)
        ok = (a > 0) & (b_ > 0)
        reps = np.sqrt((v_hi[boot.idx].sum(1)[ok] / a[ok]) / (v_lo[boot.idx].sum(1)[ok] / b_[ok]))
        out["sigma_ratio_gt5_over_le5"] = [float(pt), *np.percentile(reps, [2.5, 97.5]).tolist()]
    sig = np.sqrt(var.mean(1))
    out["spearman_speed_sigma"] = float(spearmanr(speed, sig).statistic)
    out["spearman_abs_error_sigma"] = float(spearmanr(np.sqrt((e ** 2).mean(1)), sig).statistic)
    for lab, lo, hi in zip(Mx.speed_bin_labels(), Mx.SPEED_BINS[:-1], Mx.SPEED_BINS[1:]):
        m = (speed >= lo) & (speed < hi)
        if not m.any():
            continue
        row = {"n": int(m.sum()), "n_flights": int(len(np.unique(flights[m]))),
               "rmse": float(np.sqrt((e[m] ** 2).mean())),
               "sigma_rms": float(np.sqrt(var[m].mean())),
               "mean_sigma": float(sd[m].mean()),
               "mean_sigma_aleatoric": float(np.sqrt(alea[m]).mean()),
               "mean_sigma_epistemic": float(np.sqrt(epi[m]).mean()),
               "epistemic_share_of_var": float(epi[m].sum() / var[m].sum()),
               "nll_gauss": float(nll_g[m].mean()),
               **{f"coverage_{k}": float((z[m] <= zz).mean()) for k, zz in COVERAGE.items()}}
        if have_mix:
            row["nll_mixture"] = float(nll_mix[m].mean())
        out["by_speed"][lab] = row
    return out


def member_stats(Y, mp, members, speed, sl) -> dict:
    """Each NLL member alone (aleatoric only): sigma_rms, rmse, NLL, 2-sigma coverage (point values)."""
    out = {}
    for run, (m, v) in zip(members, mp):
        if v is None:
            continue
        e, var = m[:, sl] - Y[:, sl], v[:, sl]
        z = np.abs(e) / np.sqrt(var)
        r = {}
        for name, msk in (("all", np.ones(len(speed), bool)), ("le5", speed <= 5), ("gt5", speed > 5)):
            if msk.any():
                r[name] = {"rmse": float(np.sqrt((e[msk] ** 2).mean())), "sigma_rms": float(np.sqrt(var[msk].mean())),
                           "nll_gauss": float(gauss_nll(e[msk], var[msk]).mean()),
                           "coverage_2sigma": float((z[msk] <= 2).mean()), "coverage_90": float((z[msk] <= COVERAGE["90"]).mean())}
        out[run] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensembles", nargs="*", default=None, help="registry ensemble labels (default: all)")
    ap.add_argument("--out", default="results/uncertainty_val_final.json")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    t0 = time.time()
    from src.train import pick_device
    pc = PredCache(pick_device(a.device))
    reg = pc.reg
    labels = a.ensembles or list(reg.ensembles)
    ref_path = REPO / "results" / "val_uncertainty_final.json"
    ref = json.loads(ref_path.read_text())["ensembles"] if ref_path.exists() else {}
    out = {"source": "scripts/eval_uncertainty.py", "git": git_hash(), "registry": reg.info(),
           "splits_sha256": pc.splits["meta"]["sha256"], "device": str(pc.device),
           "definitions": {"coverage_z": COVERAGE, "coverage_nominal": NOMINAL, "curve_levels": CURVE_LEVELS,
                           "nll": "per-component Gaussian NLL in nats, physical units (N or N·m)",
                           "ci": "percentile flight bootstrap, 2000 resamples, seed 0"},
           "ensembles": {}, "checks": []}
    for lab in labels:
        split = reg.split_of(lab)
        out["ensembles"][lab] = {"split": split, "members": reg.ensemble_members()[lab], "eval": {}}
        for ev in EVAL_SETS[split]:
            rows = pc.rows(ev)
            ens = pc.ensemble(lab, ev)
            Y = rows["Y"].astype(np.float64)
            boot = FlightBoot(rows["flights"])
            res = {}
            for g, sl in GROUPS.items():
                res[g] = group_stats(Y, ens, rows["speed"], rows["flights"], sl, boot)
                res[g]["members_alone"] = member_stats(Y, ens["member_preds"], ens["members"], rows["speed"], sl)
            out["ensembles"][lab]["eval"][ev] = res
            # ---- print
            for g in GROUPS:
                s = res[g]["subsets"]
                for sub in ("all", "le5", "gt5"):
                    if sub not in s:
                        continue
                    x = s[sub]
                    mix = f" mix {x['nll_mixture'][0]:+.3f}" if "nll_mixture" in x else ""
                    print(f"[{lab} -> {ev}] {g} {sub:>3}: rmse {x['rmse'][0]:.4g} sigma_rms {x['sigma_rms'][0]:.4g} "
                          f"(alea {x['sigma_aleatoric_rms'][0]:.4g}, epi {x['sigma_epistemic_rms'][0]:.4g}, "
                          f"epi share {x['epistemic_share_of_var'][0]:.2f})  NLL {x['nll_gauss'][0]:+.3f}{mix}  "
                          f"cov2s {x['coverage']['2sigma'][0]:.3f} [{x['coverage']['2sigma'][1]:.3f},"
                          f"{x['coverage']['2sigma'][2]:.3f}]  cov90 {x['coverage']['90'][0]:.3f}  "
                          f"miscal {res[g]['calibration_curve'][sub]['miscalibration_area']:.3f}")
                r = res[g].get("sigma_ratio_gt5_over_le5")
                if r:
                    print(f"{'':>8}{g} sigma ratio gt5/le5 {r[0]:.2f} [{r[1]:.2f},{r[2]:.2f}]  "
                          f"rho(speed,sigma) {res[g]['spearman_speed_sigma']:.2f}")
                print(f"{'':>8}{g} mean sigma by speed: " + "  ".join(
                    f"{k} {v['mean_sigma']:.4g}" for k, v in res[g]["by_speed"].items()))
            # ---- consistency with eval_models.py --final
            old = ref.get(lab, {}).get("eval", {}).get(ev)
            if old:
                for g in GROUPS:
                    a1, b1 = res[g]["subsets"]["all"]["sigma_rms"][0], old[g]["subsets"]["all"]["sigma_rms"][0]
                    a2, b2 = res[g]["subsets"]["all"]["coverage"]["90"][0], old[g]["subsets"]["all"]["coverage"]["90"][0]
                    ok = abs(a1 - b1) <= 1e-4 * max(abs(b1), 1e-9) + 1e-7 and abs(a2 - b2) <= 1e-4
                    out["checks"].append({"ensemble": lab, "eval": ev, "group": g, "sigma_rms": [a1, b1],
                                          "coverage90": [a2, b2], "match": bool(ok)})
                    print(f"   check vs val_uncertainty_final.json {g}: sigma_rms {a1:.5g} vs {b1:.5g}, "
                          f"cov90 {a2:.4f} vs {b2:.4f} -> {'OK' if ok else 'MISMATCH'}")
    pc.save()
    out["runtime_s"] = round(time.time() - t0, 1)
    (REPO / a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out} ({out['runtime_s']} s)")
    if any(not c["match"] for c in out["checks"]):
        print("WARNING: at least one check MISMATCH - do not use these numbers before it is explained")


if __name__ == "__main__":
    main()
