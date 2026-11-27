"""CP4: summarise the Thursday sweep, pick one config per model, write the tuned configs.

    python scripts/sweep_summary.py            -> results/sweep_thu.json + configs/*_tuned.yaml
    python scripts/sweep_summary.py --dry-run  (table only)

Candidates per model: the Wednesday seed-0 run (sw0 = configs/<model>.yaml) and <model>_sw1..sw7
(scripts/sweep_thu.sh), all Split A, seed 0. Selection = lowest best_val_loss in results/runs.csv
(standardised MSE on A/val, the training loss; force and moment weigh equally). Same rule for every
model. The chosen settings are then reused unchanged for Split B_s (B_s/val holds only slow samples,
so tuning on it could not see extrapolation), for the NLL ensemble members and for M3act.

Every TCN-based tuned config gets train.max_epochs >= 150 (the Wed ensemble members hit the 100-epoch
cap with the LR already at 2-3e-5); patience stays as chosen, so early stopping still decides.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train import load_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODELS = ["m1_mlp", "m2_hybrid_mlp", "m3_hybrid_tcn"]
IGNORE = {"name", "seed", "log"}


def flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if pre == "" and k in IGNORE:
            continue
        key = f"{pre}{k}"
        out.update(flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def overrides(cfg: dict, base: dict) -> dict:
    a, b = flat(cfg), flat(base)
    return {k: a[k] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def dump(cfg: dict, path: Path, header: str) -> None:
    body = {k: cfg[k] for k in ("name", "model", "data", "train") if k in cfg}
    path.write_text(header + yaml.safe_dump(body, sort_keys=False, default_flow_style=None, width=110))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--metric", choices=["val_loss", "paper"], default="val_loss",
                    help="val_loss: standardised MSE (training loss). paper: F/F_default + M/M_default in "
                         "paper units (runs.csv vector norms are sqrt(3) x paper RMSE, so ratios are equal)")
    ap.add_argument("--allow-incomplete", action="store_true", help="write tuned configs with < 8 configs")
    a = ap.parse_args(argv)
    rows = list(csv.DictReader((REPO / "results" / "runs.csv").open()))
    out = {"source": "scripts/sweep_summary.py", "selection": f"lowest {a.metric} on A/val, seed 0",
           "models": {}}
    for m in MODELS:
        base = load_config(REPO / "configs" / f"{m}.yaml")
        names = [m] + [f"{m}_sw{k}" for k in range(1, 8)]
        cands = []
        for nm in names:
            hits = [r for r in rows if r["name"] == nm and r["seed"] == "0"]
            if not hits:
                print(f"  missing: {nm}")
                continue
            r = hits[-1]                                   # latest row if a config was re-run
            cfg = json.loads(r["config_json"])
            if cfg["data"]["split"] != "A":
                continue
            cands.append({"name": nm, "run": r["run"], "overrides": overrides(cfg, base),
                          "best_val_loss": float(r["best_val_loss"]), "best_epoch": int(r["best_epoch"]),
                          "epochs_run": int(r["epochs_run"]), "stop_reason": r["stop_reason"],
                          "val_rmse_F_vecnorm_N": float(r["val_rmse_F_N"]), "val_rmse_M_vecnorm_Nm": float(r["val_rmse_M_Nm"]),
                          "wall_time_s": float(r["wall_time_s"]), "git_hash": r["git_hash"], "git_dirty": r["git_dirty"],
                          "config": cfg})
        if not cands:
            continue
        ref = next((c for c in cands if c["name"] == m), None)
        for c in cands:
            c["paper_score"] = (c["val_rmse_F_vecnorm_N"] / ref["val_rmse_F_vecnorm_N"]
                                + c["val_rmse_M_vecnorm_Nm"] / ref["val_rmse_M_vecnorm_Nm"]) if ref else None
        key = "best_val_loss" if a.metric == "val_loss" else "paper_score"
        cands.sort(key=lambda c: c[key])
        best = cands[0]
        print(f"\n{m}: {len(cands)} of 8 configs")
        print(f"  {'config':<22}{'val loss':>10}{'paper':>7}{'F vec':>8}{'M vec':>9}{'best ep':>8}{'epochs':>7}  overrides")
        for c in cands:
            ps = f"{c['paper_score']:>7.3f}" if c["paper_score"] is not None else f"{'-':>7}"
            print(f"  {c['name']:<22}{c['best_val_loss']:>10.5f}{ps}{c['val_rmse_F_vecnorm_N']:>8.4f}"
                  f"{c['val_rmse_M_vecnorm_Nm']:>9.5f}{c['best_epoch']:>8d}{c['epochs_run']:>7d}  "
                  f"{c['overrides'] or '(default)'}")
        gain = (1 - best["best_val_loss"] / ref["best_val_loss"]) * 100 if ref else None
        print(f"  -> chosen {best['name']}" + (f" ({gain:+.1f}% val loss vs default)" if gain is not None else ""))
        if len(cands) < 8 and not a.allow_incomplete and not a.dry_run:
            print(f"  !! only {len(cands)} of 8 configs - not writing tuned configs (re-run scripts/sweep_thu.sh)")
            a.dry_run_model = True
        out["models"][m] = {"n_configs": len(cands), "metric": a.metric, "chosen": best["name"], "chosen_overrides": best["overrides"],
                            "val_loss_gain_vs_default_pct": gain,
                            "candidates": [{k: v for k, v in c.items() if k != "config"} for c in cands]}
        if a.dry_run or getattr(a, "dry_run_model", False):
            a.dry_run_model = False
            continue
        cfg = {k: v for k, v in best["config"].items() if k not in ("seed", "log")}
        hdr = (f"# Written by scripts/sweep_summary.py from {best['name']} (lowest A/val loss of {len(cands)} configs).\n"
               f"# Overrides vs configs/{m}.yaml: {best['overrides'] or 'none'}\n")
        is_tcn = cfg["model"]["type"] == "tcn"
        if is_tcn:
            cfg["train"]["max_epochs"] = max(150, int(cfg["train"]["max_epochs"]))
        variants = {f"{m}_tuned": {}, f"{m}_Bs_tuned": {"data.split": "B_s"}}
        if is_tcn:
            variants.update({"m3_ens_tuned": {"model.nll": True}, "m3_ens_Bs_tuned": {"model.nll": True, "data.split": "B_s"},
                             "m3_hybrid_tcn_act_tuned": {"model.history": "actuator"},
                             "m3_hybrid_tcn_act_Bs_tuned": {"model.history": "actuator", "data.split": "B_s"}})
        for vname, ch in variants.items():
            c = json.loads(json.dumps(cfg))
            c["name"] = vname
            for k, v in ch.items():
                sec, key = k.split(".")
                c[sec][key] = v
            dump(c, REPO / "configs" / f"{vname}.yaml", hdr + (f"# Variant: {ch}\n" if ch else ""))
            print(f"  wrote configs/{vname}.yaml")
    if not a.dry_run:
        (REPO / "results" / "sweep_thu.json").write_text(json.dumps(out, indent=1))
        print("\nwrote results/sweep_thu.json")


if __name__ == "__main__":
    main()
