"""CP3/CP4: fit P0a / P0b / B1 / B2 on train, score on validation; estimate the target noise floor.

    python scripts/eval_baselines.py         -> results/val_baselines.json, results/noise_floor.json

Splits and evaluation sets (validation only; test parts are never loaded):
  A   : fit on A/train,   score on A/val
  B_s : fit on B_s/train (|v| <= 5 m/s), score on B_s/val (slow) AND on A/val (all speeds: same
        flights as B_s/val, never in B_s/train) -> the speed-extrapolation view used for H1.
B2 (CP4): ridge on the 20-sample history of body velocity and rates only - a learned linear
differentiator, the reference for what windowed models can get without aerodynamics.
Metrics follow the paper's Table II definition (src/eval/metrics.py). CIs: 2000 flight resamples.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_part, read_splits  # noqa: E402
from src.data.load import REPO, load_all  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.eval.baselines import ALPHAS, ALPHAS_B2, ConstantModel, KinematicFIR, PhysicsModel, RidgePoly2  # noqa: E402
from src.eval.evaluate import paired, score, scored_rows, short  # noqa: E402
from src.eval.noise import noise_floor, noise_floor_by_speed  # noqa: E402

EVAL_SETS = {"A": [("A", "val")], "B_s": [("B_s", "val"), ("A", "val")]}


def std_mse(Y, P, var):
    return float((((P - Y) ** 2).mean(0) / var).mean())


def main():
    t0 = time.time()
    splits, df = read_splits(), load_all()
    out = {"source": "scripts/eval_baselines.py", "splits_sha256": splits["meta"]["sha256"],
           "metric_definition": "paper Table II: F = sqrt(mean over samples and x,y,z of e^2); "
                                "Fxy over x,y; Fz over z; same for M. F_vecnorm = sqrt(3)*F.",
           "scored_samples": "ends of 20-sample windows inside contact-trimmed pieces", "splits": {}}
    for key, sets in EVAL_SETS.items():
        tr = scored_rows(load_part(key, "train", splits, df))
        evals = {f"{k}_{p}": scored_rows(load_part(k, p, splits, df)) for k, p in sets}
        own_val = evals[f"{key}_val"]
        var = tr["Y"].astype(np.float64).var(0)
        p0a = ConstantModel().fit(tr["X"], tr["Y"])
        p0b = PhysicsModel().fit(tr["X"], tr["Y"])
        b1 = RidgePoly2().fit_stats(tr["X"], tr["Y"])
        sel = {}
        for a in ALPHAS:          # 7-value budget, chosen on this split's own validation part
            sel[a] = std_mse(own_val["Y"], b1.set_alpha(a).predict(own_val["X"]), var)
        best = min(sel, key=sel.get)
        b1.set_alpha(best)
        b2 = KinematicFIR().fit_stats_rows(tr)
        sel2 = {a: std_mse(own_val["Y"], b2.set_alpha(a).predict_rows(own_val), var) for a in ALPHAS_B2}
        best2 = min(sel2, key=sel2.get)
        b2.set_alpha(best2)
        res = {"n_train_scored": int(len(tr["Y"])), "n_train_flights": int(len(set(tr["flights"]))),
               "P0b_train_r2": p0b.r2, "P0b_coefs": [c.tolist() for c in p0b.coefs],
               "B1_alpha_selection_val_std_mse": {str(k): v for k, v in sel.items()}, "B1_alpha": best,
               "B2_alpha_selection_val_std_mse": {str(k): v for k, v in sel2.items()}, "B2_alpha": best2,
               "eval": {}}
        print(f"\n=== split {key}: train {len(tr['Y']):,} scored samples, {res['n_train_flights']} flights; "
              f"B1 alpha {best:g}, B2 alpha {best2:g}")
        for name, rows in evals.items():
            res["eval"][name] = {}
            for model in (p0a, p0b, b1):
                res["eval"][name][model.name] = sc = score(rows, model.predict(rows["X"]))
                print(f"[{key} -> {name}] {model.name:3s} {short(sc)}")
            res["eval"][name]["B2"] = sc = score(rows, b2.predict_rows(rows))
            print(f"[{key} -> {name}] B2  {short(sc)}")
            for model in (p0b, b1, b2):
                sc = res["eval"][name][model.name]
                if sc["gt5"]["n"]:
                    print(f"[{key} -> {name}] {model.name:3s} |v|>5 only: {short(sc, 'gt5')}")
            # paired: B1 vs P0b, P0b vs P0a
            e = res["eval"][name]
            res["eval"][name]["paired"] = {"B1_minus_P0b": paired(e["B1"], e["P0b"]),
                                           "P0b_minus_P0a": paired(e["P0b"], e["P0a"]),
                                           "B2_minus_B1": paired(e["B2"], e["B1"])}
            sp = {lab: round(v.get("F", float("nan")), 3) for lab, v in e["P0b"]["by_speed"].items()}
            print(f"[{key} -> {name}] P0b F by speed bin: {sp}")
        out["splits"][key] = res

    # noise floor on A train and A val
    nf = {"source": "scripts/eval_baselines.py", "method": "src/eval/noise.py (diff and 9-sample high-pass, "
          "white-noise assumption -> lower reference, see module docstring)"}
    for p in ("train", "val"):
        pcs = load_part("A", p, splits, df)
        nf[f"A_{p}"] = {"overall": noise_floor(pcs), "by_speed": noise_floor_by_speed(pcs, M.SPEED_BINS)}
        o = nf[f"A_{p}"]["overall"]
        print(f"noise floor A/{p}: diff F {o['diff']['F']:.3f} N M {o['diff']['M']:.4f} N·m | "
              f"hp9 F {o['hp9']['F']:.3f} N M {o['hp9']['M']:.4f} N·m")
        print("   hp9 F by speed:", {k: round(v["hp9"]["F"], 3) for k, v in nf[f"A_{p}"]["by_speed"].items() if v})

    out["runtime_s"] = round(time.time() - t0, 1)
    (REPO / "results" / "val_baselines.json").write_text(json.dumps(out, indent=1))
    (REPO / "results" / "noise_floor.json").write_text(json.dumps(nf, indent=1))
    print(f"\nwrote results/val_baselines.json, results/noise_floor.json ({out['runtime_s']} s)")


if __name__ == "__main__":
    main()
