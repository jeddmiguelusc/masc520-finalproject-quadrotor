"""CP3/CP4: score trained checkpoints on validation with the paper's metrics, CIs and speed bins.

    python scripts/eval_models.py                       # the Wed runs (default list below)
    python scripts/eval_models.py --runs m3_hybrid_tcn_seed0 m2_hybrid_mlp_seed0
    -> results/val_models.json  (needs results/val_baselines.json for paired comparisons)

Each run's checkpoint (checkpoints/<run>/best.pt) carries its config; the split decides the
evaluation sets exactly as in scripts/eval_baselines.py:
  A   -> A/val
  B_s -> B_s/val (slow only) and A/val (all speeds; |v|>5 subset = speed-extrapolation view, H1)
Validation only. Sanity check: F_vecnorm here must match val_rmse_F_N in results/runs.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_part, read_splits  # noqa: E402
from src.data.load import REPO, load_all  # noqa: E402
from src.eval.evaluate import paired, score, scored_rows, short  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train import pick_device  # noqa: E402

DEFAULT_RUNS = ["m1_mlp_seed0", "m2_hybrid_mlp_seed0", "m3_hybrid_tcn_seed0",
                "m1_mlp_Bs_seed0", "m2_hybrid_mlp_Bs_seed0"]
EVAL_SETS = {"A": [("A", "val")], "B_s": [("B_s", "val"), ("A", "val")]}
LABEL = {"m1_mlp": "M1", "m2_hybrid_mlp": "M2", "m3_hybrid_tcn": "M3", "m4_hybrid_transformer": "M4",
         "m1_mlp_Bs": "M1", "m2_hybrid_mlp_Bs": "M2", "m3_hybrid_tcn_Bs": "M3"}
# (split, eval set, a, b, subset): metric(a) - metric(b); negative = a better
PAIRS = [("A", "A_val", "M2", "M1", "all"), ("A", "A_val", "M3", "M2", "all"), ("A", "A_val", "M1", "B1", "all"),
         ("A", "A_val", "M2", "P0b", "all"), ("A", "A_val", "M3", "B1", "all"),
         ("A", "A_val", "M2", "M1", "gt5"), ("A", "A_val", "M3", "M2", "gt5"),
         ("B_s", "B_s_val", "M2", "M1", "all"),
         ("B_s", "A_val", "M2", "M1", "all"), ("B_s", "A_val", "M2", "M1", "gt5"),
         ("B_s", "A_val", "M2", "P0b", "gt5"), ("B_s", "A_val", "M1", "B1", "gt5")]


@torch.no_grad()
def predict(model, rows: dict, window: int, device, batch: int = 8192) -> np.ndarray:
    ws = rows["ws"]
    X = torch.from_numpy(ws.X).to(device)
    offs = torch.arange(-window + 1, 1, device=device)
    ends = torch.from_numpy(ws.ends).long().to(device)
    out = []
    for a in range(0, len(ends), batch):
        idx = ends[a:a + batch, None] + offs[None, :]
        mean, _ = model(X[idx])
        out.append(mean.float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    device = pick_device(a.device)
    splits, df = read_splits(), load_all()
    base = json.loads((REPO / "results" / "val_baselines.json").read_text())
    runs_csv = {r["run"]: r for r in csv.DictReader((REPO / "results" / "runs.csv").open())}
    cache, out = {}, {"source": "scripts/eval_models.py", "splits_sha256": splits["meta"]["sha256"],
                      "metric_definition": base["metric_definition"], "runs": {}, "paired": []}
    table = {}                         # (split, eval set, label) -> score
    for run in a.runs:
        ck = torch.load(REPO / "checkpoints" / run / "best.pt", map_location=device, weights_only=False)
        cfg = ck["config"]
        key = cfg["data"]["split"]
        model = build_model(cfg["model"]); model.load_state_dict(ck["model"]); model.to(device).eval()
        label = LABEL.get(cfg.get("name", ""), cfg.get("name", run))
        res = {"split": key, "label": label, "config_name": cfg.get("name"), "eval": {}}
        for k, p in EVAL_SETS[key]:
            name = f"{k}_{p}"
            if (k, p) not in cache:
                cache[(k, p)] = scored_rows(load_part(k, p, splits, df))
            rows = cache[(k, p)]
            P = predict(model, rows, int(cfg["data"]["max_window"]), device)
            res["eval"][name] = sc = score(rows, P)
            table[(key, name, label)] = sc
            print(f"[{run} -> {name}] {short(sc)}")
            if sc["gt5"]["n"]:
                print(f"{'':>{len(run) + len(name) + 6}} |v|>5: {short(sc, 'gt5')}")
            if name == f"{key}_val" and run in runs_csv:
                csv_f = float(runs_csv[run]["val_rmse_F_N"]); here = sc["all"]["ci95"]["F_vecnorm"][0]
                res["check_vs_runs_csv"] = {"runs_csv_val_rmse_F_N": csv_f, "F_vecnorm_here": here,
                                            "match": bool(abs(csv_f - here) < 1e-3)}
                print(f"   check: runs.csv F {csv_f:.4f} vs F_vecnorm {here:.4f} "
                      f"{'OK' if abs(csv_f - here) < 1e-3 else 'MISMATCH'}")
        out["runs"][run] = res
    for key, name, la, lb, sub in PAIRS:
        def get(lab):
            if (key, name, lab) in table:
                return table[(key, name, lab)]
            return base["splits"][key]["eval"].get(name, {}).get(lab)
        A_, B_ = get(la), get(lb)
        if A_ is None or B_ is None:
            continue
        d = paired(A_, B_, sub)
        out["paired"].append({"split": key, "eval": name, "a": la, "b": lb, "subset": sub, "diff_ci95_p": d})
        print(f"paired [{key} -> {name}, {sub}] {la} - {lb}: F {d['F'][0]:+.3f} [{d['F'][1]:+.3f},{d['F'][2]:+.3f}] N "
              f"(p={d['F'][3]:.3f})  M {d['M'][0]:+.4f} [{d['M'][1]:+.4f},{d['M'][2]:+.4f}] (p={d['M'][3]:.3f})")
    (REPO / "results" / "val_models.json").write_text(json.dumps(out, indent=1))
    print("wrote results/val_models.json")


if __name__ == "__main__":
    main()
