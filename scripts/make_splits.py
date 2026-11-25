"""Flight-level train/val/test splits -> data/splits.json + results/splits_summary.json.

    python scripts/make_splits.py            # writes both files, prints sizes and speed ranges
    python scripts/make_splits.py --check    # rebuilds and fails if data/splits.json differs

Unit of splitting = flight (all segments of a flight go to the same part).

Excluded before splitting (EXCLUDED_FLIGHTS below, with reasons): flights not logged at 400 Hz,
because a 20-sample window would then not be 50 ms.

Split A (70/20/10 by flight, stratified by trajectory type and max speed):
  1. Sort flights by (trajectory type, max speed). Walk the list in blocks of 10 and draw one
     test flight per block (the last, short block draws one with probability len/10).
  2. Sort the remaining pool the same way, walk it in blocks of 9 and shuffle each block.
     Block positions {0,1} -> val of "A"; {2,3} -> "A_r1"; {4,5} -> "A_r2"; {6,7} -> "A_r3".
     So A, A_r1, A_r2, A_r3 share one test set and have four DISJOINT val folds
     (group k-fold over flights on the non-test pool, stratified by the sort order).
Split B (speed extrapolation): train/val = flights whose max speed <= 5 m/s (val = 1 flight,
  drawn at random), test = every other flight.
Split B_s (sample-level variant, offered as an alternative to B because only 6 flights are slow):
  same flights as A; train and val keep only samples with |v| <= 5 m/s (cut into pieces, so
  windows stay inside the slow region); test = A's test flights, all speeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import frames as F  # noqa: E402
from src.data.dataset import frame_pieces  # noqa: E402
from src.data.load import REPO, load_all, segment_index  # noqa: E402
from src.data.targets import contact_mask  # noqa: E402

SEED = 520
B_MAX_SPEED = 5.0
WINDOW = 20
SPLITS_PATH = REPO / "data" / "splits.json"
SUMMARY_PATH = REPO / "results" / "splits_summary.json"
NOMINAL_DT = 0.0025


def flight_table(df: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    speed = np.linalg.norm(F.vec(df, "vel"), axis=1)
    g = pd.DataFrame({"flight_id": df["flight_id"].astype(str).to_numpy(), "speed": speed})
    agg = g.groupby("flight_id").speed.agg(max_speed="max", n_samples="size")
    dts = {f: np.median(np.concatenate([np.diff(s.t.to_numpy()) for _, s in d.groupby("segment_id", observed=True)]))
           for f, d in df.groupby("flight_id", observed=True)}
    meta = idx.groupby("flight_id").agg(trajectory_type=("trajectory_type", "first"),
                                        n_segments=("segment_id", "size"),
                                        n_server_test_segments=("in_server_testset", "sum"))
    t = meta.join(agg)
    t["median_dt"] = pd.Series(dts)
    return t.reset_index()


def excluded_flights(ft: pd.DataFrame) -> dict[str, str]:
    bad = ft[(ft.median_dt - NOMINAL_DT).abs() > 2e-4]
    return {r.flight_id: f"median dt {r.median_dt * 1e3:.2f} ms (not 400 Hz): a 20-sample window "
                         f"would span {WINDOW * r.median_dt * 1e3:.0f} ms" for r in bad.itertuples()}


def sorted_flights(ft: pd.DataFrame) -> list[str]:
    return ft.sort_values(["trajectory_type", "max_speed", "flight_id"]).flight_id.tolist()


def split_a(ft: pd.DataFrame, rng: np.random.Generator) -> dict[str, dict[str, list[str]]]:
    order = sorted_flights(ft)
    test = []
    for i in range(0, len(order), 10):
        block = order[i:i + 10]
        if len(block) == 10 or rng.random() < len(block) / 10:
            test.append(block[rng.integers(len(block))])
    pool = [f for f in order if f not in set(test)]
    folds = {k: [] for k in range(4)}
    for i in range(0, len(pool), 9):
        block = pool[i:i + 9]
        perm = rng.permutation(9)                      # position p in the block -> slot perm[p]
        for p, f in enumerate(block):
            if perm[p] < 8:
                folds[perm[p] // 2].append(f)
    out = {}
    for k, key in enumerate(["A", "A_r1", "A_r2", "A_r3"]):
        val = sorted(folds[k])
        out[key] = {"train": sorted(set(pool) - set(val)), "val": val, "test": sorted(test)}
    return out


def split_b(ft: pd.DataFrame, rng: np.random.Generator) -> dict[str, list[str]]:
    slow = sorted(ft.loc[ft.max_speed <= B_MAX_SPEED, "flight_id"])
    fast = sorted(ft.loc[ft.max_speed > B_MAX_SPEED, "flight_id"])
    val = [slow[rng.integers(len(slow))]]
    return {"train": sorted(set(slow) - set(val)), "val": val, "test": fast}


def to_segments(flights_by_part: dict, idx: pd.DataFrame) -> dict:
    seg = idx.groupby("flight_id").segment_id.apply(sorted).to_dict()
    return {p: sorted(s for f in fl for s in seg[f]) for p, fl in flights_by_part.items()}


def part_stats(df: pd.DataFrame, ft: pd.DataFrame, flights: list[str], filt: dict | None) -> dict:
    sub = df[df.flight_id.astype(str).isin(set(flights))]
    pieces = frame_pieces(sub, filt)
    n_kept = sum(len(p) for _, p in pieces)
    n_windows = sum(max(0, len(p) - WINDOW + 1) for _, p in pieces)
    sp_kept = np.concatenate([np.linalg.norm(F.vec(p, "vel"), axis=1) for _, p in pieces]) if pieces else np.zeros(1)
    f = ft.set_index("flight_id").loc[flights]
    return {
        "n_flights": len(flights), "n_segments": int(f.n_segments.sum()), "n_pieces": len(pieces),
        "n_samples_raw": int(len(sub)), "n_contact_samples_cut": int(contact_mask(sub).sum()),
        "n_samples_kept": int(n_kept), "n_windows": int(n_windows),
        "minutes_kept": round(n_kept * NOMINAL_DT / 60, 2),
        "flight_max_speed_min_max": [round(float(f.max_speed.min()), 2), round(float(f.max_speed.max()), 2)],
        "sample_speed_p50_p99_max": [round(float(x), 2) for x in np.percentile(sp_kept, [50, 99, 100])],
        "trajectory_types": {k: int(v) for k, v in f.trajectory_type.value_counts().sort_index().items()},
        "n_server_testset_segments": int(f.n_server_test_segments.sum()),
    }


def build(df: pd.DataFrame | None = None, idx: pd.DataFrame | None = None) -> tuple[dict, dict]:
    idx = segment_index() if idx is None else idx
    df = load_all() if df is None else df
    ft = flight_table(df, idx)
    excl = excluded_flights(ft)
    ft_use = ft[~ft.flight_id.isin(excl)].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    flights = split_a(ft_use, rng)
    flights["B"] = split_b(ft_use, rng)
    flights["B_s"] = {p: list(v) for p, v in flights["A"].items()}
    filters = {"B_s": {"train": {"max_speed": B_MAX_SPEED}, "val": {"max_speed": B_MAX_SPEED}}}

    splits = {"meta": {"seed": SEED, "unit": "flight", "window": WINDOW, "b_max_speed": B_MAX_SPEED,
                       "excluded_flights": excl, "contact_rule": "mean motor < 700 rad/s and |v| < 0.3 m/s",
                       "made_by": "scripts/make_splits.py"}}
    for key, parts in flights.items():
        splits[key] = to_segments(parts, idx)
        splits[key]["flights"] = parts
        if key in filters:
            splits[key]["filter"] = filters[key]
    body = json.dumps({k: v for k, v in splits.items() if k != "meta"}, sort_keys=True)
    splits["meta"]["sha256"] = hashlib.sha256(body.encode()).hexdigest()[:16]

    summary = {"source": "scripts/make_splits.py", "splits_sha256": splits["meta"]["sha256"],
               "excluded_flights": excl, "n_flights_used": len(ft_use),
               "n_samples_used": int(ft_use.n_samples.sum()), "splits": {}}
    for key, parts in flights.items():
        summary["splits"][key] = {p: part_stats(df, ft, fl, filters.get(key, {}).get(p)) for p, fl in parts.items()}
        tot = sum(s["n_samples_kept"] for s in summary["splits"][key].values())
        for s in summary["splits"][key].values():
            s["pct_samples_of_split"] = round(100 * s["n_samples_kept"] / tot, 1)
    return splits, summary


def print_summary(summary: dict) -> None:
    print(f"flights used {summary['n_flights_used']}, samples {summary['n_samples_used']:,}; "
          f"excluded {list(summary['excluded_flights'])}")
    hdr = f"{'split':6} {'part':5} {'flights':>7} {'segs':>5} {'pieces':>6} {'samples':>9} {'%':>5} " \
          f"{'windows':>9} {'min':>6} {'cut':>4}  {'flight vmax range':>17}  {'|v| p50/p99/max':>17}  srv"
    print(hdr); print("-" * len(hdr))
    for key, parts in summary["splits"].items():
        for p, s in parts.items():
            lo, hi = s["flight_max_speed_min_max"]
            print(f"{key:6} {p:5} {s['n_flights']:7d} {s['n_segments']:5d} {s['n_pieces']:6d} {s['n_samples_kept']:9,d} "
                  f"{s['pct_samples_of_split']:5.1f} {s['n_windows']:9,d} {s['minutes_kept']:6.1f} "
                  f"{s['n_contact_samples_cut']:4d}  {lo:6.2f} - {hi:6.2f}  "
                  f"{'/'.join(f'{x:.2f}' for x in s['sample_speed_p50_p99_max']):>17}  {s['n_server_testset_segments']:3d}")
    print("trajectory types per part (A):")
    for p, s in summary["splits"]["A"].items():
        print(f"  {p:5} {s['trajectory_types']}")
    print("Split B parts:", {p: s["trajectory_types"] for p, s in summary["splits"]["B"].items() if p != "test"})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if data/splits.json differs from a rebuild")
    a = ap.parse_args(argv)
    splits, summary = build()
    if a.check:
        old = json.loads(SPLITS_PATH.read_text())
        if old["meta"]["sha256"] != splits["meta"]["sha256"]:
            raise SystemExit("data/splits.json differs from a rebuild")
        print("data/splits.json matches a rebuild:", old["meta"]["sha256"])
        return
    SPLITS_PATH.write_text(json.dumps(splits, indent=1))
    SUMMARY_PATH.write_text(json.dumps(summary, indent=1))
    print_summary(summary)
    print(f"wrote {SPLITS_PATH.relative_to(REPO)} and {SUMMARY_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
