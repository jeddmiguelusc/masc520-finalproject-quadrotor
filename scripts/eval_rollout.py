"""CP6: open-loop rollout evaluation on A/val (src/sim/rollout.py).

    python scripts/eval_rollout.py                      # default model set, MPS/CPU auto
    python scripts/eval_rollout.py --models P0b m2_hybrid_mlp_tuned m3_hybrid_tcn --stride 400
    python scripts/eval_rollout.py --final --out results/val_rollout_final.json   # labels from configs/final_models.yaml
    -> results/val_rollout.json (+ printed table)

Start points: every --stride samples (default 200 = 0.5 s) inside each contact-trimmed A/val piece
that has 19 recorded samples before and --steps after (default 100 = 0.25 s). Each rollout starts
from the true state, is driven by the recorded motor speeds / dmot / vbat, and the model sees the
usual 20-sample window whose state channels come from the simulation (src/sim/rollout.py).

Model names: "P0b" (coefficients from results/val_baselines.json), "oracle" (recorded F and M; checks
the integrator and the data's own consistency), a config name whose seeds 0..2 exist under
checkpoints/<name>_seed<k>/ (each seed rolled out separately; mean ± sd over seeds reported), or
"ens:<name>" = the Gaussian-mixture mean of seeds 0..2 used as one force model.

The "oracle" rows are the floor: the recorded F/M disagree with the recorded motion by ~0.5 N at speed
(B2's finite-difference error), so no model can track the data better than the oracle does.

Per model x mode (translational, full) x integrator (rk4, euler), at 0.1 s and 0.25 s:
  pos / vel (world) / att / body-rate error: RMS over start points, median, and a 95% flight-bootstrap
  CI of the RMS and of the median; the same split by start speed (<= 5 and > 5 m/s); count of diverged
  rollouts. Failure policy (src/eval/rollout_stats.py): an error that is non-finite or > 10 m counts as
  diverged and is set to 10 m in every statistic, the paired differences included. Paired flight-bootstrap differences of the 0.25 s position
  MSE against M2 (seed-matched where both sides have seeds).
Validation only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import frame_pieces, read_splits, segment_ids  # noqa: E402
from src.data.load import REPO, load_all  # noqa: E402
from src.models import build_model  # noqa: E402
from src.eval.registry import load_registry  # noqa: E402
from src.eval.rollout_stats import DIVERGED_M, diverged, paired_diff, summarise_errors  # noqa: E402
from src.sim import rollout as R  # noqa: E402
from src.train import pick_device  # noqa: E402

DEFAULT_MODELS = ["oracle", "P0b", "m1_mlp_tuned", "m2_hybrid_mlp", "m2_hybrid_mlp_tuned", "m3_hybrid_tcn",
                  "m3_hybrid_tcn_tuned", "m3_hybrid_tcn_act", "m3_hybrid_tcn_act_tuned", "m4_hybrid_transformer",
                  "ens:m3_ens", "ens:m3_ens_tuned"]
HORIZONS = {"0.1s": 40, "0.25s": 100}


def load_episodes(stride: int, steps: int, device, max_starts: int | None = None):
    splits = read_splits()
    ids = set(segment_ids("A", "val", splits))
    df = load_all()
    sub = df[df["segment_id"].astype(str).isin(ids)]
    eps, flights, speeds = [], [], []
    for pid, piece in frame_pieces(sub):
        n = len(piece)
        starts = list(range(R.WINDOW - 1, n - steps - 1, stride))
        if not starts:
            continue
        try:
            ep = R.Episodes.from_piece(piece, starts, steps)
        except ValueError:                       # time gap inside a piece: keep the clean starts only
            ok = []
            for s in starts:
                try:
                    R.Episodes.from_piece(piece, [s], steps); ok.append(s)
                except ValueError:
                    pass
            if not ok:
                continue
            ep = R.Episodes.from_piece(piece, ok, steps)
        eps.append(ep)
        fl = pid.split("_seg_")[0]
        flights += [fl] * ep.feats.shape[0]
        speeds += ep.feats[:, ep.H, R.V].norm(dim=-1).tolist()
    cat = lambda k: torch.cat([getattr(e, k) for e in eps])  # noqa: E731
    ep = R.Episodes(cat("feats"), cat("q"), cat("pos"), cat("Y"), eps[0].H, R.DT).to(device=device)
    flights, speeds = np.array(flights), np.array(speeds)
    if max_starts and len(flights) > max_starts:
        keep = np.linspace(0, len(flights) - 1, max_starts).astype(int)
        sl = torch.as_tensor(keep, device=device)
        ep = R.Episodes(ep.feats[sl], ep.q[sl], ep.pos[sl], ep.Y[sl], ep.H, ep.dt)
        flights, speeds = flights[keep], speeds[keep]
    return ep, flights, speeds


def load_model(run: str, device):
    ck = torch.load(REPO / "checkpoints" / run / "best.pt", map_location=device, weights_only=False)
    m = build_model(ck["config"]["model"]); m.load_state_dict(ck["model"])
    return m.to(device).eval()


class EnsembleMean:
    def __init__(self, models):
        self.models = models

    def __call__(self, win, k, tau):
        return torch.stack([m(win)[0] for m in self.models]).mean(0)


def force_sources(name: str, ep, device, reg=None) -> list[tuple[int | None, object]]:
    """[(seed or None, force callable)] for one model name (or, with a registry, one label)."""
    if reg is not None and name not in ("oracle", "P0b"):
        def checked(run):
            ck = torch.load(REPO / "checkpoints" / run / "best.pt", map_location=device, weights_only=False)
            reg.check_checkpoint(run, ck["config"], "A")
            m = build_model(ck["config"]["model"]); m.load_state_dict(ck["model"])
            return int(ck["config"].get("seed", 0)), m.to(device).eval()
        if reg.is_ensemble(name):
            return [(None, EnsembleMean([checked(r)[1] for r in reg.ensembles[name]["members"]]))]
        return [(s_, R.ModelForce(m)) for s_, m in (checked(r) for r in reg.models[name]["runs"])]
    if name == "oracle":
        return [(None, R.RecordedForce(ep))]
    if name == "P0b":
        coefs = json.loads((REPO / "results" / "val_baselines.json").read_text())["splits"]["A"]["P0b_coefs"]
        return [(None, R.ModelForce(R.physics_model_from_coefs(coefs).to(device)))]
    seeds = [s for s in range(3) if (REPO / "checkpoints" / f"{name.removeprefix('ens:')}_seed{s}" / "best.pt").exists()]
    if not seeds:
        raise FileNotFoundError(f"no checkpoints for {name}")
    if name.startswith("ens:"):
        base = name.removeprefix("ens:")
        return [(None, EnsembleMean([load_model(f"{base}_seed{s}", device) for s in seeds]))]
    return [(s, R.ModelForce(load_model(f"{name}_seed{s}", device))) for s in seeds]


def run_batched(force, ep, steps, integ, mode, batch):
    errs = {k: [] for k in ("pos_err", "vel_err", "att_err", "w_err")}
    B = ep.feats.shape[0]
    with torch.no_grad():
        for a in range(0, B, batch):
            sl = slice(a, min(a + batch, B))
            sub = R.Episodes(ep.feats[sl], ep.q[sl], ep.pos[sl], ep.Y[sl], ep.H, ep.dt)
            if isinstance(force, R.RecordedForce):
                force = R.RecordedForce(sub)
            o = R.rollout(force, sub, steps, integ, mode)
            for k in errs:
                errs[k].append(o[k].cpu().double().numpy())  # to CPU first: MPS has no float64
    return {k: np.concatenate(v) for k, v in errs.items()}           # (B, steps+1)


def summarise(err: np.ndarray, flights, speeds) -> dict:
    """Per subset (all, <= 5, > 5 m/s at the start): RMS, median, p90, flight-bootstrap CIs and the
    diverged count, under the shared failure policy (src/eval/rollout_stats.py: non-finite or > 10 m
    counts as diverged and is set to 10 m in every statistic)."""
    out = {}
    for sub, m in (("all", np.ones(len(err), bool)), ("le5", speeds <= 5), ("gt5", speeds > 5)):
        if m.any():
            out[sub] = summarise_errors(err[m], flights[m])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--stride", type=int, default=200)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--max-starts", type=int, default=None, help="subsample start points (quick check)")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--modes", nargs="*", default=["translational", "full"])
    ap.add_argument("--integrators", nargs="*", default=["rk4", "euler"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ref", default="m2_hybrid_mlp_tuned", help="reference model for paired differences")
    ap.add_argument("--out", default="results/val_rollout.json")
    ap.add_argument("--final", action="store_true", help="labels and runs from configs/final_models.yaml")
    ap.add_argument("--registry", default=None)
    a = ap.parse_args(argv)
    reg = (load_registry(a.registry) if a.registry else load_registry()) if a.final else None
    if reg is not None:
        a.models = ["oracle", "P0b"] + reg.labels("A")
        a.ref = reg.rollout_reference
    device = pick_device(a.device)
    t0 = time.time()
    ep, flights, speeds = load_episodes(a.stride, a.steps, device, a.max_starts)
    fl = np.unique(flights)
    print(f"{ep.feats.shape[0]} start points in {len(fl)} A/val flights ({(speeds > 5).sum()} start > 5 m/s), "
          f"{a.steps} steps = {a.steps * R.DT:.3f} s, device {device}  [{time.time() - t0:.0f}s]")
    out = {"source": "scripts/eval_rollout.py", "splits_sha256": read_splits()["meta"]["sha256"],
           "stride": a.stride, "steps": a.steps, "dt": R.DT, "n_starts": int(ep.feats.shape[0]),
           "n_flights": int(len(fl)), "n_starts_gt5": int((speeds > 5).sum()), "device": str(device),
           "diverged_threshold_m": DIVERGED_M,
           "failure_policy": f"non-finite or > {DIVERGED_M:g} m = diverged, set to {DIVERGED_M:g} m in every statistic",
           "registry": reg.info() if reg is not None else None, "models": {}, "paired_pos_rmse_final": []}
    pos_final = {}                     # (name, seed, mode, integ) -> pos err at the final step (raw)
    for name in a.models:
        try:
            sources = force_sources(name, ep, device, reg)
        except FileNotFoundError as e:
            print(f"skip {name}: {e}")
            continue
        res = out["models"][name] = {}
        for mode in a.modes:
            for integ in a.integrators:
                key = f"{mode}/{integ}"
                per_seed = []
                for seed, f in sources:
                    t1 = time.time()
                    errs = run_batched(f, ep, a.steps, integ, mode, a.batch)
                    div = diverged(errs["pos_err"][:, -1])
                    ent = {"seed": seed, "diverged": int(div.sum()), "horizons": {}}
                    for hname, h in HORIZONS.items():
                        if h > a.steps:
                            continue
                        ent["horizons"][hname] = {k.replace("_err", ""): summarise(errs[k][:, h], flights, speeds)
                                                  for k in ("pos_err", "vel_err", "att_err", "w_err")
                                                  if not (mode == "translational" and k in ("att_err", "w_err"))}
                    pos_final[(name, seed, mode, integ)] = errs["pos_err"][:, a.steps]
                    per_seed.append(ent)
                    h = ent["horizons"].get("0.25s") or ent["horizons"]["0.1s"]
                    print(f"{name:>26} {'s' + str(seed) if seed is not None else '':>3} {key:<19} "
                          f"pos {h['pos']['all']['rms'] * 100:7.2f} cm  vel {h['vel']['all']['rms']:.3f} m/s"
                          + (f"  att {np.degrees(h['att']['all']['rms']):6.2f} deg" if "att" in h else "")
                          + (f"  diverged {ent['diverged']}" if ent["diverged"] else "")
                          + f"  [{time.time() - t1:.0f}s]")
                summ = {"per_seed": per_seed}
                if len(per_seed) > 1:
                    for hname in HORIZONS:
                        if all(hname in p["horizons"] for p in per_seed):
                            summ.setdefault("seed_mean", {})[hname] = {
                                q: {"rms_mean": float(np.mean([p["horizons"][hname][q]["all"]["rms"] for p in per_seed])),
                                    "rms_sd": float(np.std([p["horizons"][hname][q]["all"]["rms"] for p in per_seed], ddof=1))}
                                for q in per_seed[0]["horizons"][hname]}
                res[key] = summ
    # paired differences of the final-step position RMS vs the reference (seed-matched; seed-free models vs
    # every reference seed). Same failure policy as the summaries (diverged -> 10 m on both sides).
    # With --final every model is also paired against the oracle (excess over the floor).
    refs = [a.ref] + (["oracle"] if reg is not None else [])
    for ref in refs:
        for (name, seed, mode, integ), pe in pos_final.items():
            if name == ref:
                continue
            for (rn, rs, rm, ri), pe_r in pos_final.items():
                if rn != ref or rm != mode or ri != integ:
                    continue
                if seed is not None and rs is not None and rs != seed:      # seed-matched when both have seeds
                    continue
                d = paired_diff(pe, pe_r, flights, stat="rms")
                out["paired_pos_rmse_final"].append({"a": name, "a_seed": seed, "b": ref, "b_seed": rs, "mode": mode,
                                                    "integrator": integ, "diff_rms_m": d["diff"], "ci95": d["ci95"],
                                                    "p_a_not_better": d["p"], "n_diverged_a": d["n_diverged_a"],
                                                    "n_diverged_b": d["n_diverged_b"]})
    (REPO / a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}  [{time.time() - t0:.0f}s total]")


if __name__ == "__main__":
    main()
