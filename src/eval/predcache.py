"""CP6: validation predictions of registry runs, computed once and cached (shared by
scripts/eval_uncertainty.py and scripts/error_analysis.py).

    from src.eval.predcache import PredCache
    pc = PredCache(device)                         # cache file: cache/val_predictions_final.npz (git-ignored)
    rows = pc.rows("A_val")                        # {"Y", "flights", "speed", ...} of the scored samples
    mean, var = pc.run("m3_ens_seed0", "A_val")    # var is None for models without an NLL head
    P = pc.baseline("A", "P0b", "A_val")           # P0b / B1 fitted on the split's own train part
    pc.save()

Validation only: eval names are "A_val" and "B_s_val"; anything else raises (no test part, ever).
Every cached array is keyed by eval set + run name; the cache is thrown away when the registry,
data/splits.json or a checkpoint file changes (sha / mtime recorded in the file).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.data.dataset import load_part, read_splits
from src.data.load import REPO, load_all
from src.eval.evaluate import scored_rows
from src.eval.registry import load_registry

CACHE = REPO / "cache" / "val_predictions_final.npz"
EVAL_PARTS = {"A_val": ("A", "val"), "B_s_val": ("B_s", "val")}


def _ck_stamp(run: str) -> str:
    p = REPO / "checkpoints" / run / "best.pt"
    st = p.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


class PredCache:
    def __init__(self, device=None, path: Path | str = CACHE, registry=None):
        self.device = device
        self.path = Path(path)
        self.reg = registry if registry is not None else load_registry()
        self.splits = read_splits()
        self._df = None
        self._rows: dict = {}
        self._arr: dict = {}
        self.meta = {"registry_sha256": self.reg.sha256, "splits_sha256": self.splits["meta"]["sha256"],
                     "stamps": {}}
        self.dirty = False
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as z:
                meta = json.loads(str(z["__meta__"]))
                if (meta.get("registry_sha256") == self.meta["registry_sha256"]
                        and meta.get("splits_sha256") == self.meta["splits_sha256"]):
                    self.meta["stamps"] = meta.get("stamps", {})
                    self._arr = {k: z[k] for k in z.files if k != "__meta__"}
                else:
                    print(f"[predcache] {self.path.name}: registry or splits changed -> recomputing")

    # ---------------------------------------------------------------- data
    def df(self):
        if self._df is None:
            self._df = load_all()
        return self._df

    def rows(self, eval_name: str) -> dict:
        if eval_name not in EVAL_PARTS:
            raise ValueError(f"{eval_name}: validation sets only ({sorted(EVAL_PARTS)})")
        if eval_name not in self._rows:
            k, p = EVAL_PARTS[eval_name]
            self._rows[eval_name] = scored_rows(load_part(k, p, self.splits, self.df()))
        return self._rows[eval_name]

    # ---------------------------------------------------------------- networks
    def run(self, run: str, eval_name: str) -> tuple[np.ndarray, np.ndarray | None]:
        km, kv = f"{eval_name}/{run}/mean", f"{eval_name}/{run}/var"
        stamp = _ck_stamp(run)
        if km in self._arr and self.meta["stamps"].get(run) == stamp:
            return self._arr[km].astype(np.float64), (self._arr[kv].astype(np.float64) if kv in self._arr else None)
        import torch
        from src.models import build_model
        from src.train import pick_device
        if self.device is None:
            self.device = pick_device("auto")
        ck = torch.load(REPO / "checkpoints" / run / "best.pt", map_location=self.device, weights_only=False)
        cfg = ck["config"]
        self.reg.check_checkpoint(run, cfg)
        model = build_model(cfg["model"]); model.load_state_dict(ck["model"]); model.to(self.device).eval()
        mean, var = _predict(model, self.rows(eval_name), int(cfg["data"]["max_window"]), self.device)
        self._arr[km] = mean.astype(np.float32)
        self._arr.pop(kv, None)
        if var is not None:
            self._arr[kv] = var.astype(np.float32)
        self.meta["stamps"][run] = stamp
        self.dirty = True
        print(f"[predcache] predicted {run} on {eval_name} ({len(mean):,} samples)")
        return self._arr[km].astype(np.float64), (self._arr[kv].astype(np.float64) if var is not None else None)

    def ensemble(self, label: str, eval_name: str) -> dict:
        """Gaussian mixture of the registry ensemble's members (src/models/ensemble.combine, in numpy)."""
        members = self.reg.ensemble_members()[label]
        mv = [self.run(r, eval_name) for r in members]
        return combine_np([m for m, _ in mv], [v for _, v in mv]) | {"members": members, "member_preds": mv}

    # ---------------------------------------------------------------- baselines
    def baseline(self, split: str, name: str, eval_name: str) -> np.ndarray:
        """P0b or B1 fitted on <split>/train (B1 alpha from results/val_baselines.json)."""
        key = f"{eval_name}/baseline:{split}:{name}/mean"
        if key in self._arr:
            return self._arr[key].astype(np.float64)
        from src.eval.baselines import PhysicsModel, RidgePoly2
        tr = scored_rows(load_part(split, "train", self.splits, self.df()))
        X = self.rows(eval_name)["X"]
        if name == "P0b":
            P = PhysicsModel().fit(tr["X"], tr["Y"]).predict(X)
        elif name == "B1":
            alpha = json.loads((REPO / "results" / "val_baselines.json").read_text())["splits"][split]["B1_alpha"]
            P = RidgePoly2().fit_stats(tr["X"], tr["Y"]).set_alpha(alpha).predict(X)
        else:
            raise ValueError(f"baseline {name}: P0b or B1 only")
        self._arr[key] = P.astype(np.float32)
        self.dirty = True
        print(f"[predcache] fitted {name} on {split}/train, predicted {eval_name}")
        return P

    def save(self):
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, __meta__=np.array(json.dumps(self.meta)), **self._arr)
        print(f"[predcache] wrote {self.path.relative_to(REPO)} ({self.path.stat().st_size / 1e6:.0f} MB)")
        self.dirty = False


def combine_np(means: list[np.ndarray], variances: list[np.ndarray | None]) -> dict:
    """Same formulas as src/models/ensemble.combine (population variance across members)."""
    mu = np.stack(means)
    mean = mu.mean(0)
    epi = mu.var(0)
    alea = np.stack(variances).mean(0) if all(v is not None for v in variances) else np.zeros_like(mean)
    return {"mean": mean, "var": alea + epi, "aleatoric": alea, "epistemic": epi}


def _predict(model, rows: dict, window: int, device, batch: int = 8192):
    """Copy of scripts/eval_models.predict (kept there unchanged): mean and NLL variance per scored sample."""
    import torch
    ws = rows["ws"]
    with torch.no_grad():
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
    return (np.concatenate(means).astype(np.float64),
            np.concatenate(vars_).astype(np.float64) if vars_ else None)


def git_hash() -> str:
    import subprocess
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def sha16(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
