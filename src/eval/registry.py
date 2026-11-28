"""Final model registry (configs/final_models.yaml): label -> exact checkpoints (review fix, Nov 28).

    from src.eval.registry import load_registry
    reg = load_registry()
    reg.runs("A")              # every single-model run on split A
    reg.label_of("m3_hybrid_tcn_seed1")   # -> "M3"
    reg.check_checkpoint(run, cfg)        # provenance: the checkpoint's split and config name match

Rules enforced on load: labels are unique; a run belongs to exactly one label; every label and every
pair refers to a known label or baseline; ensemble members share the ensemble's split.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO / "configs" / "final_models.yaml"
BASELINES = {"P0a", "P0b", "B1", "B2", "oracle"}


def _base(run: str) -> str:
    return re.sub(r"_seed\d+$", "", run)


def _seed(run: str) -> int | None:
    m = re.search(r"_seed(\d+)$", run)
    return int(m.group(1)) if m else None


@dataclass
class Registry:
    models: dict                      # label -> {"split", "runs"}
    ensembles: dict                   # label -> {"split", "members"}
    headline: dict
    pairs_1step: dict                 # split -> [[a, b, eval, subset]]
    pairs_rollout: list               # [[a, b]]
    rollout_reference: str
    path: str
    sha256: str
    _label: dict = field(default_factory=dict)

    # ---- lookups
    def labels(self, split: str | None = None, ensembles: bool = True) -> list[str]:
        out = [k for k, v in self.models.items() if split is None or v["split"] == split]
        if ensembles:
            out += [k for k, v in self.ensembles.items() if split is None or v["split"] == split]
        return out

    def runs(self, split: str | None = None) -> list[str]:
        return [r for k, v in self.models.items() if split is None or v["split"] == split for r in v["runs"]]

    def ensemble_members(self, split: str | None = None) -> dict:
        return {k: list(v["members"]) for k, v in self.ensembles.items() if split is None or v["split"] == split}

    def label_of(self, run: str) -> str:
        if run not in self._label:
            raise KeyError(f"{run} is not in {self.path}")
        return self._label[run]

    def split_of(self, label: str) -> str:
        return (self.models.get(label) or self.ensembles[label])["split"]

    def is_ensemble(self, label: str) -> bool:
        return label in self.ensembles

    def info(self) -> dict:
        """Recorded in every --final output."""
        return {"path": self.path, "sha256": self.sha256,
                "labels": {k: v["runs"] for k, v in self.models.items()},
                "ensembles": {k: v["members"] for k, v in self.ensembles.items()},
                "headline": self.headline}

    # ---- provenance
    def check_checkpoint(self, run: str, cfg: dict, split: str | None = None) -> None:
        """Raise unless the checkpoint config matches the registry: same split, and its config name
        and seed are those of the run name (catches renamed / copied / resumed-on-another-split runs)."""
        want = split
        if want is None:
            if run in self._label:
                want = self.models[self._label[run]]["split"]
            else:
                want = next((v["split"] for v in self.ensembles.values() if run in v["members"]), None)
        got = cfg.get("data", {}).get("split")
        if want is not None and got != want:
            raise ValueError(f"{run}: checkpoint trained on split {got}, registry says {want}")
        name = cfg.get("name")
        if name is not None and name != _base(run):
            raise ValueError(f"{run}: checkpoint config name {name!r} != run name {_base(run)!r}")
        s = _seed(run)
        if s is not None and "seed" in cfg and int(cfg["seed"]) != s:
            raise ValueError(f"{run}: checkpoint seed {cfg['seed']} != run seed {s}")


def load_registry(path: Path | str = REGISTRY_PATH) -> Registry:
    path = Path(path)
    raw = path.read_bytes()
    d = yaml.safe_load(raw)
    models, ens = d.get("models", {}) or {}, d.get("ensembles", {}) or {}
    dup = set(models) & set(ens) | (set(models) | set(ens)) & BASELINES
    if dup:
        raise ValueError(f"label used twice or clashes with a baseline: {sorted(dup)}")
    label = {}
    for k, v in models.items():
        if v.get("split") not in ("A", "B_s"):
            raise ValueError(f"{k}: split must be A or B_s")
        if not v.get("runs"):
            raise ValueError(f"{k}: no runs")
        for r in v["runs"]:
            if r in label:
                raise ValueError(f"run {r} is listed under both {label[r]} and {k}")
            label[r] = k
        bases = {_base(r) for r in v["runs"]}
        if len(bases) != 1:
            raise ValueError(f"{k}: runs of different configs {sorted(bases)} under one label")
    for k, v in ens.items():
        if v.get("split") not in ("A", "B_s") or not v.get("members"):
            raise ValueError(f"ensemble {k}: needs split and members")
        for r in v["members"]:
            if r in label and models[label[r]]["split"] != v["split"]:
                raise ValueError(f"ensemble {k}: member {r} is on another split")
    known = set(models) | set(ens) | BASELINES
    p1 = d.get("pairs_1step", {}) or {}
    for sp, lst in p1.items():
        for a, b, ev, sub in lst:
            if a not in known or b not in known:
                raise ValueError(f"pairs_1step {sp}: unknown label in {a} - {b}")
            if sub not in ("all", "le5", "gt5"):
                raise ValueError(f"pairs_1step {sp}: bad subset {sub}")
    pr = d.get("pairs_rollout", []) or []
    for a, b in pr:
        if a not in known or b not in known:
            raise ValueError(f"pairs_rollout: unknown label in {a} - {b}")
    ref = d.get("rollout_reference", "M2")
    if ref not in known:
        raise ValueError(f"rollout_reference {ref} unknown")
    try:
        shown = str(path.resolve().relative_to(REPO))
    except ValueError:
        shown = str(path)
    return Registry(models, ens, d.get("headline", {}) or {}, p1, pr, ref, shown,
                    hashlib.sha256(raw).hexdigest()[:16], label)
