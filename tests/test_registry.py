"""Review fix (Nov 28): configs/final_models.yaml loads, is consistent, and names only runs that exist."""
from __future__ import annotations

import csv

import pytest

from src.eval.registry import REPO, load_registry


def test_registry_loads_and_every_run_was_trained():
    reg = load_registry()
    trained = {r["run"] for r in csv.DictReader((REPO / "results" / "runs.csv").open())}
    named = set(reg.runs()) | {m for v in reg.ensembles.values() for m in v["members"]}
    assert named <= trained, sorted(named - trained)
    for sp, labels in reg.headline.items():
        assert all(lab in reg.models or lab in reg.ensembles for lab in labels), sp
    # default and tuned configs never share a label
    for lab, v in reg.models.items():
        assert len({r.rsplit("_seed", 1)[0] for r in v["runs"]}) == 1, lab


def test_registry_rejects_ambiguity(tmp_path):
    bad = tmp_path / "r.yaml"
    bad.write_text("models:\n  M3: {split: A, runs: [m3_hybrid_tcn_seed0, m3_hybrid_tcn_tuned_seed0]}\n")
    with pytest.raises(ValueError, match="different configs"):
        load_registry(bad)
    bad.write_text("models:\n  M3: {split: A, runs: [x_seed0]}\n  M3b: {split: A, runs: [x_seed0]}\n")
    with pytest.raises(ValueError, match="both"):
        load_registry(bad)
    bad.write_text("models:\n  M3: {split: A, runs: [x_seed0]}\npairs_rollout: [[M3, M9]]\n")
    with pytest.raises(ValueError, match="unknown"):
        load_registry(bad)


def test_check_checkpoint():
    reg = load_registry()
    ok = {"name": "m3_hybrid_tcn", "seed": 1, "data": {"split": "A"}}
    reg.check_checkpoint("m3_hybrid_tcn_seed1", ok)
    with pytest.raises(ValueError, match="split"):
        reg.check_checkpoint("m3_hybrid_tcn_seed1", {**ok, "data": {"split": "A_r1"}})
    with pytest.raises(ValueError, match="config name"):
        reg.check_checkpoint("m3_hybrid_tcn_seed1", {**ok, "name": "m3_hybrid_tcn_tuned"})
    with pytest.raises(ValueError, match="seed"):
        reg.check_checkpoint("m3_hybrid_tcn_seed1", {**ok, "seed": 2})
    reg.check_checkpoint("m3_ens_Bs_seed0", {"name": "m3_ens_Bs", "seed": 0, "data": {"split": "B_s"}})
