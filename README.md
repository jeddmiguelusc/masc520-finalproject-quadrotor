# masc520-finalproject-quadrotor

MASC 520 final project (USC, Fall 2026): replication and extensions of NeuroBEM
(Bauersfeld et al., RSS 2021) — learning quadrotor body forces and torques from flight data,
with a physics prior, uncertainty under speed extrapolation, and rollout evaluation.

> Status: CP0 (setup). Results table and reproduce commands are added as checkpoints pass.

## Setup

Requires Python 3.12+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt      # CPU-only PyTorch
```

## Data

Not included in this repo — see [data/README.md](data/README.md).

```bash
bash scripts/download_data.sh
python scripts/verify_data.py
```

## Layout

```
configs/    one YAML per model/run
data/       README, raw/ (git-ignored), splits.json
src/        data/, models/, train.py, eval/metrics.py, sim/rollout.py
scripts/    download_data.sh, verify_data.py, run_all.sh, evaluate_test.py, make_figures.py
notebooks/  01_eda.ipynb only; no logic lives here
tests/      test_frames.py, test_targets.py, test_splits.py
results/    val_*.json, test_*.json, runs.csv, main_table.md, figures/
docs/       decisions.md (decision log, one entry per checkpoint)
report/     final PDF + source
```

## AI assistance

Parts of this code were drafted with Claude (Anthropic). All results come from files in
`results/`; design decisions are logged by the author in `docs/decisions.md`.
