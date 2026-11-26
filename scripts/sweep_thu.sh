#!/usr/bin/env bash
# CP4 tuning sweep, Thu Nov 26 afternoon (~1 h 15 min on MPS). Equal budget: 8 configs per model on
# Split A, seed 0, selected on A/val best_val_loss (standardised MSE, the training loss).
# Config 0 of each model is the Wednesday run already in results/runs.csv (not re-run).
#   caffeinate -i bash scripts/sweep_thu.sh            then: python scripts/sweep_summary.py
# MLP grid (M1 and M2 identical): regularisation first - the MLPs early-stop at epochs 9-15 while
# train loss keeps falling (train/val std-MSE ~0.088/0.116 for M1).
cd "$(dirname "$0")/.." || exit 1
LOG=results/sweep_thu.log
run() {  # config name overrides...
  local cfg=$1 name=$2; shift 2
  echo "=== $(date '+%H:%M:%S') $name: $*" | tee -a "$LOG"
  python -m src.train --config "$cfg" --seed 0 --set name="$name" "$@" 2>&1 | grep -E "done:|Error|error" \
    | tee -a "$LOG" || echo "FAILED $name" | tee -a "$LOG"
}
echo "### start $(date) git $(git rev-parse --short HEAD)" | tee -a "$LOG"
for m in m1_mlp m2_hybrid_mlp; do
  run configs/$m.yaml ${m}_sw1 train.weight_decay=1e-2
  run configs/$m.yaml ${m}_sw2 train.weight_decay=1e-1
  run configs/$m.yaml ${m}_sw3 model.dropout=0.1
  run configs/$m.yaml ${m}_sw4 model.dropout=0.2
  run configs/$m.yaml ${m}_sw5 model.input_noise=0.05
  run configs/$m.yaml ${m}_sw6 model.input_noise=0.1
  run configs/$m.yaml ${m}_sw7 model.dropout=0.1 model.input_noise=0.05 train.weight_decay=1e-2
done
# M3 grid: same regularisers (sw1-sw4) + optimisation (sw5-sw7); M3 trains for 64-84 epochs and
# stops with the LR already annealed to ~1e-5, train/val std-MSE 0.054/0.074.
M=m3_hybrid_tcn
run configs/$M.yaml ${M}_sw1 train.weight_decay=1e-2
run configs/$M.yaml ${M}_sw2 model.dropout=0.1
run configs/$M.yaml ${M}_sw3 model.input_noise=0.05
run configs/$M.yaml ${M}_sw4 model.dropout=0.1 model.input_noise=0.05 train.weight_decay=1e-2
run configs/$M.yaml ${M}_sw5 train.batch_size=4096 train.lr=2e-3
run configs/$M.yaml ${M}_sw6 train.lr=3e-4
run configs/$M.yaml ${M}_sw7 train.max_epochs=150 train.patience=15
echo "### end $(date)" | tee -a "$LOG"
