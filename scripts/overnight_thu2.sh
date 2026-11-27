#!/usr/bin/env bash
# Thu Nov 26, second overnight queue (~45 min on MPS). Head start for Fri CP6.
#   caffeinate -i bash scripts/overnight_thu2.sh
# 1. M3act with the Wednesday default config (3 seeds) - matched comparison for M3 = Wed default,
#    in case the final M3 / ensemble go back to the untuned config (Fri decision).
# 2. Score them against Wed M3 / M2 (seed-matched paired diffs) -> results/val_models_act_default.json
# 3. Rollout evaluation of every candidate model on A/val, 0.1 s and 0.25 s -> results/val_rollout.json
cd "$(dirname "$0")/.." || exit 1
LOG=results/overnight_thu2.log
[ -z "$(git status --porcelain --untracked-files=no)" ] || echo "WARNING: uncommitted changes - runs.csv will log git_dirty=True" | tee -a "$LOG"
echo "### start $(date) git $(git rev-parse --short HEAD)" | tee -a "$LOG"
for s in 0 1 2; do
  echo "=== $(date '+%H:%M:%S') configs/m3_hybrid_tcn_act.yaml seed $s" | tee -a "$LOG"
  python -m src.train --config configs/m3_hybrid_tcn_act.yaml --seed $s 2>&1 | grep -E "done:|Error|error" | tee -a "$LOG" \
    || echo "FAILED m3_hybrid_tcn_act seed $s" | tee -a "$LOG"
done
echo "=== $(date '+%H:%M:%S') eval_models (M3act default vs M3 / M2 Wed)" | tee -a "$LOG"
python scripts/eval_models.py --runs m2_hybrid_mlp_seed0 m2_hybrid_mlp_seed1 m2_hybrid_mlp_seed2 \
  m3_hybrid_tcn_seed0 m3_hybrid_tcn_seed1 m3_hybrid_tcn_seed2 \
  m3_hybrid_tcn_act_seed0 m3_hybrid_tcn_act_seed1 m3_hybrid_tcn_act_seed2 \
  --out results/val_models_act_default.json > results/eval_models_act_default.log 2>&1 \
  && echo "ok -> results/val_models_act_default.json" | tee -a "$LOG" || echo "FAILED eval_models" | tee -a "$LOG"
echo "=== $(date '+%H:%M:%S') eval_rollout" | tee -a "$LOG"
python scripts/eval_rollout.py > results/eval_rollout.log 2>&1 \
  && echo "ok -> results/val_rollout.json" | tee -a "$LOG" || echo "FAILED eval_rollout (see results/eval_rollout.log)" | tee -a "$LOG"
echo "### end $(date)" | tee -a "$LOG"
