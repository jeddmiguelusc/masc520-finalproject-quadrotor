#!/usr/bin/env bash
# Wed Nov 25 overnight queue (~1.5 h on MPS). Runs one job at a time; a failed job is logged and skipped.
#   caffeinate -i bash scripts/overnight_wed.sh
# Log: results/overnight_wed.log (one "done:" line per run also lands in results/runs.csv).
cd "$(dirname "$0")/.." || exit 1
LOG=results/overnight_wed.log
run() {  # config seed
  echo "=== $(date '+%H:%M:%S') $1 seed $2" | tee -a "$LOG"
  python -m src.train --config "$1" --seed "$2" 2>&1 | grep -E "done:|Error|error" | tee -a "$LOG" \
    || echo "FAILED $1 seed $2" | tee -a "$LOG"
}
echo "### start $(date) git $(git rev-parse --short HEAD)" | tee -a "$LOG"
# 1. H4: ensemble members on B_s (needed whatever Thursday's tuning decides)
for s in 0 1 2; do run configs/m3_ensemble_member_Bs.yaml $s; done
# 2. M3 under speed extrapolation
run configs/m3_hybrid_tcn_Bs.yaml 0
# 3. Ensemble members on A
for s in 0 1 2; do run configs/m3_ensemble_member.yaml $s; done
# 4. Seed variance for the Wednesday comparisons
for s in 1 2; do
  for c in m1_mlp m2_hybrid_mlp m1_mlp_Bs m2_hybrid_mlp_Bs m3_hybrid_tcn; do run configs/$c.yaml $s; done
done
echo "### end $(date)" | tee -a "$LOG"
