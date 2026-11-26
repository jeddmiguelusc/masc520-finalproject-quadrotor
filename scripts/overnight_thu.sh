#!/usr/bin/env bash
# Thu Nov 26 overnight queue (~3.5-4.5 h on MPS). Needs the tuned configs from
#   python scripts/sweep_summary.py      (after scripts/sweep_thu.sh finished)
# Run:  caffeinate -i bash scripts/overnight_thu.sh
# One job at a time; a failed job is logged and skipped. Log: results/overnight_thu.log.
# Order = priority (core hypotheses first, stretch M4 last), so a crash late in the night costs least.
cd "$(dirname "$0")/.." || exit 1
LOG=results/overnight_thu.log
for f in m1_mlp_tuned m2_hybrid_mlp_tuned m3_hybrid_tcn_tuned m3_ens_tuned m3_hybrid_tcn_act_tuned \
         m1_mlp_Bs_tuned m2_hybrid_mlp_Bs_tuned m3_hybrid_tcn_Bs_tuned m3_ens_Bs_tuned m3_hybrid_tcn_act_Bs_tuned; do
  [ -f configs/$f.yaml ] || { echo "missing configs/$f.yaml - run python scripts/sweep_summary.py first"; exit 1; }
done
[ -z "$(git status --porcelain --untracked-files=no)" ] || echo "WARNING: uncommitted changes - runs.csv will log git_dirty=True" | tee -a "$LOG"
run() {  # config seed
  echo "=== $(date '+%H:%M:%S') $1 seed $2" | tee -a "$LOG"
  python -m src.train --config "$1" --seed "$2" 2>&1 | grep -E "done:|Error|error" | tee -a "$LOG" \
    || echo "FAILED $1 seed $2" | tee -a "$LOG"
}
echo "### start $(date) git $(git rev-parse --short HEAD)" | tee -a "$LOG"
# 1. Final M1 / M2 (A, then B_s for H1), 3 seeds each              ~20 min
for c in m1_mlp_tuned m2_hybrid_mlp_tuned m1_mlp_Bs_tuned m2_hybrid_mlp_Bs_tuned; do
  for s in 0 1 2; do run configs/$c.yaml $s; done
done
# 2. Final M3 on A, 3 seeds (H2, H3)                                ~25-40 min
for s in 0 1 2; do run configs/m3_hybrid_tcn_tuned.yaml $s; done
# 3. M3act: actuator-only history (is H2 aerodynamics or differencing?)   ~25-40 min
for s in 0 1 2; do run configs/m3_hybrid_tcn_act_tuned.yaml $s; done
# 4. Ensemble members, tuned config + NLL, max 150 epochs (H4)      ~50 min
for s in 0 1 2; do run configs/m3_ens_tuned.yaml $s; done
for s in 0 1 2; do run configs/m3_ens_Bs_tuned.yaml $s; done
# 5. Speed extrapolation for the windowed models                    ~20 min
for s in 0 1 2; do run configs/m3_hybrid_tcn_Bs_tuned.yaml $s; done
run configs/m3_hybrid_tcn_act_Bs_tuned.yaml 0
# 6. M4 transformer (stretch; GO: MPS 10.7 s/epoch in results/timing_bench.json < 36 s)   ~20 min/seed
for s in 0 1 2; do run configs/m4_hybrid_transformer.yaml $s; done
echo "### end $(date)" | tee -a "$LOG"
