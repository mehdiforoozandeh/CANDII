#!/bin/bash
# Close-out E5: score the 9 base-recipe seeds (conv autoencoder, conv + attention, U-Net; runs_fir/
# rung{2,3,4}_base_heldout, seeds 0 to 2) with foreign covariates at scoring (score.py
# --foreign-covariates). No training. Run on a Fir login node.
# DRY RUN BY DEFAULT: it prints the sbatch lines and submits nothing.
#
#   bash submit_foreign.sh            # dry run
#   DRY_RUN=0 bash submit_foreign.sh  # real submission -- the foreman's call only
#
# Per mode in MODES (default: both_keep_depth out_keep_depth in_keep_depth both_swap_depth) and seed:
# 4 CPU predict shards + 1 metrics job, with the options of runs_fir/score_final.sh (same-view pass,
# eval set evalset_p16.json, references baselines_p16), into <seed dir>/p16_foreign_<mode>, and the
# prediction store $PRED_STORE_ROOT/runs_fir/<run>/same_view/foreign_<mode>_s<FOREIGN_SEED>/seed<s>/
# p16_foreign_<mode> (removed after SCORED, KEEP_PRED_STORE=0). Donor seed FOREIGN_SEED (default 0).
# TRUE_RESCORE=1 adds one normal scoring (no foreign covariates) into <seed dir>/p16_e5: the E5
# scorer's ratio-of-means variance ratio is not in the stored p16_sv files (default 0: not run).
# Read with: select_rule.py arm-summary --runs $O/runs_fir --sub p16_foreign_<mode> (never mixed with
# p16_sv: the comparability key `foreign_covariates` differs).
# Idempotent through submit_arms.sh (a SCORED dir, or a queued job of the same name, is skipped).
# Foreign jobs are named mc_<run>_s<seed>_fc<mode>_s<FOREIGN_SEED>: a queued normal scoring of the
# run (mc_<run>_s<seed>) or a job of another mode never makes this script skip a seed.
set -u
O=/project/def-maxwl/mforooz/ONION
K="${K:-/project/def-maxwl/mforooz/onion_kit/COE5b}"
export DRY_RUN="${DRY_RUN:-1}"
MODES="${MODES:-both_keep_depth out_keep_depth in_keep_depth both_swap_depth}"
[ -f "$K/tools/onion_eval/report_extras.py" ] || { echo "kit $K has no E5 scorer" >&2; exit 2; }
module load python/3.10.13 >/dev/null 2>&1; unset EBPYTHONPREFIXES
export ONION_EXCLUDE=fc10713,fc10715,fc30601,fc30568,fc30605,fc30564,fc30654,fc30435,fc30562,fc30355,fc30570,fc30455,fc30659,fc30565,fc30613,fc30649,fc20647
export ONION_STAGE=1 CLUSTER=fir NICE=100 SEEDS="0 1 2"
export SCORE_ONLY=1 SCORE_SAME_VIEW=1 SCORE_PHASES=1 SCORE_SHARDS=4 SCORE_PRED_DEVICE=cpu SCORE_PRED_CPUS=8 SCORE_PRED_MEM=32000 SCORE_INFER_BATCH=8 SCORE_PRED_TIME=06:00:00 SCORE_CPUS=16 SCORE_MEM=64000 REFS_FROM=$O/eval/eic_v1/baselines_p16 KEEP_PRED_STORE=0
cd "$O/runs_fir" || exit 2
one() {   # <rung> <run dir name> <sub> [foreign mode]
  local r="$1" d="$2" sub="$3" mode="${4:-}"
  [ -f "$d/seed0/run.json" ] || { echo "MISSING $d/seed0/run.json"; return; }
  if [ -n "$mode" ]; then
    SCORE_SUBDIR="$sub" FOREIGN_COVARIATES="$mode" FOREIGN_SEED="${FOREIGN_SEED:-0}" \
      bash "$K/slurm/minicandi/submit_arms.sh" "$K" "$O/tables/eic_v1" "$O/eval/eic_v1/evalset_p16.json" \
      "$O/runs_fir" "$r" "$d=$d/seed0/run.json" 2>&1 | grep -v "^$"
  else
    env -u FOREIGN_SEED -u FOREIGN_COVARIATES SCORE_SUBDIR="$sub" bash "$K/slurm/minicandi/submit_arms.sh" "$K" "$O/tables/eic_v1" \
      "$O/eval/eic_v1/evalset_p16.json" "$O/runs_fir" "$r" "$d=$d/seed0/run.json" 2>&1 | grep -v "^$"
  fi
}
for mode in $MODES; do
  echo "## foreign covariates $mode -> p16_foreign_$mode"
  for r in 2 3 4; do one "$r" "rung${r}_base_heldout" "p16_foreign_$mode" "$mode"; done
done
if [ "${TRUE_RESCORE:-0}" = 1 ]; then
  echo "## normal rescore with the E5 scorer -> p16_e5"
  for r in 2 3 4; do one "$r" "rung${r}_base_heldout" p16_e5; done
fi
echo "DRY_RUN=$DRY_RUN"
