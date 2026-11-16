#!/bin/bash
# Close-out D1 (PI ruling of 2026-11-16): retrain and score the 3 base recipes, 3 seeds each, on a
# new tables version (default eic_v1_2: eic_v1 with lab, antibody, donor and a code-only platform,
# vocabularies from the training rows only; tools/onion_data/fill_covariates.py). eic_v1_1 has
# untrained embedding ids for values outside training: never train on it. Run on a Fir login
# node. DRY RUN BY DEFAULT.
#
#   STAGE=baselines bash base_on_version.sh            # dry run of stage 1
#   STAGE=baselines DRY_RUN=0 bash base_on_version.sh  # stage 1: the 4 screening baselines
#   STAGE=runs bash base_on_version.sh                 # dry run of stage 2
#   STAGE=runs DRY_RUN=0 bash base_on_version.sh       # stage 2: 9 trainings + their scoring
#
# Stage 1 scores marginal_nb, position_average, depth_only_scaling and pseudoreplicate_floor on
# $O/eval/$V/evalset_p16.json into $O/eval/$V/baselines_p16/<name>/seed0 (the sbatch line of the
# eic_v1 screening baselines, kit I2a, 2026-11-14; skip when SCORED exists). The oracle reference
# (baselines_p16/oracle_floor/seed0) is not a job here: it is built beside the eval set.
# The baselines read no covariate, so their scores must equal the eic_v1 scores. Check each one,
# from a worktree, after stage 1 ends (exit 0 and "differences: 0" expected):
#   bash tools/onion_kit/remote_check.sh --cluster fir COD1 tools/onion_eval/compare_scores.py \
#     /project/def-maxwl/mforooz/ONION/eval/eic_v1/baselines_p16/<name>/seed0 \
#     /project/def-maxwl/mforooz/ONION/eval/$V/baselines_p16/<name>/seed0 \
#     --ignore wall_seconds --ignore timing_seconds --ignore scorer_settings
# The scorer refuses a reference whose evalset_sha256 or tables_sha256 differs from the eval set,
# so the baselines of one version are never reused for another; they are rerun per version.
# Stage 2 refuses unless the 4 baselines are SCORED and the oracle reference exists. It then
# submits rung{2,3,4}/base_heldout.json (conv autoencoder, conv + attention, U-Net; the configs of
# the reference runs runs_fir/rung{2,3,4}_base_heldout, identical key for key) as
# $V_rung{2,3,4}_base_heldout, seeds 0 1 2, with the scoring options of
# configs/ablations/submit_all.sh (same-view pass into p16_sv) and REFS_FROM on the new version.
set -u
module load python/3.10.13 >/dev/null 2>&1; unset EBPYTHONPREFIXES
O=/project/def-maxwl/mforooz/ONION
K="${K:-/project/def-maxwl/mforooz/onion_kit/COD1b}"
V="${V:-eic_v1_2}"
T=$O/tables/$V
EV=$O/eval/$V/evalset_p16.json
B=$O/eval/$V/baselines_p16
export DRY_RUN="${DRY_RUN:-1}"
STAGE="${STAGE:?set STAGE=baselines or STAGE=runs}"
for p in "$T/measurements.tsv" "$T/covariates.json" "$EV"; do
  [ -e "$p" ] || { echo "missing $p" >&2; exit 2; }
done
case $STAGE in
baselines)
  n=0
  for name in marginal_nb position_average depth_only_scaling pseudoreplicate_floor; do
    out=$B/$name/seed0
    [ -e "$out/SCORED" ] && { echo "SCORED $out"; continue; }
    line=(sbatch --parsable --account=def-maxwl --job-name=onion_score_${V}_p16_$name --time=02:00:00
          --cpus-per-task=16 --mem=64000M --gres=none --output=$O/logs/%x_%j.out
          $K/slurm/onion_eval/score.sh $K $EV $out baseline $name)
    if [ "$DRY_RUN" = 1 ]; then echo "DRY ${line[*]}"; else mkdir -p "$out"; echo "$("${line[@]}") $name"; fi
    n=$((n+1))
  done
  echo "baselines submitted (or printed): $n; DRY_RUN=$DRY_RUN" ;;
runs)
  miss=""
  for name in marginal_nb position_average depth_only_scaling pseudoreplicate_floor; do
    [ -e "$B/$name/seed0/SCORED" ] || miss="$miss $name"
  done
  [ -e "$B/oracle_floor/seed0/oracle_floor.json" ] || miss="$miss oracle_floor"
  if [ -n "$miss" ]; then
    echo "refused: baselines not ready on $V:$miss" >&2
    [ "$DRY_RUN" = 1 ] || exit 3
  fi
  export ONION_EXCLUDE=fc10713,fc10715,fc30601,fc30568,fc30605,fc30564,fc30654,fc30435,fc30562,fc30355,fc30570,fc30455,fc30659,fc30565,fc30613,fc30649
  export ONION_STAGE=1 CLUSTER=fir NICE=100 SEEDS="${SEEDS:-0 1 2}" TRAIN_TIME=03:00:00
  export SCORE_SAME_VIEW=1 SCORE_PHASES=1 SCORE_SUBDIR=p16_sv SCORE_SHARDS=4 SCORE_PRED_DEVICE=cpu SCORE_PRED_CPUS=8 SCORE_PRED_MEM=32000 SCORE_INFER_BATCH=8 SCORE_PRED_TIME=06:00:00 SCORE_CPUS=16 SCORE_MEM=64000 REFS_FROM=$B KEEP_PRED_STORE=0
  for r in 2 3 4; do
    bash $K/slurm/minicandi/submit_arms.sh $K $T $EV $O/runs_fir $r \
      "${V}_rung${r}_base_heldout=$K/tools/minicandi/configs/rung$r/base_heldout.json" 2>&1 | grep -v "^$"
  done
  echo "runs: 3 recipes x seeds ($SEEDS); DRY_RUN=$DRY_RUN" ;;
*) echo "STAGE must be baselines or runs" >&2; exit 2 ;;
esac
