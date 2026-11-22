#!/bin/bash
# t371: re-predict and score trained miniCANDI recipe runs with the deviance switch, into their
# own score folder, so the current scores (p16_sv, p256_sv) are never touched. Login node only.
#
#   K=<kit dir> RUNS_LIST="<run name> ..." SUB=p16|p256 [SEEDS="0 1"] [DRY_RUN=0] \
#       bash $K/slurm/onion_eval/deviance_submit.sh
#   -> <runs_rcp>/<run>/seed<s>/{p16_dev|p256_dev}/{scores.json,...}; scores.json carries every
#      current key (same options as submit_recipe.sh: same-view on, CPU predict, batch 8) plus the
#      key `deviance`. Stores go to /scratch/mforooz/onion/pred_store_dev and are kept
#      (KEEP_PRED_STORE=1), so a metric fix re-runs the metrics job only (METRICS_ONLY=1).
# Job names start with dev_ (JOB_PREFIX), so no job of the recipe run is skipped or matched.
# DRY_RUN defaults to 1 (print only).
set -u
set -o pipefail
module load python/3.10.13 >/dev/null 2>&1; unset EBPYTHONPREFIXES
O="${O:-/project/def-maxwl/mforooz/ONION}"
K="${K:?set K=<kit dir>}"
V=eic_v1_2
SUBT="${SUB:?set SUB=p16 or p256}"
RUNS_LIST="${RUNS_LIST:?set RUNS_LIST=<run names>}"
export DRY_RUN="${DRY_RUN:-1}"
export SEEDS="${SEEDS:-0 1}"
PY="${PYTHON:-python3}"
RUNS=$O/runs_rcp
T=$O/tables/$V
SUBMIT="$K/slurm/minicandi/submit_arms.sh"
RA="$K/tools/minicandi/recipe_arms.py"
unset FOREIGN_COVARIATES FOREIGN_SEED SCORE_INPUT_DEPTH SCORE_TIME
export SCORE_SAME_VIEW=1 SCORE_PHASES=1 SCORE_ONLY=1 SCORE_PRED_DEVICE=cpu SCORE_PRED_CPUS=8 \
  SCORE_PRED_MEM=32000 SCORE_INFER_BATCH=8 SCORE_PRED_TIME=06:00:00 SCORE_CPUS=16 SCORE_MEM=64000 \
  KEEP_PRED_STORE=1 EXTRA_SCORE_ARGS="--score-deviance" JOB_PREFIX="dev_" \
  PRED_STORE_ROOT=/scratch/mforooz/onion/pred_store_dev
case $SUBT in
  p16) EV=$O/eval/$V/evalset_p16.json; B=$O/eval/$V/baselines_p16; export SCORE_SHARDS=4 SCORE_SUBDIR=p16_dev ;;
  p256) EV=$O/eval/$V/evalset.json; B=$O/eval/$V/baselines
        export SCORE_SHARDS=8 SCORE_TIME=06:00:00 SCORE_SUBDIR=p256_dev ;;
  *) echo "SUB must be p16 or p256" >&2; exit 2 ;;
esac
export REFS_FROM=$B
EXC="$(sed -n 's/^ *export ONION_EXCLUDE=\([^ ]*\).*/\1/p' "$K/tools/minicandi/configs/base_on_version.sh" | head -n 1)"
export ONION_EXCLUDE="$EXC" ONION_STAGE=1 CLUSTER=fir NICE="${NICE:-100}"

LIST="$("$PY" "$RA" list --stage all --models "2 3 4")" || { echo "recipe_arms.py list failed" >&2; exit 2; }
for want in $RUNS_LIST; do
  line="$(printf '%s\n' "$LIST" | awk -F'\t' -v w="$want" '$3 == w' | head -n 1)"
  [ -n "$line" ] || { echo "no recipe arm for run $want" >&2; exit 2; }
  IFS=$'\t' read -r r arm run cfg tt <<< "$line"
  [ -f "$RUNS/$run/seed0/DONE" ] || { echo "skip $run: not trained (no seed0/DONE)"; continue; }
  echo "## $run ($arm, rung $r) -> $SCORE_SUBDIR"
  bash "$SUBMIT" "$K" "$T" "$EV" "$RUNS" "$r" "$run=$K/$cfg" < /dev/null 2>&1 | { grep -v "^$" || true; }
done
