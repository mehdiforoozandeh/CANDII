#!/bin/bash
# t396 C8: one shard of tools/t396/score_spaces.py (records, table or check) inside an ONION kit.
#
#   export KIT=/project/def-maxwl/mforooz/onion_kit/SC7b
#   sbatch --job-name=sc_spaces --output=<logs>/%x_%A_%a.out --array=0-<N-1>%12 [--time=...] \
#       --export=ALL "$KIT/slurm/t396/score_spaces.sh" "$KIT" records --truth <dir> \
#       --model <name> --seed <k> (--arrays <dir> | --pred-store <dir>) --scores-json <json> \
#       --regions all pair even --out <records_dir>/<model>_seed<k>_SHARD.jsonl [--validate-truth]
#   sbatch ... "$KIT/slurm/t396/score_spaces.sh" "$KIT" check --records <files...> --out check.tsv
#
# Shards: in records mode, when the arguments carry no --shard and the job is an array task, the
# script appends `--shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT` (target j belongs to shard
# j mod N; the array must be 0..N-1) and replaces the text SHARD in --out with the task id.
# numpy and scipy only (no torch): the shared candi_venv python, pinned to the kit's candi.
# Measured locally: one RL step on 256 x 16384 bins takes about 1.1 s, so the reference's grid
# (200 steps in all, run once and continued) costs about 4 CPU-min per target; one fixed cost of
# about 40 s per process builds deviance's NB table. DRY_RUN=1 prints the command and runs nothing.
#SBATCH --account=def-maxwl
#SBATCH --job-name=sc_spaces
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --gres=none
#SBATCH --exclude=fc10713,fc10715

set -euo pipefail
USAGE="usage: t396/score_spaces.sh <kit_dir> records|table|check <score_spaces.py arguments...>"
KIT="${1:?$USAGE}"
MODE="${2:?$USAGE}"
shift 2
case "$MODE" in records|table|check) ;; *) echo "$USAGE" >&2; exit 2 ;; esac
KIT=$(cd "$KIT" && pwd)
PY="${PY:-/project/def-maxwl/mforooz/candi_venv/bin/python}"
ARGS=("$@")
if [ "$MODE" = records ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  HAS_SHARD=0
  for x in "${ARGS[@]}"; do [ "$x" = "--shard" ] && HAS_SHARD=1; done
  if [ "$HAS_SHARD" = 0 ]; then
    ARGS+=(--shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT:?array task without a count}")
  fi
  for i in "${!ARGS[@]}"; do ARGS[$i]="${ARGS[$i]//SHARD/${SLURM_ARRAY_TASK_ID}}"; done
fi
T="${SLURM_CPUS_PER_TASK:-1}"
CMD=("$PY" "$KIT/tools/t396/score_spaces.py" "$MODE" "${ARGS[@]}")
echo "${CMD[*]}"
if [ "${DRY_RUN:-0}" = 1 ]; then exit 0; fi
[ -f "$KIT/tools/t396/score_spaces.py" ] || { echo "no score_spaces.py in $KIT" >&2; exit 2; }

T0=$(date +%s)
echo "=== t396 score_spaces $MODE job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-none} host=$(hostname) $(date -u)"
cat "$KIT/.kit_stamp" 2>/dev/null || true
cd "$KIT"
set +u; source "$(dirname "$PY")/activate"; set -u
source "$KIT/slurm/_kit_pin.sh" || exit 1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$T" OPENBLAS_NUM_THREADS="$T" FLEXIBLAS_NUM_THREADS="$T" MKL_NUM_THREADS="$T"
set +e
"${CMD[@]}"
RC=$?
set -e
T1=$(date +%s)
echo "timing: $MODE $((T1 - T0)) s, exit $RC"
echo "=== done $(date -u)"
exit "$RC"
