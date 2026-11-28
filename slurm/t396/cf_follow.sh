#!/bin/bash
# t396 B2: one shard of `tools/t396/cf_follow.py predict` (or a `table`) inside an ONION code kit.
#
#   export KIT=/project/def-maxwl/mforooz/onion_kit/SC2B4
#   sbatch --job-name=sc_follow --output=<logs>/%x_%A_%a.out --array=0-1%12 \
#       [--gres=... --cpus-per-task=... --mem=... --time=...] "$KIT/slurm/t396/cf_follow.sh" \
#       "$KIT" predict --run <run dir> --pairs <pairs.tsv> --evalset <json> --out <dir> \
#       [--model-name <label>] [--device cuda]
#   sbatch ... "$KIT/slurm/t396/cf_follow.sh" "$KIT" table --pred <dirs...> --out <dir>
#
# Shards: when a predict job is an array task and the arguments carry no --shard, the script
# appends `--shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT` (pair j belongs to shard j mod N;
# the array must be 0..N-1, capped at %12: it imports torch). Without --device the script passes
# cuda when the job has a GPU, else cpu. The environment is the kit venv of
# slurm/onion_eval/predict.sh. An existing npz is skipped. DRY_RUN=1 prints the command only.
#SBATCH --account=def-maxwl
#SBATCH --job-name=sc_follow
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --gres=none
#SBATCH --exclude=fc10713,fc10715

set -euo pipefail
USAGE="usage: t396/cf_follow.sh <kit_dir> predict|table <cf_follow.py arguments...>"
KIT="${1:?$USAGE}"
MODE="${2:?$USAGE}"
shift 2
case "$MODE" in predict|table) ;; *) echo "$USAGE" >&2; exit 2 ;; esac
KIT=$(cd "$KIT" && pwd)
ARGS=("$@")
HAS_SHARD=0
HAS_DEVICE=0
for x in "${ARGS[@]}"; do
  [ "$x" = "--shard" ] && HAS_SHARD=1
  [ "$x" = "--device" ] && HAS_DEVICE=1
done
if [ "$MODE" = predict ]; then
  if [ "$HAS_SHARD" = 0 ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    ARGS+=(--shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT:?array task without a count}")
  fi
  if [ "$HAS_DEVICE" = 0 ]; then
    if [ -n "${SLURM_JOB_GPUS:-}" ] || [ "${SLURM_GPUS_ON_NODE:-0}" != 0 ]; then
      ARGS+=(--device cuda)
    else
      ARGS+=(--device cpu)
    fi
  fi
fi
T="${SLURM_CPUS_PER_TASK:-1}"
CMD=(python3 "$KIT/tools/t396/cf_follow.py" "$MODE" "${ARGS[@]}")
echo "${CMD[*]}"
if [ "${DRY_RUN:-0}" = 1 ]; then exit 0; fi
[ -f "$KIT/tools/t396/cf_follow.py" ] || { echo "no cf_follow.py in $KIT" >&2; exit 2; }

T0=$(date +%s)
echo "=== t396 cf_follow job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-none} host=$(hostname) $(date -u)"
cat "$KIT/.kit_stamp" 2>/dev/null || true
set +u; module load python/3.10.13; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$KIT/requirements-fir.txt"
pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$T" OPENBLAS_NUM_THREADS="$T" FLEXIBLAS_NUM_THREADS="$T" MKL_NUM_THREADS="$T" \
  OMP_WAIT_POLICY=PASSIVE
T1=$(date +%s)
"${CMD[@]}"
T2=$(date +%s)
echo "timing by phase (job): environment $((T1 - T0)) s, $MODE $((T2 - T1)) s, total $((T2 - T0)) s"
echo "=== done $(date -u)"
