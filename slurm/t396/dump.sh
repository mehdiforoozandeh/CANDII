#!/bin/bash
# t396 C6: one shard of tools/t396/dump_native.py (truth or model) inside an ONION code kit.
#
#   export KIT=/project/def-maxwl/mforooz/onion_kit/SC7a
#   sbatch --job-name=sc_dump --output=<logs>/%x_%A_%a.out --array=0-3%12 \
#       [--time=...] "$KIT/slurm/t396/dump.sh" "$KIT" truth --evalset <json> --tables <dir> \
#       --rs-config <rs run dir> --out <dir>
#   sbatch ... "$KIT/slurm/t396/dump.sh" "$KIT" model --run <rs run dir> --evalset <json> \
#       --pred-store <dir> --out <dir> [--regions pair|all]
#
# Shards: when the arguments carry no --shard and the job is an array task, the script appends
# `--shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT` (target j belongs to shard j mod N; the
# array must be 0..N-1; write it as --array=0-<N-1>%12, the cap on torch-importing arrays). Both modes import torch (read-start dataset, model), so the environment is
# the kit venv of slurm/onion_eval/predict.sh. An existing output file is skipped.
# DRY_RUN=1 prints the command and runs nothing.
#SBATCH --account=def-maxwl
#SBATCH --job-name=sc_dump
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --gres=none
#SBATCH --exclude=fc10713,fc10715

set -euo pipefail
USAGE="usage: t396/dump.sh <kit_dir> truth|model <dump_native.py arguments...>"
KIT="${1:?$USAGE}"
MODE="${2:?$USAGE}"
shift 2
case "$MODE" in truth|model) ;; *) echo "$USAGE" >&2; exit 2 ;; esac
KIT=$(cd "$KIT" && pwd)
ARGS=("$@")
HAS_SHARD=0
for x in "${ARGS[@]}"; do [ "$x" = "--shard" ] && HAS_SHARD=1; done
if [ "$HAS_SHARD" = 0 ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  ARGS+=(--shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT:?array task without a count}")
fi
T="${SLURM_CPUS_PER_TASK:-1}"
CMD=(python3 "$KIT/tools/t396/dump_native.py" "$MODE" "${ARGS[@]}")
echo "${CMD[*]}"
if [ "${DRY_RUN:-0}" = 1 ]; then exit 0; fi
[ -f "$KIT/tools/t396/dump_native.py" ] || { echo "no dump_native.py in $KIT" >&2; exit 2; }

T0=$(date +%s)
echo "=== t396 dump job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-none} host=$(hostname) $(date -u)"
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
echo "timing by phase (job): environment $((T1 - T0)) s, dump $((T2 - T1)) s, total $((T2 - T0)) s"
echo "=== done $(date -u)"
