#!/bin/bash
# C6 (plan t370, decision D9): leave-control-out evaluation of one run's predicted ChIP control
# (tools/onion_eval/control_eval.py run). One CPU job.
#
#   sbatch --output=<logs>/%x_%j.out $KIT/slurm/onion_eval/control_eval.sh \
#       <kit_dir> <run_dir> <evalset.json> <out_dir>
#   -> <out_dir>/{control_eval.json,CONTROL_EVAL_DONE}
# Variables:
#   ONION_CTL_REGIONS  eval regions scored (default 64)
#   ONION_CTL_CELLS    comma-separated cells (default: every cell with a control)
#   ONION_INFER_BATCH  windows per forward (default 8)
#   ONION_STAGE        1 (default): copy the h5 files read to $SLURM_TMPDIR/stage first
#   DRY_RUN=1          print the command, run nothing
# Skips (exit 0) when <out_dir>/CONTROL_EVAL_DONE exists. Never point <out_dir> at a scorer's dir.
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_control_eval
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --gres=none

set -euo pipefail
USAGE="usage: onion_eval/control_eval.sh <kit_dir> <run_dir> <evalset.json> <out_dir>"
KIT="${1:?$USAGE}"
RUN="${2:?$USAGE}"
EVALSET="${3:?$USAGE}"
OUT="${4:?$USAGE}"
[ $# -eq 4 ] || { echo "$USAGE" >&2; exit 2; }

STAGE="${ONION_STAGE:-1}"
T="${SLURM_CPUS_PER_TASK:-1}"
CMD=(python3 "$KIT/tools/onion_eval/control_eval.py" run --run "$RUN" --evalset "$EVALSET"
     --out "$OUT" --regions "${ONION_CTL_REGIONS:-64}" --device cpu
     --infer-batch "${ONION_INFER_BATCH:-8}")
[ -z "${ONION_CTL_CELLS:-}" ] || CMD+=(--cells "$ONION_CTL_CELLS")
if [ "$STAGE" = 1 ] && [ -n "${SLURM_TMPDIR:-}" ]; then
  CMD+=(--stage-dir "$SLURM_TMPDIR/stage")
fi

if [ -f "$OUT/CONTROL_EVAL_DONE" ]; then
  echo "skip: $OUT/CONTROL_EVAL_DONE exists"
  exit 0
fi
if [ "${DRY_RUN:-0}" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
[ -f "$EVALSET" ] || { echo "no evalset $EVALSET" >&2; exit 2; }
[ -f "$RUN/ckpt.pt" ] || { echo "no checkpoint $RUN/ckpt.pt" >&2; exit 2; }

T0=$(date +%s)
echo "=== onion control_eval job ${SLURM_JOB_ID:-local} host=$(hostname) $(date -u)"
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
echo "timing by phase (job): environment $((T1 - T0)) s, control_eval $((T2 - T1)) s, total $((T2 - T0)) s"
echo "=== done $(date -u)"
