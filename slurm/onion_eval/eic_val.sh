#!/bin/bash
# C16 (plan t370, D10): genome-wide predictions of one miniCANDI run on the EIC validation tracks,
# one array task per validation cell (tools/minicandi/predict.py --split val --cell-index i).
#
#   bash   $KIT/slurm/onion_eval/eic_val.sh <kit> <run dir> <out dir>    # launcher: submits itself
#   sbatch --array=0-<n-1>%12 [...] $KIT/slurm/onion_eval/eic_val.sh <kit> <run dir> <out dir>
#   -> <out>/counts/, <out>/signal/ (expectation, headline), <out>/signal_plug_in/, <out>/DONE
#      (predict.py's layout). Score each signal folder with tools/onion_eval/eic_val_compare.py score.
#
# Modes. Outside Slurm (no SLURM_JOB_ID) the script is a launcher: it counts the validation cells
# of the eval set (eic_val_compare.py val-cells) and submits itself as --array=0-<n-1>%12, job name
# rcp_eicval, --exclude=$ONION_EXCLUDE when that is set. Inside an array task it predicts the
# targets of cell $SLURM_ARRAY_TASK_ID; inside a job with no array index it predicts every cell.
# Variables:
#   EIC_VAL_CHROMS   comma list passed to predict.py as --chroms (e.g. chr21: the cost probe; such
#                    a run writes per-chromosome parts only, no .npy and no DONE). Unset: every
#                    chromosome. Defined here; no other code reads it.
#   ONION_ROOT       default /project/def-maxwl/mforooz/ONION (tables $ONION_ROOT/tables/eic_v1_2,
#                    eval set $ONION_ROOT/eval/eic_v1_2/evalset.json, logs $ONION_ROOT/logs)
#   ONION_EXCLUDE    node list for --exclude (launcher only)
#   DRY_RUN=1        print the sbatch line (launcher) or the predict command (task), run nothing
# Resources: CPU, 8 CPUs, 32000M (the 4000 MB per core rule), --time from the chr21 probe (C19)
# scaled by the genome's bins; the header's 12 h is a placeholder that sbatch --time overrides.
#SBATCH --account=def-maxwl
#SBATCH --job-name=rcp_eicval
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --gres=none

set -euo pipefail
USAGE="usage: onion_eval/eic_val.sh <kit> <run dir> <out dir>"
KIT="${1:?$USAGE}"
RUN="${2:?$USAGE}"
OUT="${3:?$USAGE}"
[ $# -eq 3 ] || { echo "$USAGE" >&2; exit 2; }

O="${ONION_ROOT:-/project/def-maxwl/mforooz/ONION}"
TABLES="$O/tables/eic_v1_2"
EVALSET="$O/eval/eic_v1_2/evalset.json"
EIC_VAL_CHROMS="${EIC_VAL_CHROMS:-}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  # ------------------------------------------------------------------ launcher
  N=$(python3 "$KIT/tools/onion_eval/eic_val_compare.py" val-cells --tables "$TABLES" \
      --evalset "$EVALSET")
  case "$N" in ''|*[!0-9]*|0) echo "no validation cells ($N) in $EVALSET" >&2; exit 2 ;; esac
  SB=(sbatch --parsable --array="0-$((N - 1))%12" --job-name=rcp_eicval
      --output="$O/logs/%x_%A_%a.out" --export=ALL)
  EXCLUDE="${ONION_EXCLUDE:-}"; EXCLUDE="${EXCLUDE//[[:space:]]/}"
  [ -z "$EXCLUDE" ] || SB+=(--exclude="$EXCLUDE")
  SB+=("$KIT/slurm/onion_eval/eic_val.sh" "$KIT" "$RUN" "$OUT")
  echo "validation cells: $N; EIC_VAL_CHROMS=${EIC_VAL_CHROMS:-all}"
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "${SB[*]}"; exit 0; fi
  export EIC_VAL_CHROMS
  "${SB[@]}"
  exit 0
fi

# -------------------------------------------------------------------- one task
CMD=(python3 "$KIT/tools/minicandi/predict.py" --run "$RUN" --tables "$TABLES"
     --evalset "$EVALSET" --split val --out "$OUT" --device cpu)
[ -z "${SLURM_ARRAY_TASK_ID:-}" ] || CMD+=(--cell-index "$SLURM_ARRAY_TASK_ID")
[ -z "$EIC_VAL_CHROMS" ] || CMD+=(--chroms "$EIC_VAL_CHROMS")

echo "EIC_VAL_CHROMS=${EIC_VAL_CHROMS:-all} cell index ${SLURM_ARRAY_TASK_ID:-all}"
if [ "${DRY_RUN:-0}" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
[ -f "$EVALSET" ] || { echo "no evalset $EVALSET" >&2; exit 2; }
[ -f "$RUN/ckpt.pt" ] || { echo "no checkpoint $RUN/ckpt.pt" >&2; exit 2; }

T0=$(date +%s)
echo "=== eic_val job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-none} host=$(hostname) $(date -u)"
set +u; module load python/3.10.13; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$KIT/requirements-fir.txt"
pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
T="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$T" OPENBLAS_NUM_THREADS="$T" FLEXIBLAS_NUM_THREADS="$T" MKL_NUM_THREADS="$T" \
  OMP_WAIT_POLICY=PASSIVE
T1=$(date +%s)
"${CMD[@]}"
T2=$(date +%s)
echo "timing by phase (job): environment $((T1 - T0)) s, predict $((T2 - T1)) s, total $((T2 - T0)) s"
echo "=== done $(date -u)"
