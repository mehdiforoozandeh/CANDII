#!/bin/bash
# t396 C9 (leg A), iteration 2 A4: one step of tools/t396/calib_cov.py (fit, score or table) inside
# an ONION kit. Iteration 2: one fit per arm, seed and half (--fit-regions odd | even); a start-output
# arm passes --arrays, a count-output arm passes only --ref-pred-store (its own prediction store).
#
#   export KIT=/project/def-maxwl/mforooz/onion_kit/SC<id>
#   sbatch --job-name=sc_calib --export=ALL --output=<logs>/%x_%j.out [--time=<probe>] \
#       "$KIT/slurm/t396/calib_cov.sh" "$KIT" fit --truth /scratch/mforooz/sc396/truth \
#       --arrays /scratch/mforooz/sc396/arrays/rs_halves_mix_avg_pearson_unet/seed<k> \
#       --ref-pred-store /scratch/mforooz/onion/pred_store_dev/runs_rcp/rcp_halves_mix_avg_pearson_unet/same_view/seed<k>/p16_dev \
#       --seed <k> --out /project/def-maxwl/mforooz/t396/legA_covariance/fit_seed<k>.npz
#   sbatch ... --dependency=afterok:<fit job> "$KIT/slurm/t396/calib_cov.sh" "$KIT" score \
#       <the same arguments> --fit <fit.npz> --out <records_seed<k>.jsonl>
#   sbatch ... "$KIT/slurm/t396/calib_cov.sh" "$KIT" table --records <3 files> \
#       [--legB-table <table.tsv>] [--scores-json <scores.json>] --out <dir>
#
# numpy and scipy only (no torch), so the shared candi_venv python. fit and score skip when their
# --out exists. Pass --window-bins / --infer-batch only if the read-start store differs from
# L = 1024, infer_batch = 8 (STORE.json of the store); score's `n_rel_diff_identity` checks it.
# Resources below are a guess before the probe (fit ~1.5 h of fit_scale bisection, score ~15 min
# for 43 targets); set --time from the probe. DRY_RUN=1 prints the command and runs nothing.
#SBATCH --account=def-maxwl
#SBATCH --job-name=sc_calib
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --gres=none
#SBATCH --exclude=fc10713,fc10715

set -euo pipefail
USAGE="usage: t396/calib_cov.sh <kit_dir> fit|score|table <calib_cov.py arguments...>"
KIT="${1:?$USAGE}"
MODE="${2:?$USAGE}"
shift 2
case "$MODE" in fit|score|table) ;; *) echo "$USAGE" >&2; exit 2 ;; esac
KIT=$(cd "$KIT" && pwd)
ARGS=("$@")
PY=/project/def-maxwl/mforooz/candi_venv/bin/python
CMD=("$PY" "$KIT/tools/t396/calib_cov.py" "$MODE" "${ARGS[@]}")
echo "${CMD[*]}"
if [ "${DRY_RUN:-0}" = 1 ]; then exit 0; fi
[ -f "$KIT/tools/t396/calib_cov.py" ] || { echo "no calib_cov.py in $KIT" >&2; exit 2; }

if [ "$MODE" != table ]; then
  OUT=""
  for ((i = 0; i < ${#ARGS[@]}; i++)); do
    [ "${ARGS[$i]}" = "--out" ] && OUT="${ARGS[$((i + 1))]:-}"
  done
  if [ -n "$OUT" ] && [ -s "$OUT" ]; then echo "$OUT exists, skipped"; exit 0; fi
fi

T0=$(date +%s)
echo "=== t396 calib_cov $MODE job ${SLURM_JOB_ID:-local} host=$(hostname) $(date -u)"
cat "$KIT/.kit_stamp" 2>/dev/null || true
source "$(dirname "$PY")/activate"
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
T="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$T" OPENBLAS_NUM_THREADS="$T" FLEXIBLAS_NUM_THREADS="$T" MKL_NUM_THREADS="$T"
"${CMD[@]}"
echo "timing: $MODE $(( $(date +%s) - T0 )) s"
echo "=== done $(date -u)"
