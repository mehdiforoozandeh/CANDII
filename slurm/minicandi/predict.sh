#!/bin/bash
# PG: genome-wide prediction of one miniCANDI run in the EIC challenge format
# (tools/minicandi/predict.py). An array task predicts the targets of one cell (cells of the split
# sorted; SLURM_ARRAY_TASK_ID = the cell index; a task past the last cell exits 0); a job with no
# array predicts every target. Skip-if-exists: <out>/DONE skips the job, a finished target or
# chromosome is skipped.
#
#   CPU form (the header: 4 cores x 4000 MB):
#     sbatch --time=<measured> --output=$OUT/predict_%A_%a.out --array=0-<n cells - 1>%12 \
#         $KIT/slurm/minicandi/predict.sh $KIT $RUN $TABLES $EVALSET <val|blind> $OUT cpu [predict.py args]
#   GPU form (the fixed slice; 1 CPU on Fir, 2 on Nibi; on Fir also --exclude=fc10713,fc10715):
#     sbatch --time=<measured> --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=2 \
#         --mem=24000M ... predict.sh ... gpu [predict.py args]
#   -> $OUT/counts/<eic_track>.h5, $OUT/signal/<eic_track>.npy, $OUT/signal_plug_in/<eic_track>.npy,
#      $OUT/lambda_source.tsv, $OUT/DONE
#
# Cost first: one job without an array and with "--chroms chr21" measures the cost per model
# (chr21 is 1.87 M of the 123.5 M challenge bins); then the array over the cells, which resumes the
# probe's ledger pass. Blind (gate G4 only): the extra args are
#   --allow-blind --freeze <FREEZE.md naming this run dir>
# and predict.py ledgers the pass in BLIND_PASSES.tsv beside the eval set under the key
# predict:<run dir>; a second pass is refused without --repeat-blind-reason "<text>".
# Storage: about 25 GB of counts and 50 GB of signal per model: write under /scratch.
# DRY_RUN=1 prints the command and exits.
#SBATCH --account=def-maxwl
#SBATCH --job-name=minicandi_predict
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --gres=none

set -euo pipefail
USAGE="usage: minicandi/predict.sh <kit_dir> <run_dir> <tables_dir> <evalset.json> <val|blind> <out_dir> <cpu|gpu> [predict.py args]"
KIT="${1:?$USAGE}"
RUN="${2:?$USAGE}"
TABLES="${3:?$USAGE}"
EVALSET="${4:?$USAGE}"
SPLIT="${5:?$USAGE}"
OUT="${6:?$USAGE}"
MODE="${7:?$USAGE}"
shift 7
case "$SPLIT" in val|blind) ;; *) echo "split must be val or blind" >&2; exit 2 ;; esac
case "$MODE" in
  cpu) DEVICE=cpu ;;
  gpu) DEVICE=cuda ;;
  *) echo "mode must be cpu or gpu" >&2; exit 2 ;;
esac
[ -f "$EVALSET" ] || { echo "no evalset $EVALSET" >&2; exit 2; }
CELL=()
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] && CELL=(--cell-index "$SLURM_ARRAY_TASK_ID")
CMD=(python3 "$KIT/tools/minicandi/predict.py" --run "$RUN" --tables "$TABLES" --evalset "$EVALSET"
     --split "$SPLIT" --out "$OUT" --device "$DEVICE" ${CELL[@]+"${CELL[@]}"} "$@")
if [ "${DRY_RUN:-0}" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
if [ -f "$OUT/DONE" ]; then echo "skip: $OUT/DONE exists"; exit 0; fi
if [ "$MODE" = gpu ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "mode gpu but the job has no GPU: submit with --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" >&2
  exit 2
fi
mkdir -p "$OUT"
echo "=== minicandi predict job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-none} host=$(hostname) $(date -u)"
set +u; module load python/3.10.13; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$KIT/requirements-fir.txt"
pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
"${CMD[@]}"
echo "=== done $(date -u)"
