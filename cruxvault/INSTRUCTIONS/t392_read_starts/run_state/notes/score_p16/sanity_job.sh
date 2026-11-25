#!/bin/bash
# t392 C13a: one sanity job (tools/t392/sanity.py; read-start model vs read-start truth, R12).
#   sbatch [resources] sanity_job.sh <kit dir> <run dir> <evalset.json> <out json>
# Copy on Fir: /project/def-maxwl/mforooz/t392/sanity_job.sh (environment as slurm/onion_eval/predict.sh).
set -euo pipefail
KIT="$1"; RUN="$2"; EV="$3"; OUT="$4"
T="${SLURM_CPUS_PER_TASK:-1}"
echo "=== rs392 sanity job ${SLURM_JOB_ID:-local} host=$(hostname) $(date -u)"
set +u; module load python/3.10.13; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$KIT/requirements-fir.txt"
pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$T" OPENBLAS_NUM_THREADS="$T" FLEXIBLAS_NUM_THREADS="$T" MKL_NUM_THREADS="$T"
python3 "$KIT/tools/t392/sanity.py" --run "$RUN" --evalset "$EV" --out "$OUT"
echo "=== done $(date -u)"
