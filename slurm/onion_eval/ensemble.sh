#!/bin/bash
# t370 ens: one step of the seed ensemble and dispersion recalibration
# (tools/onion_eval/ensemble_store.py): `fit` (per-assay NB dispersion factors on calibration
# regions; loads every seed's model) or `combine` (a new prediction store from seed stores). The
# metric phase then scores a combined store with slurm/onion_eval/score.sh as it scores any store.
#
#   sbatch --output=<logs>/%x_%j.out [resources] $KIT/slurm/onion_eval/ensemble.sh <kit_dir> \
#       fit|combine [ensemble_store.py args]
# Submitter: tools/minicandi/configs/recipe/submit_ensemble.sh. DRY_RUN=1 prints the command.
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_ensemble
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --gres=none

set -euo pipefail
USAGE="usage: onion_eval/ensemble.sh <kit_dir> fit|combine [ensemble_store.py args]"
KIT="${1:?$USAGE}"
STEP="${2:?$USAGE}"
shift 2
case "$STEP" in fit|combine) ;; *) echo "$USAGE" >&2; exit 2 ;; esac
CMD=(python3 "$KIT/tools/onion_eval/ensemble_store.py" "$STEP" "$@")
if [ "${DRY_RUN:-0}" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
T0=$(date +%s)
echo "=== onion ensemble $STEP job ${SLURM_JOB_ID:-local} host=$(hostname) $(date -u)"
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
echo "timing by phase (job): environment $((T1 - T0)) s, $STEP $((T2 - T1)) s, total $((T2 - T0)) s"
echo "=== done $(date -u)"
