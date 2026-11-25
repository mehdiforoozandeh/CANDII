#!/bin/bash
# t392 C13fix: tools/t392/stage_probe.py in a code kit (read-start scoring unstaged vs staged).
#   sbatch --output=<dir>/probe_%j.out $KIT/slurm/t392/rs_stage_probe.sh <kit> <run> <evalset> \
#       <pred store> <out json> [stage_probe.py args]
#SBATCH --account=def-maxwl
#SBATCH --job-name=rs392_stage_probe
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --gres=none
set -euo pipefail
USAGE="usage: rs_stage_probe.sh <kit> <run> <evalset> <pred store> <out json> [args]"
KIT="${1:?$USAGE}"; RUN="${2:?$USAGE}"; EV="${3:?$USAGE}"; STORE="${4:?$USAGE}"; OUT="${5:?$USAGE}"
shift 5
echo "=== rs392 stage probe ${SLURM_JOB_ID:-none} host=$(hostname) $(date -u)"
set +u; module load python/3.10.13; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$KIT/requirements-fir.txt"
pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
T="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 ONION_SCORE_SAME_VIEW=1
export OMP_NUM_THREADS="$T" OPENBLAS_NUM_THREADS="$T" FLEXIBLAS_NUM_THREADS="$T" MKL_NUM_THREADS="$T" \
  OMP_WAIT_POLICY=PASSIVE
python3 "$KIT/tools/t392/stage_probe.py" --run "$RUN" --evalset "$EV" --pred-store "$STORE" \
  --stage-dir "$SLURM_TMPDIR/stage" --out "$OUT" "$@"
echo "=== done $(date -u)"
