#!/bin/bash
# Full test suite of a code kit, as one CPU job (t392 chunk C10a; slurm/minicandi/rcp_tests.sh
# with the job name rs392_tests and the failure list below).
#   KIT=/project/def-maxwl/mforooz/onion_kit/<id> sbatch --export=ALL slurm/t392/rs_tests.sh
# Writes the sorted ids of the failed (and erroring) tests, one per line, to
#   /project/def-maxwl/mforooz/t392/logs/failures_<kit id>.txt   (<kit id> = basename of KIT)
#SBATCH --job-name=rs392_tests
#SBATCH --account=def-maxwl
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=03:00:00
#SBATCH --gres=none
#SBATCH --output=/project/def-maxwl/mforooz/t392/logs/%x_%j.out
set -euo pipefail
: "${KIT:?set KIT}"
LOGS=/project/def-maxwl/mforooz/t392/logs
KIT_ID=$(basename "$KIT")
mkdir -p "$LOGS"
set +u; module load python/3.10.13; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$KIT/requirements-fir.txt"
pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
pip install --no-index pytest || true
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$KIT/src:$KIT:$KIT/tests"
REPORT="$SLURM_TMPDIR/pytest_report.txt"
rc=0
python -m pytest tests/ -q -rfE -p no:cacheprovider > "$REPORT" 2>&1 || rc=$?
tail -60 "$REPORT"
# "FAILED <id> - <message>" and "ERROR <id> - <message>" lines of the short summary -> <id>
{ grep -E '^(FAILED|ERROR) ' "$REPORT" || true; } | sed -E 's/^(FAILED|ERROR) //; s/ - .*$//' \
    | LC_ALL=C sort -u > "$LOGS/failures_${KIT_ID}.txt"
echo "pytest exit $rc; $(wc -l < "$LOGS/failures_${KIT_ID}.txt") failed ids -> $LOGS/failures_${KIT_ID}.txt"
