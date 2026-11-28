#!/bin/bash
# Full test suite of a code kit, as one CPU job (t370 chunk C11).
#   KIT=/project/def-maxwl/mforooz/onion_kit/<id> sbatch --export=ALL slurm/minicandi/rcp_tests.sh
#SBATCH --job-name=rcp_tests
#SBATCH --account=def-maxwl
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=03:00:00
#SBATCH --gres=none
#SBATCH --output=/project/def-maxwl/mforooz/ONION/runs_rcp/logs/%x_%j.out
set -euo pipefail
: "${KIT:?set KIT}"
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
python -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -60
