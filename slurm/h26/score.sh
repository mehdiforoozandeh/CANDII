#!/bin/bash
#SBATCH --job-name=h26_score
#SBATCH --account=def-maxwl
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=3:00:00
#SBATCH --gres=none
#SBATCH --output=/project/def-maxwl/mforooz/h26/logs/%x_%j.out
# h26 on Fir, CPU. STAGE=eligibility (truth only; run BEFORE infer.sh) or STAGE=score (after).
set -euo pipefail
H26=/project/def-maxwl/mforooz/h26
VENV=${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}
source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
python "$H26/code/tools/h26/score.py" "${STAGE:?set STAGE}" --halves "$H26/halves_chr21.npz" \
    --preds "$H26/preds" --out "$H26/out" --workers "${SLURM_CPUS_PER_TASK:-4}"
