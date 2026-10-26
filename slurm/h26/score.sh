#!/bin/bash
#SBATCH --job-name=h26r2_score
#SBATCH --account=def-maxwl
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=3:00:00
#SBATCH --gres=none
#SBATCH --exclude=fc10713,fc10715
#SBATCH --output=/project/def-maxwl/mforooz/h26/rerun2/logs/%x_%j.out
# h26 rerun 2 on Fir, CPU. STAGE=eligibility (truth only) or STAGE=score (after eligibility).
# Reads the first run's chr21 halves extract in place (PI ruling (iv): no re-copy) and rerun 1's
# predictions in place (infer.py unchanged, so no new inference; PI second amendment, 2026-10-26).
set -euo pipefail
H26=/project/def-maxwl/mforooz/h26
RUN=${RUN:-$H26/rerun2}
PREDS=${PREDS:-$H26/rerun1/preds}
VENV=${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}
source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
python "$RUN/code/tools/h26/score.py" "${STAGE:?set STAGE}" --halves "$H26/halves_chr21.npz" \
    --preds "$PREDS" --out "$RUN/out" --workers "${SLURM_CPUS_PER_TASK:-4}"
