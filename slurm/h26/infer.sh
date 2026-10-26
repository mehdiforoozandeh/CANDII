#!/bin/bash
#SBATCH --job-name=h26r1_infer
#SBATCH --account=def-maxwl
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=3:00:00
#SBATCH --gres=none
#SBATCH --exclude=fc10713,fc10715
#SBATCH --array=0-13%12
#SBATCH --output=/project/def-maxwl/mforooz/h26/rerun1/logs/%x_%A_%a.out
# h26 rerun 1 on Fir, CPU: one (track, seed) per array task; 7 tracks x seeds 0,1. Submit only after
# STAGE=eligibility has written $RUN/out/eligibility.json (eligibility is frozen before inference).
# Prompts are the T_ cell self-paired; no B_ array is opened (PI ruling (iii), 2026-10-26).
set -euo pipefail
H26=/project/def-maxwl/mforooz/h26
RUN=${RUN:-$H26/rerun1}
KIT=${KIT:-/project/def-maxwl/mforooz/CANDII_main}
VENV=${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}
test -s "$RUN/out/eligibility.json" || { echo "[h26] no eligibility.json — refusing to infer first"; exit 3; }
TRACKS=(C12M02 C19M16 C40M17 C40M18 C07M20 C19M22 C07M29)
i=${SLURM_ARRAY_TASK_ID:?}
TRACK=${TRACKS[$((i % 7))]}
SEED=$((i / 7))
source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONPATH=$KIT/src
cd "$KIT"
echo "[h26] track=$TRACK seed=$SEED kit=$(git rev-parse --short HEAD) host=$(hostname)"
python "$RUN/code/tools/h26/infer.py" --track "$TRACK" --seed "$SEED" --out "$RUN/preds"
