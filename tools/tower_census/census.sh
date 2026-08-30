#!/bin/bash
# Tower census over the CANDI_STORE eic `counts` layer, one array task per biosample.
#   sbatch --export=RUN=/scratch/$USER/<run>,TRACKS=signal census.sh
# TRACKS is signal (default, the 363-track census) | control | all.
# $RUN/biosamples.txt holds one biosample per line; --array must match its length.
#SBATCH --account=def-maxwl
#SBATCH --job-name=tower_census
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=0:30:00
set -euo pipefail
: "${RUN:?set RUN=/scratch/\$USER/<run dir>}"
TRACKS="${TRACKS:-signal}"
BS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$RUN/biosamples.txt")
echo "task $SLURM_ARRAY_TASK_ID -> $BS  (tracks=$TRACKS)"
/project/def-maxwl/mforooz/EpiDenoise/candi_venv/bin/python \
    "$RUN/code/tower_census.py" --biosample "$BS" --outdir "$RUN/out" --tracks "$TRACKS"
