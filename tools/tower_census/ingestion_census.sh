#!/bin/bash
# Tower-INGESTION census (t78): do keep-dup-1 towers land inside a D12-eligible window?
#   sbatch --array=1-89%12 --export=RUN=/scratch/$USER/<run> ingestion_census.sh
# $RUN/biosamples.txt holds one biosample per line; --array must match its length.
# $RUN/src/ is the CANDII src/ this run imports candi.store from (D12 and D32 are the
# repo's own implementations, not copies).
#SBATCH --account=def-maxwl
#SBATCH --job-name=tower_ingest
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
set -euo pipefail
: "${RUN:?set RUN=/scratch/\$USER/<run dir>}"
BS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$RUN/biosamples.txt")
echo "task $SLURM_ARRAY_TASK_ID -> $BS"
/project/def-maxwl/mforooz/EpiDenoise/candi_venv/bin/python \
    "$RUN/code/ingestion_census.py" --biosample "$BS" --outdir "$RUN/out" \
    --tracks all --src "$RUN/src" --pilot-bed "$RUN/code/encode_pilot_hg38.bed"
