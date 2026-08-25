#!/bin/bash
# Rebuild the challenge's Average baseline from the Dataset-3 training+validation tracks.
# One array task per scored chromosome; 23 tasks for chr1-22 + chrX.
#
# "training set" for the round-2 baseline means train+validation, not train alone. 001 settled that
# via the ATAC-seq exception -- ATAC has zero validation experiments and is the only assay that
# matches published on training-only -- and its train-only variant misses published rows
# (frac within 1% = 0.18) while trainval matches them exactly. VARIANT therefore defaults to
# trainval, and train is kept only so the discriminating comparison can be re-run.
#
#   sbatch --array=0-22 --export=ALL,VARIANT=trainval,OUT=$SC/avg_official slurm/build_average.slurm
#
#SBATCH --job-name=t54_avg
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=slurm_logs/t54_avg_%A_%a.out
#SBATCH --error=slurm_logs/t54_avg_%A_%a.err

set -uo pipefail
REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_t54}"
ENT="$REPO/competitors/entrants"
DATA="${DATA:-/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE}"
M001="${M001:-/project/def-maxwl/mforooz/MAX_EPI_IMPUTATION/experiments/001_2026-07-27_eic_leaderboard_repro}"
OUT="${OUT:-/project/def-maxwl/mforooz/CANDII_t54_work/avg_official}"
VARIANT="${VARIANT:-trainval}"

# shellcheck disable=SC1091
source "$ENT/env_fir.sh"

CHROMS=(chr1 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr2 \
        chr20 chr21 chr22 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chrX)
CHROM=${CHROMS[$SLURM_ARRAY_TASK_ID]}

# The build script reads every source track from ONE directory, so train and validation bigwigs are
# presented through a symlink farm rather than copied -- 254 GB is not duplicated to satisfy an
# argument shape. Links are per-job in $SLURM_TMPDIR and vanish with the job.
FARM="$SLURM_TMPDIR/tracks"
mkdir -p "$FARM"
ln -sfn "$DATA"/training_data/*.bigwig "$FARM/" 2>/dev/null
ln -sfn "$DATA"/validation_data/*.bigwig "$FARM/" 2>/dev/null
echo "[farm] $(ls "$FARM" | wc -l) tracks linked for $CHROM/$VARIANT" >&2

python3 "$ENT/vendor/fir_build_average_official.py" \
    --chrom "$CHROM" \
    --bridge "$M001/output/bridge/eic_bridge.csv" \
    --track-dir "$FARM" \
    --out "$OUT" \
    --variant "$VARIANT"
