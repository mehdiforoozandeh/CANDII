#!/bin/bash
# CANDI_STORE -> one cache directory per chromosome. CPU only; the cost is store reads.
#
#   sbatch competitors/lavawizard/slurm/cache.sh
#
# 267 training tracks x n_bins float32, plus an int8 tercile row each: 2.0 GiB on chr21 and
# 10.6 GiB on chr1. The builder holds one biosample's block at a time, so the ask is set by the
# npy memmaps the OS writes back, not by the whole matrix.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t53_cache
#SBATCH --output=slurm-logs/%x_%A_%a.out
#SBATCH --error=slurm-logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-22
# `$0` is sbatch's spool copy, not this file, so the path comes from the submit directory.
ENV="${SLURM_SUBMIT_DIR:-$PWD}/competitors/lavawizard/slurm/_env.sh"
[ -f "$ENV" ] || { echo "[error] no $ENV -- submit from the repo root" >&2; exit 2; }
source "$ENV"
C=${CHROMS[$SLURM_ARRAY_TASK_ID]:-}
[ -n "$C" ] || { echo "[error] array index $SLURM_ARRAY_TASK_ID names no chromosome" >&2; exit 2; }
echo "[cache] $C  host=$(hostname)"
srun python -u -m lavawizard.store_eic cache --regime "$REGIME" --chrom "$C" --cache "$CACHE"
