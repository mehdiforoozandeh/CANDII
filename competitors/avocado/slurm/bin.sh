#!/bin/bash
# t50 stage 1 -- CANDI_STORE to one (n_bins, 267) float32 matrix per chromosome.
#
# 23 array tasks, one per chromosome, and each one is pure h5 reading. The MIG slice is requested
# for the same reason `slurm/bake.sh` requests one: no GPU compute happens here, the smallest slice
# only routes the job through the GPU account, and AGENTS.md 3.13 permits no other gres spec.
#
#   mkdir -p slurm-logs && sbatch --array=0-22 competitors/avocado/slurm/bin.sh
#
#SBATCH --account=def-maxwl
#SBATCH --job-name=t50_bin
#SBATCH --output=slurm-logs/t50_bin_%A_%a.out
#SBATCH --error=slurm-logs/t50_bin_%A_%a.err
# MEASURED, not guessed: chr21 (1.87M bins x 267 tracks) took 1444 s. The cost is close to linear
# in bins, so chr1 at 9.96M bins projects to ~2.1 h -- too near a 3 h limit to risk, and a task that
# times out leaves a .tmp behind and restarts from nothing.
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source /project/def-maxwl/mforooz/CANDII_t50/competitors/avocado/slurm/_env.sh

CHROM=$(sed -n "$((${SLURM_ARRAY_TASK_ID:-0} + 1))p" "$AVO/chroms.txt")
[ -z "$CHROM" ] && { echo "no chromosome for index ${SLURM_ARRAY_TASK_ID:-0}"; exit 1; }

echo "[t50_bin] chrom=$CHROM host=$(hostname) ws=$WS"
python "$AVO/bin_store.py" --regime "$REGIME" --out "$WS/binned" --chrom "$CHROM"
