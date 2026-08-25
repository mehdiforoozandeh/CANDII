#!/bin/bash
# t50 stage 3 -- write the RIVALS_PLAN.md 4.1 prediction root, one chromosome per array task.
#
# The root covers all 23 chromosomes, which serves BOTH protocols: P1 is the scorer reading only
# the regime's eval_chroms, P2 is the same root with --chroms naming every chromosome. Nothing is
# predicted twice.
#
#   mkdir -p slurm-logs && sbatch --array=0-22 competitors/avocado/slurm/predict.sh
#
#SBATCH --account=def-maxwl
#SBATCH --job-name=t50_pred
#SBATCH --output=slurm-logs/t50_pred_%A_%a.out
#SBATCH --error=slurm-logs/t50_pred_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

source /project/def-maxwl/mforooz/CANDII_t50/competitors/avocado/slurm/_env.sh

CHROM=$(sed -n "$((${SLURM_ARRAY_TASK_ID:-0} + 1))p" "$AVO/chroms.txt")
[ -z "$CHROM" ] && { echo "no chromosome for index ${SLURM_ARRAY_TASK_ID:-0}"; exit 1; }

echo "[t50_pred] chrom=$CHROM host=$(hostname)"
python "$AVO/predict.py" \
    --regime "$REGIME" --chrom "$CHROM" \
    --shared "$WS/ckpt/shared_chr20.pt" \
    --genome "$WS/ckpt/genome_${CHROM}.pt" \
    --out "${PRED:-$WS/pred/eic_val}"
