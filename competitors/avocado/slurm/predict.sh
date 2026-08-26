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

# A train.py job that hits MAXHOURS writes a .partial and exits 0 -- deliberately, so a resubmit
# continues it. That means `--dependency=afterok` on the genome arrays is NOT sufficient to prove
# the checkpoint exists, and a chained predict array would otherwise start against a chromosome
# that never finished. Refuse here, naming the chromosome, rather than letting torch.load raise
# somewhere less legible.
CK="$WS/ckpt/genome_${CHROM}.pt"
if [ ! -s "$CK" ]; then
    if [ -s "$CK.partial" ]; then
        echo "[t50_pred] $CHROM: only $CK.partial exists -- its genome fit stopped at a wall-clock" \
             "deadline and has not finished. Re-submit that chromosome's train.sh array task to" \
             "continue it, then re-run this one." >&2
    else
        echo "[t50_pred] $CHROM: no $CK and no .partial -- its genome fit never ran." >&2
    fi
    exit 1
fi

python "$AVO/predict.py" \
    --regime "$REGIME" --chrom "$CHROM" \
    --shared "$WS/ckpt/shared_chr20.pt" \
    --genome "$WS/ckpt/genome_${CHROM}.pt" \
    --out "${PRED:-$WS/pred/eic_val}"
