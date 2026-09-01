#!/bin/bash
# Avocado stage 3 -- write the RIVALS_PLAN.md 4.1 prediction root, one eval chromosome per task.
#
# THREE array tasks, not 23. §4 blanks Avocado's `genome-wide` cell and rules that a blanked cell is
# not computed, so the root covers the regime's eval_chroms and nothing else. The old header's "the
# root covers all 23 chromosomes, which serves BOTH protocols: P1 ... P2 ..." described the retired
# P1/P2 naming (§9) and a scope that no longer exists.
#
#   mkdir -p slurm-logs && sbatch --array=0-2 competitors/avocado/slurm/predict.sh
#
#SBATCH --account=def-maxwl
#SBATCH --job-name=t81_avo_pred
#SBATCH --output=slurm-logs/t81_avo_pred_%A_%a.out
#SBATCH --error=slurm-logs/t81_avo_pred_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

source "${SLURM_SUBMIT_DIR:-$PWD}/competitors/avocado/slurm/_env.sh"

CHROM="${EVAL_CHROMS[${SLURM_ARRAY_TASK_ID:-0}]:-}"
[ -z "$CHROM" ] && { echo "no chromosome for index ${SLURM_ARRAY_TASK_ID:-0}; this regime has ${#EVAL_CHROMS[@]} eval chromosome(s): ${EVAL_CHROMS[*]}"; exit 1; }

echo "[avo_pred] regime=$REGIME_NAME chrom=$CHROM host=$(hostname)"

# A train.py job that hits MAXHOURS writes a .partial and exits 0 -- deliberately, so a resubmit
# continues it. That means `--dependency=afterok` on the genome arrays is NOT sufficient to prove
# the checkpoint exists, and a chained predict array would otherwise start against a chromosome
# that never finished. Refuse here, naming the chromosome, rather than letting torch.load raise
# somewhere less legible.
CK="$WS/ckpt/genome_${CHROM}.pt"
if [ ! -s "$CK" ]; then
    if [ -s "$CK.partial" ]; then
        echo "[avo_pred] $CHROM: only $CK.partial exists -- its genome fit stopped at a wall-clock" \
             "deadline and has not finished. Re-submit that chromosome's train.sh array task to" \
             "continue it, then re-run this one." >&2
    else
        echo "[avo_pred] $CHROM: no $CK and no .partial -- its genome fit never ran." >&2
    fi
    exit 1
fi

python "$AVO/predict.py" \
    --regime "$REGIME" --chrom "$CHROM" \
    --shared "$WS/ckpt/shared_${JOINT_CHROM}.pt" \
    --genome "$WS/ckpt/genome_${CHROM}.pt" \
    --out "${PRED:-$WS/pred}"
