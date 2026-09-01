#!/bin/bash
# Avocado stage 2 -- the paper's two-stage fit. One GPU per task.
#
#   MODE=shared   ONE task, on the regime's train_chroms[0], everything trainable, 60 epochs
#   MODE=genome   ONE TASK PER EVAL CHROMOSOME (three), shared parameters frozen, per-chromosome
#                 genomic factors, 30 epochs
#
# RETARGETED 2026-08-31. Both counts moved, and both for a ruled reason:
#
#   * The joint fit was on chr20. chr20 is now an EVAL chromosome and that stage fits transferable
#     parameters, so it violates Rule 2. It moves to the regime's train_chroms -- chr19 under
#     eic.19, matching CANDI's own dev scope (§3.2). Nothing is hard-coded: _env.sh reads it.
#   * The genome stage was `--array 0-22`. §4 blanks Avocado's genome-wide cell and rules that a
#     blanked cell is NOT COMPUTED, so Avocado is only ever predicted on chr20+21+22 -- three
#     genomic-factor fits per regime, not 23 (§12.2, corrected 2026-08-30). The other 20 would fit
#     factors for positions nothing is ever scored at.
#
# The joint fit's own chromosome comes free with the shared run and is NOT one of the three.
#
# 60/30 is the HALVED budget RIVALS_PLAN.md 7.1 asks for: 005 ran 120/60 and measured that its
# held-out MSE bottoms out near epoch 25-34 (0.07177) and drifts up afterwards (0.07572 at epoch
# 119). Half is not a saving on a converged run; it is the better run.
#
# NO CHECKPOINT IS SELECTED HERE, AND THAT IS AN OPEN ITEM, NOT A SETTING. §5 asks every trainable
# method to select its best checkpoint on V_. train.py logs a `holdout_mse` on a deterministic
# 1-in-50 (position, track) mask over the 267 TRAINING columns and then writes the LAST epoch's
# weights regardless; there is no "best" checkpoint and no V_ read anywhere in the loop. Adding one
# is a design decision (what does Avocado evaluate on V_, and how often, when a V_ pass needs the
# genomic factors of the eval chromosomes that stage 2 has not fitted yet?) and is the PI's, not
# this script's.
#
# The gres is the AGENTS.md 3.13 MIG slice. 005 used a full H100 and estimated a 1g.10gb slice at
# ~7x slower -- MEASURE the smoke run's steps/s before trusting that number here, and if the
# projection does not fit, raise the rule rather than reaching for another spec.
#
#   sbatch --array=0-0  --export=ALL,MODE=shared,EPOCHS=60 <this>
#   sbatch --array=0-2  --export=ALL,MODE=genome,EPOCHS=30 <this>
#
#SBATCH --account=def-maxwl
#SBATCH --job-name=t81_avo_train
#SBATCH --output=slurm-logs/t81_avo_train_%A_%a.out
#SBATCH --error=slurm-logs/t81_avo_train_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6

source "${SLURM_SUBMIT_DIR:-$PWD}/competitors/avocado/slurm/_env.sh"

MODE="${MODE:?set MODE=shared|genome}"
case "$MODE" in
  shared) CHROM="$JOINT_CHROM" ;;
  genome) CHROM="${EVAL_CHROMS[${SLURM_ARRAY_TASK_ID:-0}]:-}" ;;
  *) echo "MODE must be shared or genome, got $MODE" >&2; exit 2 ;;
esac
[ -z "$CHROM" ] && { echo "no chromosome for index ${SLURM_ARRAY_TASK_ID:-0}; this regime has ${#EVAL_CHROMS[@]} eval chromosome(s): ${EVAL_CHROMS[*]}"; exit 1; }

ARGS=(--regime "$REGIME" --chrom "$CHROM" --mode "$MODE"
      --data-root "$WS/binned"
      --out "$WS/ckpt/${MODE}_${CHROM}.pt"
      --log "$WS/logs/train_${MODE}_${CHROM}.jsonl"
      --epochs "${EPOCHS:?set EPOCHS}" --batch-positions "${BATCHPOS:-1024}"
      --lr "${LR:-1e-3}" --genome-lr "${GENOMELR:-1e-2}" --seed "${SEED:-0}"
      --max-hours "${MAXHOURS:-11.5}")
[ "$MODE" = "genome" ] && ARGS+=(--init "$WS/ckpt/shared_${JOINT_CHROM}.pt")
[ -n "${SMOKESTEPS:-}" ] && ARGS+=(--smoke-steps "$SMOKESTEPS")

echo "[avo_train] regime=$REGIME_NAME mode=$MODE chrom=$CHROM joint=$JOINT_CHROM host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python "$AVO/train.py" "${ARGS[@]}"
