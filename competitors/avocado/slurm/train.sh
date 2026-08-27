#!/bin/bash
# t50 stage 2 -- Avocado's two-stage fit. One GPU per task.
#
#   MODE=shared   one task, chr20, everything trainable, 60 epochs
#   MODE=genome   23 tasks, shared parameters frozen, per-chromosome genomic factors, 30 epochs
#
# 60/30 is the HALVED budget RIVALS_PLAN.md 7.1 asks for: 005 ran 120/60 and measured that its
# chr20 held-out MSE bottoms out near epoch 25-34 (0.07177) and drifts up afterwards (0.07572 at
# epoch 119). Half is not a saving on a converged run; it is the better run.
#
# The gres is the AGENTS.md 3.13 MIG slice. 005 used a full H100 and estimated a 1g.10gb slice at
# ~7x slower -- MEASURE the smoke run's steps/s before trusting that number here, and if the
# projection does not fit, raise the rule rather than reaching for another spec.
#
#   sbatch --array=0-0 --export=ALL,MODE=shared,CHROMFILE=$WS/chrom20.txt,EPOCHS=60 <this>
#   sbatch --array=0-22 --export=ALL,MODE=genome,EPOCHS=30 <this>
#
#SBATCH --account=def-maxwl
#SBATCH --job-name=t50_train
#SBATCH --output=slurm-logs/t50_train_%A_%a.out
#SBATCH --error=slurm-logs/t50_train_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6

source /project/def-maxwl/mforooz/CANDII_t50/competitors/avocado/slurm/_env.sh

MODE="${MODE:?set MODE=shared|genome}"
CHROMFILE="${CHROMFILE:-$AVO/chroms.txt}"
CHROM=$(sed -n "$((${SLURM_ARRAY_TASK_ID:-0} + 1))p" "$CHROMFILE")
[ -z "$CHROM" ] && { echo "no chromosome for index ${SLURM_ARRAY_TASK_ID:-0}"; exit 1; }

ARGS=(--regime "$REGIME" --chrom "$CHROM" --mode "$MODE"
      --data-root "$WS/binned"
      --out "$WS/ckpt/${MODE}_${CHROM}.pt"
      --log "$WS/logs/train_${MODE}_${CHROM}.jsonl"
      --epochs "${EPOCHS:?set EPOCHS}" --batch-positions "${BATCHPOS:-1024}"
      --lr "${LR:-1e-3}" --genome-lr "${GENOMELR:-1e-2}" --seed "${SEED:-0}"
      --max-hours "${MAXHOURS:-11.5}")
[ "$MODE" = "genome" ] && ARGS+=(--init "$WS/ckpt/shared_chr20.pt")
[ -n "${SMOKESTEPS:-}" ] && ARGS+=(--smoke-steps "$SMOKESTEPS")

echo "[t50_train] mode=$MODE chrom=$CHROM host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python "$AVO/train.py" "${ARGS[@]}"
