#!/bin/bash
# t62 — the two leaderboard CANDI training runs, all three heads (count, signal, peak).
#
#   sbatch --export=ALL,MODE=chr19 slurm/t62_train.sh
#   sbatch --export=ALL,MODE=gw --time=143:00:00 slurm/t62_train.sh
#
# MODE=chr19: the mainline recipe (regime.eic_val, train chr19 / eval chr21), 25 epochs,
#             the only change from the recorded arm being --heads count,signal,peak.
# MODE=gw:    regime.eic_gw — identical to eic_val except train_chroms is every store
#             chromosome except chr21, so the eval chromosome stays held out. The 143 h wall
#             IS the budget rule (PI 2026-08-27): the cap equals Avocado's recorded training
#             total on this same MIG slice, and at ~32 h/genome-wide epoch it buys 4 epochs,
#             so --epochs 4 with --eval-every 1 lets the loop finish inside the wall and the
#             monitor pick the best checkpoint; the trainer has no resume, so a wall kill
#             mid-loop would forfeit the run's tail — the epoch count must fit the wall.
#
# --gres is the HARD-RULE MIG slice — never any other spec.
#SBATCH --account=def-maxwl
#SBATCH --job-name=candi_t62
#SBATCH --output=slurm-logs/t62_%x_%j.out
#SBATCH --error=slurm-logs/t62_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

set -uo pipefail
KIT="${KIT:-$HOME/projects/def-maxwl/$USER/CANDII_t62}"
VENV="${VENV:-$HOME/projects/def-maxwl/$USER/candi_venv}"
MODE="${MODE:?set MODE=chr19 or MODE=gw}"
OUT="${OUT:-/scratch/$USER/t62_candi/$MODE}"
SEED="${SEED:-0}"

case "$MODE" in
  chr19) REGIME="configs/regime.eic_val.json"; EPOCHS=25; EVAL_EVERY=3 ;;
  gw)    REGIME="configs/regime.eic_gw.json";  EPOCHS=4;  EVAL_EVERY=1 ;;
  *) echo "[error] MODE must be chr19 or gw, got '$MODE'" >&2; exit 2 ;;
esac
TAG="t62_${MODE}_csp_s${SEED}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg WANDB_MODE=disabled
module load StdEnv/2023 python/3.10.13 >/dev/null 2>&1
source "$VENV/bin/activate" || { echo "[error] no venv at $VENV" >&2; exit 1; }
cd "$KIT"; mkdir -p "$OUT" slurm-logs
source "$KIT/slurm/_kit_pin.sh"

echo "[t62] mode=$MODE regime=$REGIME epochs=$EPOCHS eval_every=$EVAL_EVERY seed=$SEED tag=$TAG"
echo "[t62] host=$(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi -L || true

python -m candi.train \
  --store "$REGIME" --out-dir "$OUT" \
  --tag "$TAG" --offset on --seed "$SEED" --weight-decay 0.0 \
  --d-model 288 --dsf-sampling uniform \
  --epochs "$EPOCHS" --eval-every "$EVAL_EVERY" --batch-size 8 --full-coverage \
  --heads count,signal,peak
rc=$?
echo "[t62] DONE mode=$MODE rc=$rc"
