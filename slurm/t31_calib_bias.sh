#!/bin/bash
# t31 (a child of t30) — WHICH tracks carry the coverage bias calibration (a) found in the macro.
#
#   sbatch slurm/t31_calib_bias.sh
#
# Calibration (a) measured a -0.152 bias in the mid-training macro at the shipped 0.45% coverage,
# gone by ~7%. It could not say which tracks carried it, because `quick_eval` collapsed its
# per-track rows before returning. It now returns them and this job differences them.
#
# SHORT ON PURPOSE. Across (a)'s ten epochs the bias moved by 0.0057 at the shipped coverage and
# 0.0010 at 7%, while the model moved 0.0215 per epoch — the bias is a property of WHICH WINDOWS a
# coverage samples, not of how good the model is. So this trains only enough to leave the
# initialisation behind. One hour is generous; (a) spent three, almost all of it on 40 checks.
#
# --gres is the HARD-RULE MIG slice — never any other spec.
#SBATCH --account=def-maxwl
#SBATCH --job-name=candi_t31_bias
#SBATCH --output=slurm-logs/t31bias_%x_%j.out
#SBATCH --error=slurm-logs/t31bias_%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

set -uo pipefail
KIT="${KIT:-$HOME/projects/def-maxwl/$USER/CANDII}"
VENV="${VENV:-$HOME/projects/def-maxwl/$USER/candi_venv}"
REGIME="${REGIME:-configs/regime.eic_val.json}"
OUT="${OUT:-/scratch/$USER/candi_kit/calib_bias}"
LEVELS="${LEVELS:-2 8 32 128}"
STEPS="${STEPS:-100}"
SEED="${SEED:-0}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg WANDB_MODE=disabled
module load StdEnv/2023 python/3.10.13 >/dev/null 2>&1
source "$VENV/bin/activate" || { echo "[error] no venv at $VENV" >&2; exit 1; }
cd "$KIT"; mkdir -p "$OUT" slurm-logs
source "$KIT/slurm/_kit_pin.sh"
echo "[t31bias] regime=$REGIME levels='$LEVELS' steps=$STEPS seed=$SEED host=$(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi -L || true

python tools/calib_bias.py --regime "$REGIME" --out "$OUT" \
  --levels $LEVELS --epochs 1 --steps-per-epoch "$STEPS" --seed "$SEED"
rc=$?
echo "[t31bias] exit=$rc"
exit $rc
