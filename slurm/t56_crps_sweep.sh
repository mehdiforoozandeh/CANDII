#!/bin/bash
# t56 — the sampled-CRPS k-curve on the real t49 P1 tracks. THREE STEPS, in this order.
#
#   # 1. chr21 truth for every declared track, once (the only step that opens the store)
#   MODE=truth sbatch --time=2:00:00 --mem=32G --cpus-per-task=2 slurm/t56_crps_sweep.sh
#
#   # 2. the sweep: 45 tracks x 4 count-arm methods x k in {10,25,50,100,250} x seed in {0,1,2}
#   MODE=sweep sbatch --array=0-14 --time=8:00:00 --mem=16G --cpus-per-task=1 \
#       slurm/t56_crps_sweep.sh
#
#   # 3. where the count arm's hours actually go, on one real track
#   MODE=profile sbatch --time=1:00:00 --mem=16G --cpus-per-task=1 slurm/t56_crps_sweep.sh
#
# ONE CPU PER SWEEP TASK because the work is numpy RNG and a sort, both single-threaded; asking for
# more would idle them. 15 shards over 45 tracks is 3 tracks x 4 methods each.
#
# READ-ONLY ON EVERYTHING t49 OWNS. This job opens `p1v2/n1e4/preds` and the store and writes
# nowhere near either; the running P2 score array is untouched.
#
# NO --gres, unlike the scoring jobs. There is no model here: `truth` reads the store and `sweep`
# is numpy, so a GPU would be requested and left idle, and the CPU partition schedules sooner.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t56_crps
#SBATCH --output=slurm-logs/t56_crps_%A_%a.out
#SBATCH --error=slurm-logs/t56_crps_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

set -uo pipefail

KIT="${KIT:-/project/def-maxwl/$USER/CANDII_t56}"
VENV="${VENV:-/project/def-maxwl/$USER/candi_venv}"
REGIME="${REGIME:-configs/regime.eic_val.json}"
PREDS="${PREDS:-/project/def-maxwl/$USER/t49_baselines/p1v2/n1e4/preds}"
WORK="${WORK:-/project/def-maxwl/$USER/t56_crps}"
CHROMS="${CHROMS:-chr21}"
MODE="${MODE:-sweep}"
NSHARDS="${NSHARDS:-15}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
module load StdEnv/2023 python/3.10.13 >/dev/null 2>&1
source "$VENV/bin/activate" || { echo "[error] no venv at $VENV" >&2; exit 1; }
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
export PYTHONPATH="$KIT/src:$KIT"

case "$REGIME" in *eic_test*) echo "[t56] REFUSING: $REGIME is the B-pair regime (A4)"; exit 2;; esac
mkdir -p "$WORK/truth" "$WORK/sweep" slurm-logs
echo "[t56] host=$(hostname) commit=$(git rev-parse --short HEAD) mode=$MODE"

case "$MODE" in
  truth)
    python tools/t56_crps_sweep.py truth --store "$REGIME" --chroms "$CHROMS" \
        --out "$WORK/truth" ;;
  sweep)
    python tools/t56_crps_sweep.py sweep --store "$REGIME" --chroms "$CHROMS" \
        --truth "$WORK/truth" --preds "$PREDS" --out "$WORK/sweep" \
        --shard "${SLURM_ARRAY_TASK_ID:-0}" --nshards "$NSHARDS" ;;
  profile)
    python tools/t56_crps_sweep.py profile --store "$REGIME" --chroms "$CHROMS" \
        --truth "$WORK/truth" --preds "$PREDS" --method "${METHOD:-avg}" \
        --track-index "${TRACK_INDEX:-0}" --out "$WORK/profile_${METHOD:-avg}.json" ;;
  *) echo "[t56] unknown MODE=$MODE"; exit 2 ;;
esac
rc=$?
echo "[t56] mode=$MODE exit=$rc"
exit $rc
