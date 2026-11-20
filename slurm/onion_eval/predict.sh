#!/bin/bash
# SP2a (plan t152, Amendment 10, pinned interface M): the forwards of one run, one shard, into a
# prediction store (tools/onion_eval/predict_store.py run). The metric phase (score.py --phase
# metrics) reads the store later in CPU workers.
#
#   sbatch --output=<logs>/%x_%j.out [resources] $KIT/slurm/onion_eval/predict.sh \
#       <kit_dir> <run_dir> <evalset.json> <store_dir>
#   -> <store_dir>/{STORE.json,<sha1>.npz,guardrail.json (shard 0),PREDICTED_<i>of<N>.json}
# Variables:
#   ONION_SHARD        i/N (default 0/1); request j of the scoring belongs to shard j mod N
#   ONION_INFER_BATCH  windows per forward (default: the run's batch_size, the legacy batch)
#   ONION_DEVICE       cuda|cpu (default: cuda when the job has a GPU allocated, else cpu)
#   ONION_STAGE        1 (default): copy the h5 files the shard reads to $SLURM_TMPDIR/stage first
#   ONION_PREDICT_THREADS  torch and BLAS threads (default: the allocated CPUs; the forward speeds
#                      of FORWARD_SPEED.md were measured with 8 threads on 8 CPUs)
#   DRY_RUN=1          print the command and each variable, run nothing
# Job shapes (interface M): GPU slice --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=4
# --mem=16000M (rung 1: batch 32, 8 shards, --time=01:00:00; rung 4 at 256 pair regions: batch 32,
# 4 shards, --time=02:00:00); CPU --gres=none --cpus-per-task=8 --mem=32000M, batch 8, 1 shard,
# --time=06:00:00. A shard already stored is skipped request by request (a rerun stores nothing new).
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_predict
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --gres=none

set -euo pipefail
USAGE="usage: onion_eval/predict.sh <kit_dir> <run_dir> <evalset.json> <store_dir>"
KIT="${1:?$USAGE}"
RUN="${2:?$USAGE}"
EVALSET="${3:?$USAGE}"
STORE="${4:?$USAGE}"
[ $# -eq 4 ] || { echo "$USAGE" >&2; exit 2; }

SHARD="${ONION_SHARD:-0/1}"
case "$SHARD" in
  [0-9]*/[0-9]*) ;;
  *) echo "ONION_SHARD=$SHARD: want i/N" >&2; exit 2 ;;
esac
if [ -n "${ONION_DEVICE:-}" ]; then
  DEVICE="$ONION_DEVICE"
elif [ -n "${SLURM_JOB_GPUS:-}" ] || [ "${SLURM_GPUS_ON_NODE:-0}" != 0 ]; then
  DEVICE=cuda
else
  DEVICE=cpu
fi
case "$DEVICE" in cuda|cpu) ;; *) echo "ONION_DEVICE=$DEVICE: want cuda or cpu" >&2; exit 2 ;; esac
STAGE="${ONION_STAGE:-1}"
T="${ONION_PREDICT_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

CMD=(python3 "$KIT/tools/onion_eval/predict_store.py" run --run "$RUN" --evalset "$EVALSET"
     --pred-store "$STORE" --shard "$SHARD" --device "$DEVICE")
[ -z "${ONION_INFER_BATCH:-}" ] || CMD+=(--infer-batch "$ONION_INFER_BATCH")
if [ "$STAGE" = 1 ] && [ -n "${SLURM_TMPDIR:-}" ]; then
  CMD+=(--stage-dir "$SLURM_TMPDIR/stage")
fi

echo "ONION_SHARD=$SHARD ONION_DEVICE=$DEVICE ONION_INFER_BATCH=${ONION_INFER_BATCH:-run batch_size}" \
     "ONION_STAGE=$STAGE ONION_PREDICT_THREADS=$T"
if [ "${DRY_RUN:-0}" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
[ -f "$EVALSET" ] || { echo "no evalset $EVALSET" >&2; exit 2; }
[ -f "$RUN/ckpt.pt" ] || { echo "no checkpoint $RUN/ckpt.pt" >&2; exit 2; }

T0=$(date +%s)
echo "=== onion predict job ${SLURM_JOB_ID:-local} host=$(hostname) $(date -u)"
[ -z "${SLURM_TMPDIR:-}" ] || df -h "$SLURM_TMPDIR" | tail -1
set +u; module load python/3.10.13; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$KIT/requirements-fir.txt"
pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# Threads pinned as score.sh pins them (torch and every BLAS; idle OpenMP threads sleep).
export OMP_NUM_THREADS="$T" OPENBLAS_NUM_THREADS="$T" FLEXIBLAS_NUM_THREADS="$T" MKL_NUM_THREADS="$T" \
  OMP_WAIT_POLICY=PASSIVE
T1=$(date +%s)
"${CMD[@]}"
T2=$(date +%s)
echo "timing by phase (job): environment $((T1 - T0)) s, predict $((T2 - T1)) s, total $((T2 - T0)) s"
echo "=== done $(date -u)"
