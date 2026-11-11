#!/bin/bash
# miniCANDI: train one (arm, seed) run, then score it (tools/minicandi/train.py, then
# tools/onion_eval/score.py on the same run directory). One job, no array.
#
#   CPU form (the header below: 8 cores x 4000 MB, no GPU):
#     sbatch --time=03:00:00 --output=$OUT/train_%j.out \
#         $KIT/slurm/minicandi/train.sh $KIT $CONFIG $TABLES $SEED $OUT cpu [$EVALSET]
#   GPU form (the fixed slice only; 1 CPU on Fir, 2 on Nibi; on Fir also --exclude=fc10713,fc10715):
#     sbatch --time=03:00:00 --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=1 \
#         --mem=10000M --output=$OUT/train_%j.out \
#         $KIT/slurm/minicandi/train.sh $KIT $CONFIG $TABLES $SEED $OUT gpu [$EVALSET]
#   -> $OUT/{ckpt.pt,run.json,log.jsonl,DONE} and, with EVALSET, $OUT/{scores.json,suffstats.npz}
#
# Rerunning the same command resumes from $OUT/ckpt.pt (train.py) and skips training when DONE
# exists. EXTRA_TRAIN_ARGS (space-separated, e.g. "--max-windows 100000 --val-every 20000") is
# passed to train.py. Never pass a comma-valued variable through sbatch --export; export it in the
# shell and use --export=ALL. DRY_RUN=1 prints the commands and exits.
#SBATCH --account=def-maxwl
#SBATCH --job-name=minicandi_train
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --gres=none

set -euo pipefail
USAGE="usage: minicandi/train.sh <kit_dir> <config.json> <tables_dir> <seed> <out_dir> <cpu|gpu> [evalset.json]"
KIT="${1:?$USAGE}"
CONFIG="${2:?$USAGE}"
TABLES="${3:?$USAGE}"
SEED="${4:?$USAGE}"
OUT="${5:?$USAGE}"
MODE="${6:?$USAGE}"
EVALSET="${7:-}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be an integer" >&2; exit 2; }
case "$MODE" in
  cpu) DEVICE=cpu ;;
  gpu) DEVICE=cuda ;;
  *) echo "mode must be cpu or gpu" >&2; exit 2 ;;
esac
read -r -a EXTRA <<< "${EXTRA_TRAIN_ARGS:-}"
TRAIN=(python3 "$KIT/tools/minicandi/train.py" --config "$CONFIG" --tables "$TABLES"
       --seed "$SEED" --out "$OUT" --device "$DEVICE" ${EXTRA[@]+"${EXTRA[@]}"})
SCORE=(python3 "$KIT/tools/onion_eval/score.py" --run "$OUT" --evalset "$EVALSET" --out "$OUT")
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "${TRAIN[*]}"
  [ -n "$EVALSET" ] && echo "${SCORE[*]}"
  exit 0
fi
if [ "$MODE" = gpu ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "mode gpu but the job has no GPU: submit with --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" >&2
  exit 2
fi
mkdir -p "$OUT"
echo "=== minicandi train job ${SLURM_JOB_ID:-none} host=$(hostname) mode=$MODE $(date -u)"
set +u; module load python/3.10.13; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$KIT/requirements-fir.txt"
pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
"${TRAIN[@]}"
if [ -z "$EVALSET" ]; then
  echo "no evalset given: scoring skipped"
elif [ -f "$OUT/scores.json" ]; then
  echo "skip scoring: $OUT/scores.json exists"
else
  "${SCORE[@]}"
fi
echo "=== done $(date -u)"
