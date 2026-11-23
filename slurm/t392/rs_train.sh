#!/bin/bash
# t392 C8: train one seed of the read-start miniCANDI inside an ONION code kit.
#
#   sbatch --output=<out>/train_%j.out \
#       <kit>/slurm/t392/rs_train.sh <kit dir> <config.json> <tables dir> <seed> <out dir>
#   -> <out>/{ckpt.pt,run.json,log.jsonl,DONE}
#
# The config sets `read_start` (tools/minicandi/configs/t392/rs_halves_mix_avg_pearson.json).
# tools/minicandi/train.py runs res_dataset.train for such a config (the read-start dataset and
# read_start.read_start_loss); it is called through train.py and not `res_dataset.py train`
# because only train.py has --stage-dir. Staging is on, as in the reference jobs (Fir
# 5264674/80/87 staged their counts and average_h5 to $SLURM_TMPDIR/stage): train.py copies the
# read-start starts.h5 files and read_start_average_h5 to $SLURM_TMPDIR/stage, or logs
# "staging skipped" and reads them in place when the node's local disk is too small.
# Header = the reference jobs' allocation (12 CPUs, 48000M, no GPU); --workers is CPUs - 1. The
# batch stream does not depend on the worker count (any N > 0), so it equals the reference's 8.
# Rerunning the same command resumes from <out>/ckpt.pt and skips when <out>/DONE exists.
# DRY_RUN=1 prints the command and exits. Nothing goes via --export.
#SBATCH --account=def-maxwl
#SBATCH --job-name=rs392_train
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --gres=none

set -euo pipefail
USAGE="usage: rs_train.sh <kit dir> <config.json> <tables dir> <seed> <out dir>"
KIT="${1:?$USAGE}"
CONFIG="${2:?$USAGE}"
TABLES="${3:?$USAGE}"
SEED="${4:?$USAGE}"
OUT="${5:?$USAGE}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be an integer" >&2; exit 2; }
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-12} - 1 ))
[ "$WORKERS" -ge 0 ] || WORKERS=0
TRAIN=(python3 "$KIT/tools/minicandi/train.py" --config "$CONFIG" --tables "$TABLES"
       --seed "$SEED" --out "$OUT" --device cpu --workers "$WORKERS"
       --stage-dir "${SLURM_TMPDIR:-/tmp}/stage")
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "${TRAIN[*]}"
  exit 0
fi
grep -q '"read_start": true' "$CONFIG" || { echo "$CONFIG does not set read_start" >&2; exit 2; }
mkdir -p "$OUT"
echo "=== rs392 train job ${SLURM_JOB_ID:-none} host=$(hostname) seed=$SEED $(date -u)"
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
echo "=== done $(date -u)"
