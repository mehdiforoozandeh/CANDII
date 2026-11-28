#!/bin/bash
# t396 iteration 2: train one factorial run (one arm, one learning rate, one seed) on a GPU,
# inside an ONION code kit.
#
#   export KIT=/project/def-maxwl/mforooz/onion_kit/SC2A5
#   sbatch --time=<from the probe> <GPU form> --output=$OUT/train_%j.out \
#       $KIT/slurm/t396/train_arm.sh $KIT $KIT/tools/minicandi/configs/t396/<arm>.json <lr> \
#       <tables dir> <seed> $OUT [more train.py args]
#   GPU forms (the caller passes one; this header has no --gres line):
#     full  --gres=gpu:h100:1 --cpus-per-task=12 --mem=48000M
#     slice --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=2 --mem=10000M
#   -> $OUT/config.json (the arm config with "lr" replaced) and train.py's
#      $OUT/{ckpt.pt,run.json,log.jsonl,DONE}
#
# train.py runs with --device cuda --workers <allocated CPUs - 1> --stage-dir $SLURM_TMPDIR/stage.
# --workers is never below 1, because the batch stream at 0 workers differs from the one at any
# N > 0: a 1-CPU allocation is refused (exit 2). Rerunning the same command resumes from
# $OUT/ckpt.pt (train.py); train.py exits at once when DONE exists. The environment block is the
# one of slurm/minicandi/train.sh (no self-retry: a failed setup exits 3). DRY_RUN=1 prints the
# train.py command and exits (writes nothing). Never pass a comma-valued variable through
# sbatch --export; export it in the shell and use --export=ALL.
#SBATCH --account=def-maxwl
#SBATCH --job-name=sc_train
#SBATCH --time=03:00:00
#SBATCH --exclude=fc10713,fc10715
#SBATCH --export=ALL

set -euo pipefail
USAGE="usage: t396/train_arm.sh <kit> <arm config> <lr> <tables> <seed> <out> [train.py args]"
KIT="${1:?$USAGE}"
ARM_CONFIG="${2:?$USAGE}"
LR="${3:?$USAGE}"
TABLES="${4:?$USAGE}"
SEED="${5:?$USAGE}"
OUT="${6:?$USAGE}"
shift 6
EXTRA=("$@")
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be an integer" >&2; exit 2; }
[[ "$LR" =~ ^[0-9.]+([eE][-+]?[0-9]+)?$ ]] || { echo "lr must be a positive number: $LR" >&2; exit 2; }
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-1} - 1 ))
if [ "$WORKERS" -lt 1 ]; then
  echo "train_arm.sh: ${SLURM_CPUS_PER_TASK:-1} CPU(s) allocated; --workers must be >= 1, so" \
       "submit with --cpus-per-task >= 2" >&2
  exit 2
fi
CONFIG="$OUT/config.json"
TRAIN=(python3 "$KIT/tools/minicandi/train.py" --config "$CONFIG" --tables "$TABLES"
       --seed "$SEED" --out "$OUT" --device cuda --workers "$WORKERS"
       --stage-dir "${SLURM_TMPDIR:-/tmp}/stage" ${EXTRA[@]+"${EXTRA[@]}"})
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "${TRAIN[*]}"
  exit 0
fi
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "train_arm.sh: the job has no GPU; submit with a GPU form (--gres=gpu:...)" >&2
  exit 2
fi
mkdir -p "$OUT"
echo "=== t396 train_arm job ${SLURM_JOB_ID:-none} host=$(hostname) arm=$(basename "$ARM_CONFIG" .json) lr=$LR seed=$SEED $(date -u)"
setup() {
  local rc
  set +u; module load python/3.10.13; rc=$?; set -u
  [ "$rc" = 0 ] &&
  virtualenv --no-download "$SLURM_TMPDIR/venv" &&
  source "$SLURM_TMPDIR/venv/bin/activate" &&
  pip install --no-index --upgrade pip &&
  pip install --no-index -r "$KIT/requirements-fir.txt" &&
  pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
}
if ! setup; then
  echo "train_arm.sh: environment setup failed on $(hostname)"
  exit 3
fi
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
python3 - "$ARM_CONFIG" "$LR" "$CONFIG" <<'EOF'
import json, sys
src, lr, dst = sys.argv[1], float(sys.argv[2]), sys.argv[3]
cfg = json.load(open(src, encoding="utf-8"))
cfg["lr"] = lr
text = json.dumps(cfg, indent=1) + "\n"
try:
    old = open(dst, encoding="utf-8").read()
except FileNotFoundError:
    old = None
if old is not None and old != text:
    sys.exit(f"{dst} exists with another config; refusing to overwrite a run's config")
if old is None:
    open(dst, "w", encoding="utf-8").write(text)
EOF
"${TRAIN[@]}"
echo "=== done $(date -u)"
