#!/bin/bash
# t146: train one run of the 42-row CAC testbed table (`python3 tools/t146/tasks.py tasks`) on a
# 10 GB MIG slice. One shared design-D model over all 7 tracks, count space, depth 7.5M held out,
# 12,000-step cap, early stopping on chr22 by the correction loss only (tools/t146/train.py).
# Scoring is a separate CPU job (slurm/t146/score.sh): the CRPS pass needs many cores, the slice
# allows one.
#
#   python3 $KIT/tools/t146/train.py train-index <manifest> <covariates.tsv> <cache_dir> \
#       <out_dir>/runs <index> --features <features_dir> [--max-steps $6] [--eval-every $7]
#                                       -> <out_dir>/runs/<run_name>/ckpt.pt, TRAIN_DONE
# A task whose TRAIN_DONE exists exits 0 without work.
#
# NO --export. Positional arguments only; the task index is $SLURM_ARRAY_TASK_ID. Paths (Nibi):
#   KIT=/project/def-maxwl/mforooz/t146/code/K<n>   (a snapshot of this branch + GIT_SHA + wheels/)
#   PRODUCTS=/project/def-maxwl/mforooz/t112_cf/products
#   CACHE=/project/def-maxwl/mforooz/t118/ladder/cache   (t118's count cache, read only)
#   OUT=/project/def-maxwl/mforooz/t146               (the out dir must be t146 or under it)
#   FEATURES=/project/def-maxwl/mforooz/t146/features (slurm/t146/features.sh; map / dna arms)
#
# THE VENV is t118's pinned recipe (python/3.10.13, requirements-fir.txt, then
# requirements-pypi.txt from $KIT/wheels), built per task in $SLURM_TMPDIR, as slurm/t136/train.sh.
#
# RESOURCES. One 10 GB MIG slice, 1 core, 32 GB (smoke MaxRSS 3-11 GB, memmap pages counted) (inside the slice's billing: <= 1 CPU, <= 36G),
# --time set in the header (Nibi's default is 15 min). Nibi nodes c128, c166, c537, c371, c329
# excluded (the t126/t136 list).
#
# DRY_RUN=1 prints the python line and exits 0 before any module load.
#
# Usage, from the Nibi login node:
#   smoke (8 code paths, reduced steps, its own tree):
#     sbatch --parsable --array=<idx> --time=<t> --output=$OUT/smoke/logs/%x_%A_%a.out \
#         $KIT/slurm/t146/train.sh $KIT $PRODUCTS $CACHE $OUT/smoke $FEATURES <steps> <eval_every>
#   full (one array; then score.sh with --dependency=aftercorr on the LIVE array):
#     sbatch --parsable --array=0-41%12 --output=$OUT/logs/%x_%A_%a.out \
#         $KIT/slurm/t146/train.sh $KIT $PRODUCTS $CACHE $OUT $FEATURES
#SBATCH --account=def-maxwl
#SBATCH --job-name=t146_train
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --exclude=c128,c166,c537,c371,c329
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000M

set -euo pipefail
USAGE="usage: t146/train.sh <kit_dir> <products_dir> <cache_dir> <out_dir> <features_dir> [max_steps [eval_every]]"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
CACHE="${3:?$USAGE}"
OUT="${4:?$USAGE}"
FEATURES="${5:?$USAGE}"
MAX_STEPS="${6:-}"
EVAL_EVERY="${7:-}"
IDX="${SLURM_ARRAY_TASK_ID:?array task only}"
[[ "$IDX" =~ ^[0-9]+$ ]] || { echo "SLURM_ARRAY_TASK_ID '$IDX' is not an index" >&2; exit 2; }
[[ -z "$MAX_STEPS" || "$MAX_STEPS" =~ ^[0-9]+$ ]] || { echo "max_steps '$MAX_STEPS' is not an integer" >&2; exit 2; }
[[ -z "$EVAL_EVERY" || "$EVAL_EVERY" =~ ^[0-9]+$ ]] || { echo "eval_every '$EVAL_EVERY' is not an integer" >&2; exit 2; }
case "$OUT" in
  */t146|*/t146/*) ;;
  *) echo "out dir $OUT is not t146 or under it; t146 writes only there" >&2; exit 2 ;;
esac
MANIFEST="$PRODUCTS/MANIFEST.tsv"
COVARIATES="$KIT/tools/t118/covariates.tsv"
RUNS="$OUT/runs"
STEP_ARGS=()
[ -z "$MAX_STEPS" ] || STEP_ARGS+=(--max-steps "$MAX_STEPS")
[ -z "$EVAL_EVERY" ] || STEP_ARGS+=(--eval-every "$EVAL_EVERY")

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "python3 $KIT/tools/t146/train.py train-index $MANIFEST $COVARIATES $CACHE $RUNS $IDX --features $FEATURES ${STEP_ARGS[*]:-}"
  exit 0
fi

[ -f "$KIT/tools/t146/train.py" ] || { echo "no tools/t146/train.py under $KIT" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "no $MANIFEST" >&2; exit 2; }
[ -d "$CACHE" ] || { echo "no cache dir $CACHE" >&2; exit 2; }
KIT=$(cd "$KIT" && pwd)
mkdir -p "$RUNS"
echo "=== t146 train idx=$IDX job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX host=$(hostname) $(date -u)"
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || echo unknown)"

set +u; module load python/3.10.13; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$KIT/requirements-fir.txt"
pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
export PYTHONPATH="$KIT/src"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
case "$(python3 -c 'import candi; print(candi.__file__)')" in
  "$KIT"/src/*) ;;
  *) echo "candi does not import from $KIT/src" >&2; exit 3 ;;
esac

python3 "$KIT/tools/t146/train.py" train-index "$MANIFEST" "$COVARIATES" "$CACHE" "$RUNS" "$IDX" \
    --features "$FEATURES" ${STEP_ARGS[@]+"${STEP_ARGS[@]}"}
echo "=== done $(date -u)"
