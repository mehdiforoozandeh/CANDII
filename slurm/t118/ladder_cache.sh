#!/bin/bash
# t118 ladder: build the memory-mapped .npy cache of one product (both spaces). One array task =
# one product; 130 tasks.
#
# WHAT ONE TASK DOES.
#   python3 $KIT/tools/t118/ladder/train.py cache <products_dir>/MANIFEST.tsv <products_dir> \
#       <cache_dir> <index>
# = data.build_cache for the index-th of all 130 products sorted by pid (the two DNase MAPQ arms
# are cached too, harmlessly). It checks each npz md5 against the manifest and writes
# <cache_dir>/<pid>__<space>.npy then .json (tmp + rename). A task whose two json files already
# exist exits 0 without work; build_cache also skips a space whose json exists.
#
# THE PRODUCT ORDER. train.py cache is the authority. The early exit before the venv reads the pid
# from the manifest's `pid` column sorted in C byte order (= python's sorted() on these ASCII pids)
# only to skip a product that is already cached.
#
# NO --export. Every input is a POSITIONAL argument; the task index is $SLURM_ARRAY_TASK_ID.
# The cache dir the foreman passes is /project/def-maxwl/mforooz/t118/ladder/cache (~126 GB).
#
# THE VENV is the repo's pinned recipe, copied from slurm/t118/qm_rungs.sh (python/3.10.13,
# requirements-fir.txt, then requirements-pypi.txt from $KIT/wheels), built per task in
# $SLURM_TMPDIR.
#
# RESOURCES. CPU only, no GPU request. 1 core, 8 GB (one chromosome's array at a time), 1 h. Peak
# memory from `sacct -o MaxRSS` (no /usr/bin/time on compute nodes).
#
# DRY_RUN=1 prints the python command line and exits 0 before any module load; nothing is checked.
#
# Usage, from the Nibi login node, after snapshotting the repo to $KIT:
#   OUT=/project/def-maxwl/mforooz/t118/ladder; mkdir -p $OUT/logs $OUT/cache
#   sbatch --test-only --array=0-129%40 --output=$OUT/logs/%x_%A_%a.out \
#       $KIT/slurm/t118/ladder_cache.sh $KIT $PRODUCTS $OUT/cache
#   sbatch --parsable  --array=0-129%40 --output=$OUT/logs/%x_%A_%a.out \
#       $KIT/slurm/t118/ladder_cache.sh $KIT $PRODUCTS $OUT/cache
#   # then verify: ls $OUT/cache/*.json | wc -l  -> 260, before submitting ladder_train.sh
#SBATCH --account=def-maxwl
#SBATCH --job-name=t118L_cache
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000M

set -euo pipefail

USAGE="usage: ladder_cache.sh <kit_dir> <products_dir> <cache_dir>"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
CACHE="${3:?$USAGE}"
IDX="${SLURM_ARRAY_TASK_ID:?array task only}"
[[ "$IDX" =~ ^[0-9]+$ ]] || { echo "SLURM_ARRAY_TASK_ID '$IDX' is not an index" >&2; exit 2; }
PY_MODULES="python/3.10.13"
MANIFEST="$PRODUCTS/MANIFEST.tsv"

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "python3 $KIT/tools/t118/ladder/train.py cache $MANIFEST $PRODUCTS $CACHE $IDX"
  exit 0
fi

[ -f "$KIT/tools/t118/ladder/train.py" ] || { echo "no tools/t118/ladder/train.py under $KIT" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "no $MANIFEST" >&2; exit 2; }
KIT=$(cd "$KIT" && pwd)
mkdir -p "$CACHE"

PID=$(awk -F'\t' 'NR == 1 { for (c = 1; c <= NF; c++) if ($c == "pid") p = c; next } { print $p }' \
        "$MANIFEST" | LC_ALL=C sort | sed -n "$((IDX + 1))p")
[ -n "$PID" ] || { echo "no product at index $IDX in $MANIFEST" >&2; exit 2; }

echo "=== t118 ladder_cache $PID idx=$IDX job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX host=$(hostname) $(date -u)"
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || git -C "$KIT" rev-parse HEAD 2>/dev/null || echo unknown)"
if [ -f "$CACHE/${PID}__counts.json" ] && [ -f "$CACHE/${PID}__pval.json" ]; then
  echo "skip $PID: both cache json files exist"; exit 0
fi

set +u; module load $PY_MODULES; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
ls "$KIT"/wheels/x_transformers-*-py3-none-any.whl >/dev/null 2>&1 \
  || { echo "no x_transformers wheel in $KIT/wheels" >&2; exit 2; }
pip install --no-index -r "$KIT/requirements-fir.txt"
pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
export PYTHONPATH="$KIT/src"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
if [ -f "$KIT/GIT_SHA" ]; then export T118_GIT_SHA=$(cat "$KIT/GIT_SHA"); fi
python3 -c "import candi.metrics, candi.bench.distributional, candi.store.genome; print('venv ok', candi.__file__)"
# the library must come from this kit, not from some other install (tests/test_slurm_kit_pin.py)
case "$(python3 -c 'import candi; print(candi.__file__)')" in
  "$KIT"/src/*) ;;
  *) echo "candi does not import from $KIT/src" >&2; exit 3 ;;
esac

python3 "$KIT/tools/t118/ladder/train.py" cache "$MANIFEST" "$PRODUCTS" "$CACHE" "$IDX"

echo "=== done $PID $(date -u)"
ls -l "$CACHE/${PID}__counts.json" "$CACHE/${PID}__pval.json"
