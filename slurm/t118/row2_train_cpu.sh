#!/bin/bash
# t118 ladder, ROW 2, CPU copy of row2_train.sh: for when the 10 GB slice nodes are held (as on
# 2026-09-25, when explore_x_cpu.sh stood in for explore_x.sh). Identical to row2_train.sh except:
# no GPU request, train.py runs with --device cpu, 16 cores, 64 GB, 8 h, and OMP_NUM_THREADS /
# MKL_NUM_THREADS are set to the core count. Same task table (`pairs.py tasks <manifest> --row 2`,
# index 54 = A2_C19M16_counts_real_s0), same $OUT/tasks.tsv in the row-2 out dir, same outputs, so
# a run trained here and one trained by row2_train.sh are interchangeable to score.py and to
# row2_law.sh; a task whose SCORE_DONE exists exits 0 without work.
#
# WHAT ONE TASK DOES.
#   1. python3 $KIT/tools/t118/ladder/train.py train-index <manifest> <covariates.tsv> <cache_dir> \
#          <out_dir>/runs <index> --row 2 --device cpu
#   2. python3 $KIT/tools/t118/ladder/score.py trained <manifest> <covariates.tsv> <cache_dir> \
#          <blacklist> <out_dir>/runs/<run_name> --workers 4
#
# NO --export: every input is a POSITIONAL argument; the task index is $SLURM_ARRAY_TASK_ID.
#   OUT=/project/def-maxwl/mforooz/t118/row2
#   CACHE=/project/def-maxwl/mforooz/t118/ladder/cache   (row 1's cache, read-only here)
# THE VENV is the pinned recipe of row2_train.sh (python/3.10.13, requirements-fir.txt, then
# requirements-pypi.txt from $KIT/wheels), built per task in $SLURM_TMPDIR. Nodes c128, c166 and
# c537 are excluded. Peak memory from `sacct -o MaxRSS` (no /usr/bin/time on compute nodes).
# DRY_RUN=1 behaves as in row2_train.sh.
#
# Usage, from the Nibi login node:
#   OUT=/project/def-maxwl/mforooz/t118/row2; mkdir -p $OUT/logs
#   CACHE=/project/def-maxwl/mforooz/t118/ladder/cache
#   sbatch --test-only --array=0-575%40 --exclude=c128,c166,c537 --output=$OUT/logs/%x_%A_%a.out \
#       $KIT/slurm/t118/row2_train_cpu.sh $KIT $PRODUCTS $CACHE $OUT
#   sbatch --parsable --array=0-575%40 --exclude=c128,c166,c537 --output=$OUT/logs/%x_%A_%a.out \
#       $KIT/slurm/t118/row2_train_cpu.sh $KIT $PRODUCTS $CACHE $OUT
# Then submit row2_law.sh DIRECTLY (no dependency) for the tasks whose SCORE_DONE exists.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t118L_r2traincpu
#SBATCH --exclude=c128,c166,c537
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M

set -euo pipefail

DEFAULT_BLACKLIST=/project/def-maxwl/mforooz/EIC_REPRO/002/scripts/hg38_blacklist_v2.bed
BLACKLIST_SHA256_PREFIX=31c69342
USAGE="usage: row2_train_cpu.sh <kit_dir> <products_dir> <cache_dir> <out_dir> [blacklist.bed]"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
CACHE="${3:?$USAGE}"
OUT="${4:?$USAGE}"
BLACKLIST="${5:-$DEFAULT_BLACKLIST}"
IDX="${SLURM_ARRAY_TASK_ID:?array task only}"
[[ "$IDX" =~ ^[0-9]+$ ]] || { echo "SLURM_ARRAY_TASK_ID '$IDX' is not an index" >&2; exit 2; }
PY_MODULES="python/3.10.13"
MANIFEST="$PRODUCTS/MANIFEST.tsv"
COVARIATES="$KIT/tools/t118/covariates.tsv"
RUNS="$OUT/runs"
TASKS_TSV="$OUT/tasks.tsv"

# The run_name of task $IDX, read from a pairs.py task table on stdin (columns found by header).
run_name_from() {
  awk -F'\t' -v i="$IDX" 'NR == 1 { for (c = 1; c <= NF; c++) h[$c] = c; next }
                          $h["index"] == i { print $h["run_name"] }'
}

if [ "${DRY_RUN:-0}" = 1 ]; then
  TOOLS="$KIT"
  [ -f "$TOOLS/tools/t118/ladder/pairs.py" ] || TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  TABLE_MANIFEST="$MANIFEST"
  [ -f "$TABLE_MANIFEST" ] || TABLE_MANIFEST="$TOOLS/tests/fixtures/t112_meta/MANIFEST.tsv"
  TABLE=$(PYTHONPATH="$TOOLS/src${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-python3}" \
          "$TOOLS/tools/t118/ladder/pairs.py" tasks "$TABLE_MANIFEST" --row 2)
  RUN=$(run_name_from <<< "$TABLE")
  [ -n "$RUN" ] || { echo "no task $IDX in the pairs.py table" >&2; exit 2; }
  if [ -f "$RUNS/$RUN/SCORE_DONE" ]; then echo "skip $RUN: SCORE_DONE exists" >&2; exit 0; fi
  echo "python3 $KIT/tools/t118/ladder/train.py train-index $MANIFEST $COVARIATES $CACHE $RUNS $IDX --row 2 --device cpu"
  echo "python3 $KIT/tools/t118/ladder/score.py trained $MANIFEST $COVARIATES $CACHE $BLACKLIST $RUNS/$RUN --workers 4"
  exit 0
fi

[ -f "$KIT/tools/t118/ladder/train.py" ] || { echo "no tools/t118/ladder/train.py under $KIT" >&2; exit 2; }
[ -f "$KIT/tools/t118/ladder/score.py" ] || { echo "no tools/t118/ladder/score.py under $KIT" >&2; exit 2; }
[ -f "$COVARIATES" ] || { echo "no $COVARIATES" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "no $MANIFEST" >&2; exit 2; }
[ -d "$CACHE" ] || { echo "no cache dir $CACHE (run ladder_cache.sh first)" >&2; exit 2; }
[ -s "$BLACKLIST" ] || { echo "blacklist $BLACKLIST missing or empty" >&2; exit 2; }
BL_SHA=$(sha256sum "$BLACKLIST" | cut -d' ' -f1)
case "$BL_SHA" in
  "$BLACKLIST_SHA256_PREFIX"*) ;;
  *) echo "blacklist sha256 $BL_SHA does not start $BLACKLIST_SHA256_PREFIX" >&2; exit 2 ;;
esac
KIT=$(cd "$KIT" && pwd)
COVARIATES="$KIT/tools/t118/covariates.tsv"
mkdir -p "$RUNS"

echo "=== t118 row2_train_cpu idx=$IDX job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX host=$(hostname) $(date -u)"
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || git -C "$KIT" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "blacklist sha256: $BL_SHA"

# Early exit before the venv, only from a table an earlier task already wrote with pairs.py.
if [ -f "$TASKS_TSV" ]; then
  RUN=$(run_name_from < "$TASKS_TSV")
  if [ -n "$RUN" ] && [ -f "$RUNS/$RUN/SCORE_DONE" ]; then
    echo "skip $RUN: SCORE_DONE exists"; exit 0
  fi
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
if [ -f "$KIT/GIT_SHA" ]; then export T118_GIT_SHA=$(cat "$KIT/GIT_SHA"); fi
python3 -c "import candi.metrics, candi.bench.distributional, candi.store.genome; print('venv ok', candi.__file__)"
# the library must come from this kit, not from some other install (tests/test_slurm_kit_pin.py)
case "$(python3 -c 'import candi; print(candi.__file__)')" in
  "$KIT"/src/*) ;;
  *) echo "candi does not import from $KIT/src" >&2; exit 3 ;;
esac

# The authoritative task table, from pairs.py; a stale shared copy is an error, not a guess.
TMP_TABLE="$TASKS_TSV.tmp.${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX"
python3 "$KIT/tools/t118/ladder/pairs.py" tasks "$MANIFEST" --row 2 > "$TMP_TABLE"
# The run comes from this task's own table. The shared copy is created once and never replaced:
# replacing it under 40 concurrent readers on /project gave "Stale file handle" (8 failed tasks,
# job 22657297). cmp exit 1 = the tables differ (an error); exit 2 = a read problem (not fatal).
RUN=$(run_name_from < "$TMP_TABLE")
if [ -f "$TASKS_TSV" ]; then
  CMP_RC=0; cmp -s "$TMP_TABLE" "$TASKS_TSV" || CMP_RC=$?
  if [ "$CMP_RC" = 1 ]; then
    echo "$TASKS_TSV differs from this kit's pairs.py table; remove it or fix the kit" >&2
    rm -f "$TMP_TABLE"; exit 2
  fi
else
  mv -n "$TMP_TABLE" "$TASKS_TSV" 2>/dev/null || true
fi
rm -f "$TMP_TABLE"
[ -n "$RUN" ] || { echo "no task $IDX in the pairs.py table" >&2; exit 2; }
echo "run: $RUN"
if [ -f "$RUNS/$RUN/SCORE_DONE" ]; then echo "skip $RUN: SCORE_DONE exists"; exit 0; fi

python3 "$KIT/tools/t118/ladder/train.py" train-index "$MANIFEST" "$COVARIATES" "$CACHE" "$RUNS" "$IDX" --row 2 --device cpu
echo "=== trained $(date -u)"
python3 "$KIT/tools/t118/ladder/score.py" trained "$MANIFEST" "$COVARIATES" "$CACHE" "$BLACKLIST" \
    "$RUNS/$RUN" --workers 4

echo "=== done $RUN $(date -u)"
