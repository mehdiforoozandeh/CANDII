#!/bin/bash
# t118 ladder: train one (rung, g, space, model, seed) run on a 10 GB MIG slice, then score its
# trained pairs, shuffle, swap and depth law. One array task = one run; 576 tasks.
#
# WHAT ONE TASK DOES.
#   1. python3 $KIT/tools/t118/ladder/train.py train-index <manifest> <covariates.tsv> <cache_dir> \
#          <out_dir>/runs <index>                        -> <out_dir>/runs/<run_name>/ckpt.pt, TRAIN_DONE
#   2. python3 $KIT/tools/t118/ladder/score.py trained <manifest> <covariates.tsv> <cache_dir> \
#          <blacklist> <out_dir>/runs/<run_name> --workers 4   -> scores.json, figdata.npz, SCORE_DONE
# A task whose SCORE_DONE exists exits 0 without work, so a resubmission redoes only the missing
# runs (train.py also skips on TRAIN_DONE, so a task that died while scoring does not retrain).
#
# THE TASK TABLE. Index -> run comes from `python3 tools/t118/ladder/pairs.py tasks <manifest>`
# (576 rows, rung slowest, seed fastest; index 54 = A_C19M16_counts_real_s0), a pure function of
# the manifest. The first task that has a venv writes it to <out_dir>/tasks.tsv; a later task reads
# that copy BEFORE building its venv only to exit early when its run is already done, and then
# checks it against a fresh pairs.py table (a mismatch exits 2). Order A -> B -> C -> D, so with
# %40 rungs A and B finish first.
#
# NO --export. Alliance arrays truncate a comma-valued exported variable, so every input is a
# POSITIONAL argument and the task index is $SLURM_ARRAY_TASK_ID. Every output is under <out_dir>
# (the foreman passes /project/def-maxwl/mforooz/t118/ladder).
#
# THE VENV is the repo's pinned recipe, copied from slurm/t118/qm_rungs.sh: python/3.10.13,
# requirements-fir.txt from the CVMFS wheelhouse, then requirements-pypi.txt (x-transformers only)
# from $KIT/wheels with --no-index --find-links. Built per task in $SLURM_TMPDIR.
#
# RESOURCES. One 10 GB MIG slice of an H100, 4 cores (score.py --workers 4), 16 GB host memory,
# 3 h. Planner's estimate: ~20 min per per-track run, ~35 min per across-track run; the pilot
# measures it. No /usr/bin/time on compute nodes: read peak memory from `sacct -o MaxRSS`.
#
# DRY_RUN=1 prints the two python command lines (or nothing, when the run is already done) and
# exits 0 before any module load. It resolves the task table with ${PYTHON:-python3} (needs numpy,
# scipy, torch: run it with the candii env active or PYTHON set) from $KIT when $KIT holds
# pairs.py, else from the checkout this script sits in; and from <products_dir>/MANIFEST.tsv when
# it exists, else from that checkout's tests/fixtures/t112_meta/MANIFEST.tsv (the same file,
# md5-pinned). Nothing is checked for existence except the run's SCORE_DONE.
#
# Usage, from the Nibi login node, after snapshotting the repo to $KIT:
#   OUT=/project/def-maxwl/mforooz/t118/ladder; mkdir -p $OUT/logs   # SLURM opens --output first
#   sbatch --test-only --array=0-575%40 --output=$OUT/logs/%x_%A_%a.out \
#       $KIT/slurm/t118/ladder_train.sh $KIT $PRODUCTS $OUT/cache $OUT
#   # pilot (rung A, g C19M16, both spaces, three models, seed 0):
#   sbatch --parsable --array=54,57,60,63,66,69%6 --output=$OUT/logs/%x_%A_%a.out \
#       $KIT/slurm/t118/ladder_train.sh $KIT $PRODUCTS $OUT/cache $OUT
#   # full run; submit the law array IN THE SAME BREATH, while this one is live (ladder_law.sh):
#   TRAIN=$(sbatch --parsable --array=0-575%40 --output=$OUT/logs/%x_%A_%a.out \
#       $KIT/slurm/t118/ladder_train.sh $KIT $PRODUCTS $OUT/cache $OUT | cut -d';' -f1)
#   sbatch --parsable --array=0-575%40 --dependency=aftercorr:$TRAIN --output=$OUT/logs/%x_%A_%a.out \
#       $KIT/slurm/t118/ladder_law.sh $KIT $PRODUCTS $OUT/cache $BLACKLIST $OUT
#   # never chain afterok onto a job that may have finished: verify finished work by its files.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t118L_train
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M

set -euo pipefail

DEFAULT_BLACKLIST=/project/def-maxwl/mforooz/EIC_REPRO/002/scripts/hg38_blacklist_v2.bed
BLACKLIST_SHA256_PREFIX=31c69342
USAGE="usage: ladder_train.sh <kit_dir> <products_dir> <cache_dir> <out_dir> [blacklist.bed]"
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
          "$TOOLS/tools/t118/ladder/pairs.py" tasks "$TABLE_MANIFEST")
  RUN=$(run_name_from <<< "$TABLE")
  [ -n "$RUN" ] || { echo "no task $IDX in the pairs.py table" >&2; exit 2; }
  if [ -f "$RUNS/$RUN/SCORE_DONE" ]; then echo "skip $RUN: SCORE_DONE exists" >&2; exit 0; fi
  echo "python3 $KIT/tools/t118/ladder/train.py train-index $MANIFEST $COVARIATES $CACHE $RUNS $IDX"
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

echo "=== t118 ladder_train idx=$IDX job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX host=$(hostname) $(date -u)"
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
if [ -f "$KIT/GIT_SHA" ]; then export T118_GIT_SHA=$(cat "$KIT/GIT_SHA"); fi
python3 -c "import candi.metrics, candi.bench.distributional, candi.store.genome; print('venv ok', candi.__file__)"
# the library must come from this kit, not from some other install (tests/test_slurm_kit_pin.py)
case "$(python3 -c 'import candi; print(candi.__file__)')" in
  "$KIT"/src/*) ;;
  *) echo "candi does not import from $KIT/src" >&2; exit 3 ;;
esac

# The authoritative task table, from pairs.py; a stale shared copy is an error, not a guess.
TMP_TABLE="$TASKS_TSV.tmp.${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX"
python3 "$KIT/tools/t118/ladder/pairs.py" tasks "$MANIFEST" > "$TMP_TABLE"
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

python3 "$KIT/tools/t118/ladder/train.py" train-index "$MANIFEST" "$COVARIATES" "$CACHE" "$RUNS" "$IDX"
echo "=== trained $(date -u)"
python3 "$KIT/tools/t118/ladder/score.py" trained "$MANIFEST" "$COVARIATES" "$CACHE" "$BLACKLIST" \
    "$RUNS/$RUN" --workers 4

echo "=== done $RUN $(date -u)"
