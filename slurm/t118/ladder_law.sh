#!/bin/bash
# t118 ladder: the law test of one trained run on CPU — every never-trained arm->arm pair of its
# g, scored from the run's checkpoint. One array task = one run; 576 tasks, the same index as
# ladder_train.sh.
#
# WHAT ONE TASK DOES.
#   python3 $KIT/tools/t118/ladder/score.py law <manifest> <covariates.tsv> <cache_dir> \
#       <blacklist> <out_dir>/runs/<run_name> --workers 8        -> law.json, LAW_DONE
# Predictions are made in-process from the checkpoint and never written. A task whose LAW_DONE
# exists exits 0 without work. A task whose run has no SCORE_DONE exits 3, not 0, so a failed or
# unfinished train task shows up as a failed law task instead of a silent gap.
#
# THE TASK TABLE. Index -> run comes from `python3 tools/t118/ladder/pairs.py tasks <manifest>`
# (index 54 = A_C19M16_counts_real_s0). The copy in <out_dir>/tasks.tsv (written by the first
# task with a venv) is read before the venv only to exit early, then checked against a fresh
# pairs.py table (a mismatch exits 2).
#
# NO --export. Every input is a POSITIONAL argument; the task index is $SLURM_ARRAY_TASK_ID.
# Every output is under <out_dir> (the foreman passes /project/def-maxwl/mforooz/t118/ladder).
#
# THE VENV is the repo's pinned recipe, copied from slurm/t118/qm_rungs.sh (python/3.10.13,
# requirements-fir.txt, then requirements-pypi.txt from $KIT/wheels), built per task in
# $SLURM_TMPDIR.
#
# RESOURCES. CPU only, no GPU request. 8 cores (score.py --workers 8), 24 GB, 3 h. Planner's
# estimate: 10-25 min per per-track run (A -> D), 60-90 min per across-track run. Peak memory
# from `sacct -o MaxRSS` (no /usr/bin/time on compute nodes).
#
# DRY_RUN=1 prints the law command line (nothing when LAW_DONE exists; a note on stderr when
# SCORE_DONE is absent, where a real task would exit 3) and exits 0 before any module load. The
# task table is resolved as in ladder_train.sh: ${PYTHON:-python3}, $KIT or else this script's
# checkout, <products_dir>/MANIFEST.tsv or else tests/fixtures/t112_meta/MANIFEST.tsv.
#
# Usage, from the Nibi login node. Submit it right after the train array, while that array is
# live: aftercorr releases task i when train task i succeeds. Slurm refuses a dependency on a job
# that has already left the queue, so never chain onto a finished array; verify finished work by
# its files and submit without the dependency.
#   OUT=/project/def-maxwl/mforooz/t118/ladder; mkdir -p $OUT/logs
#   sbatch --test-only --array=0-575%40 --output=$OUT/logs/%x_%A_%a.out \
#       $KIT/slurm/t118/ladder_law.sh $KIT $PRODUCTS $OUT/cache $BLACKLIST $OUT
#   sbatch --parsable --array=54,57,60,63,66,69%6 --dependency=aftercorr:$PILOT_TRAIN \
#       --output=$OUT/logs/%x_%A_%a.out $KIT/slurm/t118/ladder_law.sh $KIT $PRODUCTS $OUT/cache $BLACKLIST $OUT
#   sbatch --parsable --array=0-575%40 --dependency=aftercorr:$TRAIN \
#       --output=$OUT/logs/%x_%A_%a.out $KIT/slurm/t118/ladder_law.sh $KIT $PRODUCTS $OUT/cache $BLACKLIST $OUT
#SBATCH --account=def-maxwl
#SBATCH --job-name=t118L_law
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000M

set -euo pipefail

BLACKLIST_SHA256_PREFIX=31c69342
USAGE="usage: ladder_law.sh <kit_dir> <products_dir> <cache_dir> <blacklist.bed> <out_dir>"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
CACHE="${3:?$USAGE}"
BLACKLIST="${4:?$USAGE}"
OUT="${5:?$USAGE}"
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
  if [ -f "$RUNS/$RUN/LAW_DONE" ]; then echo "skip $RUN: LAW_DONE exists" >&2; exit 0; fi
  [ -f "$RUNS/$RUN/SCORE_DONE" ] || echo "note: no $RUNS/$RUN/SCORE_DONE; a real task would exit 3" >&2
  echo "python3 $KIT/tools/t118/ladder/score.py law $MANIFEST $COVARIATES $CACHE $BLACKLIST $RUNS/$RUN --workers 8"
  exit 0
fi

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

echo "=== t118 ladder_law idx=$IDX job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX host=$(hostname) $(date -u)"
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || git -C "$KIT" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "blacklist sha256: $BL_SHA"

# Early exits before the venv, only from a table an earlier task already wrote with pairs.py.
if [ -f "$TASKS_TSV" ]; then
  RUN=$(run_name_from < "$TASKS_TSV")
  if [ -n "$RUN" ]; then
    if [ -f "$RUNS/$RUN/LAW_DONE" ]; then echo "skip $RUN: LAW_DONE exists"; exit 0; fi
    [ -f "$RUNS/$RUN/SCORE_DONE" ] || { echo "no $RUNS/$RUN/SCORE_DONE: train task $IDX did not finish" >&2; exit 3; }
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
if [ -f "$TASKS_TSV" ] && ! cmp -s "$TMP_TABLE" "$TASKS_TSV"; then
  echo "$TASKS_TSV differs from this kit's pairs.py table; remove it or fix the kit" >&2
  rm -f "$TMP_TABLE"; exit 2
fi
mv -f "$TMP_TABLE" "$TASKS_TSV"
RUN=$(run_name_from < "$TASKS_TSV")
[ -n "$RUN" ] || { echo "no task $IDX in $TASKS_TSV" >&2; exit 2; }
echo "run: $RUN"
if [ -f "$RUNS/$RUN/LAW_DONE" ]; then echo "skip $RUN: LAW_DONE exists"; exit 0; fi
[ -f "$RUNS/$RUN/SCORE_DONE" ] || { echo "no $RUNS/$RUN/SCORE_DONE: train task $IDX did not finish" >&2; exit 3; }

python3 "$KIT/tools/t118/ladder/score.py" law "$MANIFEST" "$COVARIATES" "$CACHE" "$BLACKLIST" \
    "$RUNS/$RUN" --workers 8

echo "=== done $RUN $(date -u)"
