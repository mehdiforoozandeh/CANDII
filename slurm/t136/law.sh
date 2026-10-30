#!/bin/bash
# t136 (h28): the law test of one trained arm-to-arm grid run on CPU, on Nibi. It scores every
# arm->arm pair of its g (trained extra pairs included; the readings subtract config.json's
# `extra_pairs`) from the run's checkpoint, with the oracle_scaled / scale_error CRPS split. One
# array task = one run; 42 tasks, the same index as slurm/t136/train.sh. This is
# slurm/t128p3/law_3b.sh with `tools/t136/pairs.py tasks`, the out dir t136 and NO via-base step
# (the reference via-base runs exist under t128p2/viabase and are not rerun).
#
#   1. python3 $KIT/tools/t118/ladder/score.py law <manifest> <covariates.tsv> <cache_dir> \
#          <blacklist> <out_dir>/runs/<run_name> --workers 8 --split
#                                                  -> <out_dir>/runs/<run_name>/law.json, LAW_DONE
# Skipped when runs/<run>/LAW_DONE exists (exit 0). No SCORE_DONE: exit 3 (a failed train task
# shows as a failed law task).
#
# NO --export; positional arguments only. CPU only, 8 cores, 24 GB, 6 h, on Nibi (the train array's
# cluster: aftercorr cannot chain across clusters); nodes c128, c166, c537, c371, c329 excluded. The out dir must be t136 or under it. DRY_RUN=1 prints the command line that a
# real task would run and exits 0.
#
# Usage: submit right after the train array, WHILE IT IS LIVE (aftercorr releases task i when
# train task i succeeds; never chain onto a finished array), on the same array as train.sh:
#          sbatch --parsable --array=0-20%12 --dependency=aftercorr:$TRAIN \
#              --output=$OUT/logs/%x_%A_%a.out \
#              $KIT/slurm/t136/law.sh $KIT $PRODUCTS $CACHE $BLACKLIST $OUT      # Darm1
#          sbatch --parsable --array=21-41%12 --dependency=aftercorr:$TRAIN \
#              --output=$OUT/logs/%x_%A_%a.out \
#              $KIT/slurm/t136/law.sh $KIT $PRODUCTS $CACHE $BLACKLIST $OUT      # Darm2
#SBATCH --account=def-maxwl
#SBATCH --job-name=t136_law
#SBATCH --exclude=c128,c166,c537,c371,c329
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000M

set -euo pipefail

BLACKLIST_SHA256_PREFIX=31c69342
USAGE="usage: t136/law.sh <kit_dir> <products_dir> <cache_dir> <blacklist.bed> <out_dir>"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
CACHE="${3:?$USAGE}"
BLACKLIST="${4:?$USAGE}"
OUT="${5:?$USAGE}"
IDX="${SLURM_ARRAY_TASK_ID:?array task only}"
[[ "$IDX" =~ ^[0-9]+$ ]] || { echo "SLURM_ARRAY_TASK_ID '$IDX' is not an index" >&2; exit 2; }
case "$OUT" in
  */t136|*/t136/*) ;;
  *) echo "out dir $OUT is not t136 or under it; t136 writes only there" >&2; exit 2 ;;
esac
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

law_line() {
  echo "python3 $KIT/tools/t118/ladder/score.py law $MANIFEST $COVARIATES $CACHE $BLACKLIST $RUNS/$1 --workers 8 --split"
}

if [ "${DRY_RUN:-0}" = 1 ]; then
  TOOLS="$KIT"
  [ -f "$TOOLS/tools/t136/pairs.py" ] || TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  TABLE_MANIFEST="$MANIFEST"
  [ -f "$TABLE_MANIFEST" ] || TABLE_MANIFEST="$TOOLS/tests/fixtures/t112_meta/MANIFEST.tsv"
  TABLE=$(PYTHONPATH="$TOOLS/src${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-python3}" \
          "$TOOLS/tools/t136/pairs.py" tasks "$TABLE_MANIFEST")
  RUN=$(run_name_from <<< "$TABLE")
  [ -n "$RUN" ] || { echo "no task $IDX in the pairs.py table" >&2; exit 2; }
  if [ -f "$RUNS/$RUN/LAW_DONE" ]; then echo "skip $RUN: LAW_DONE exists" >&2; exit 0; fi
  [ -f "$RUNS/$RUN/SCORE_DONE" ] || echo "note: no $RUNS/$RUN/SCORE_DONE; a real task would exit 3" >&2
  law_line "$RUN"
  exit 0
fi

[ -f "$KIT/tools/t118/ladder/score.py" ] || { echo "no tools/t118/ladder/score.py under $KIT" >&2; exit 2; }
[ -f "$KIT/tools/t136/pairs.py" ] || { echo "no tools/t136/pairs.py under $KIT" >&2; exit 2; }
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

echo "=== t136 law idx=$IDX job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX host=$(hostname) $(date -u)"
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
python3 "$KIT/tools/t136/pairs.py" tasks "$MANIFEST" > "$TMP_TABLE"
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
if [ -f "$RUNS/$RUN/LAW_DONE" ]; then echo "skip $RUN: LAW_DONE exists"; exit 0; fi
[ -f "$RUNS/$RUN/SCORE_DONE" ] || { echo "no $RUNS/$RUN/SCORE_DONE: train task $IDX did not finish" >&2; exit 3; }

python3 "$KIT/tools/t118/ladder/score.py" law "$MANIFEST" "$COVARIATES" "$CACHE" "$BLACKLIST" \
    "$RUNS/$RUN" --workers 8 --split
echo "=== law done $(date -u)"

echo "=== done $RUN $(date -u)"
