#!/bin/bash
# t125 part 2: score one h ALONE — on the true X' of every usable product (128), chr19 + chr21,
# blacklist bins removed. One array task = one h of `tasks.py alone`: 38 tasks, the 2 formula h
# (nothing learned) then the 36 h trained on the true X' (h_train.sh table "true").
#
# WHAT ONE TASK DOES.
#   python3 $KIT/tools/t125/hmap/score.py alone <manifest> <covariates.tsv> <cache_dir> <ctl_cache> \
#       <blacklist> <h_spec> <out_root>/alone --workers 8   -> alone/<h_name>/scores.json, ALONE_DONE
# h_spec is the row's source for a formula ("formula:bin", "formula:frag") and the run dir
# <out_root>/runs_true/<h_name> for a learned h. A task whose ALONE_DONE exists exits 0 without
# work. A learned h whose run dir has no TRAIN_DONE exits 3, so an unfinished train task shows up
# as a failed scoring task instead of a silent gap.
#
# THE TASK TABLE. Index -> h comes from `python3 tools/t125/hmap/tasks.py alone <manifest>`
# (index 0 = formula:bin, 1 = formula:frag, 2 = h_affine_r1_push_true_s0). The first task with a
# venv writes it to <out_root>/tasks_alone.tsv; a later task reads that copy BEFORE its venv only
# to exit early, then checks it against a fresh tasks.py table (a mismatch exits 2). The copy is
# created once and never replaced. An <out_root> under a t118 directory is refused (exit 2).
#
# NO --export. Every input is a POSITIONAL argument; the task index is $SLURM_ARRAY_TASK_ID.
# Every output is under <out_root>/alone:
#   OUT_H=/project/def-maxwl/mforooz/t125/h
#   CTL=/project/def-maxwl/mforooz/t125/ctl_cache     (h_ctl_cache.sh builds it)
#   CACHE=/project/def-maxwl/mforooz/t118/ladder/cache (t118's count and pval cache, read-only here)
#
# THE VENV is the pinned recipe of slurm/t118/row2_train.sh (python/3.10.13, requirements-fir.txt,
# then requirements-pypi.txt from $KIT/wheels), built per task in $SLURM_TMPDIR; PYTHONPATH is
# "$KIT/src" and the t125 CLIs put tools/t118 and tools/t125 on sys.path themselves.
#
# RESOURCES. CPU only, no GPU request. 8 cores (score.py --workers 8), 32 GB, 2 h. Nodes c128,
# c166, c537 and c371 are excluded. Peak memory from `sacct -o MaxRSS` (no /usr/bin/time on
# compute nodes).
#
# DRY_RUN=1 prints the scoring command line (nothing when ALONE_DONE exists; a note on stderr when
# a learned h has no TRAIN_DONE, where a real task would exit 3) and exits 0 before any module
# load. The task table is resolved as in h_train.sh: ${PYTHON:-python3}, $KIT or else this
# script's checkout, <products_dir>/MANIFEST.tsv or else tests/fixtures/t112_meta/MANIFEST.tsv.
#
# Usage, from the Nibi login node, once the 36 true-X' h have their TRAIN_DONE (check the files;
# never a dependency on an array that may have finished). Every job id goes into
# plan/T125_STATUS.md:
#   OUT_H=/project/def-maxwl/mforooz/t125/h; mkdir -p $OUT_H/logs
#   CTL=/project/def-maxwl/mforooz/t125/ctl_cache; CACHE=/project/def-maxwl/mforooz/t118/ladder/cache
#   sbatch --test-only --array=0-37%38 --exclude=c128,c166,c537,c371 --output=$OUT_H/logs/%x_%A_%a.out \
#       $KIT/slurm/t125/h_alone.sh $KIT $PRODUCTS $CACHE $CTL $BLACKLIST $OUT_H
#   sbatch --parsable --array=0-37%38 --exclude=c128,c166,c537,c371 --output=$OUT_H/logs/%x_%A_%a.out \
#       $KIT/slurm/t125/h_alone.sh $KIT $PRODUCTS $CACHE $CTL $BLACKLIST $OUT_H
# The two formulas (indices 0, 1) need no trained h and may go first.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t125_halone
#SBATCH --exclude=c128,c166,c537,c371
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M

set -euo pipefail

BLACKLIST_SHA256_PREFIX=31c69342
USAGE="usage: h_alone.sh <kit_dir> <products_dir> <cache_dir> <ctl_cache> <blacklist.bed> <out_root>"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
CACHE="${3:?$USAGE}"
CTL="${4:?$USAGE}"
BLACKLIST="${5:?$USAGE}"
OUT_ROOT="${6:?$USAGE}"
IDX="${SLURM_ARRAY_TASK_ID:?array task only}"
[[ "$IDX" =~ ^[0-9]+$ ]] || { echo "SLURM_ARRAY_TASK_ID '$IDX' is not an index" >&2; exit 2; }
case "$OUT_ROOT" in
  */t118|*/t118/*) echo "out root $OUT_ROOT is under a t118 directory; t125 writes only under t125/" >&2; exit 2 ;;
esac
PY_MODULES="python/3.10.13"
MANIFEST="$PRODUCTS/MANIFEST.tsv"
COVARIATES="$KIT/tools/t118/covariates.tsv"
RUNS_TRUE="$OUT_ROOT/runs_true"
ALONE="$OUT_ROOT/alone"
TASKS_TSV="$OUT_ROOT/tasks_alone.tsv"

# "<h_name>\t<source>" of task $IDX, read from a tasks.py table on stdin (columns found by header).
row_from() {
  awk -F'\t' -v i="$IDX" 'NR == 1 { for (c = 1; c <= NF; c++) h[$c] = c; next }
                          $h["index"] == i { print $h["h_name"] "\t" $h["source"] }'
}
# The h_spec of a row: the formula source itself, or the true-X' run dir of a learned h.
spec_of() { case "$2" in formula:*) echo "$2" ;; *) echo "$RUNS_TRUE/$1" ;; esac; }

if [ "${DRY_RUN:-0}" = 1 ]; then
  TOOLS="$KIT"
  [ -f "$TOOLS/tools/t125/hmap/tasks.py" ] || TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  TABLE_MANIFEST="$MANIFEST"
  [ -f "$TABLE_MANIFEST" ] || TABLE_MANIFEST="$TOOLS/tests/fixtures/t112_meta/MANIFEST.tsv"
  ROWS=$(PYTHONPATH="$TOOLS/src${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-python3}" \
         "$TOOLS/tools/t125/hmap/tasks.py" alone "$TABLE_MANIFEST")
  ROW=$(row_from <<< "$ROWS")
  [ -n "$ROW" ] || { echo "no task $IDX in the tasks.py alone table" >&2; exit 2; }
  H=${ROW%%$'\t'*}; SPEC=$(spec_of "$H" "${ROW#*$'\t'}")
  if [ -f "$ALONE/$H/ALONE_DONE" ]; then echo "skip $H: ALONE_DONE exists" >&2; exit 0; fi
  case "$SPEC" in
    formula:*) ;;
    *) [ -f "$SPEC/TRAIN_DONE" ] || echo "note: no $SPEC/TRAIN_DONE; a real task would exit 3" >&2 ;;
  esac
  echo "python3 $KIT/tools/t125/hmap/score.py alone $MANIFEST $COVARIATES $CACHE $CTL $BLACKLIST $SPEC $ALONE --workers 8"
  exit 0
fi

[ -f "$KIT/tools/t125/hmap/score.py" ] || { echo "no tools/t125/hmap/score.py under $KIT" >&2; exit 2; }
[ -f "$KIT/tools/t125/hmap/tasks.py" ] || { echo "no tools/t125/hmap/tasks.py under $KIT" >&2; exit 2; }
[ -f "$COVARIATES" ] || { echo "no $COVARIATES" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "no $MANIFEST" >&2; exit 2; }
[ -d "$CACHE" ] || { echo "no cache dir $CACHE (t118 ladder_cache.sh builds it)" >&2; exit 2; }
[ -d "$CTL" ] || { echo "no control cache $CTL (h_ctl_cache.sh builds it)" >&2; exit 2; }
[ -s "$BLACKLIST" ] || { echo "blacklist $BLACKLIST missing or empty" >&2; exit 2; }
BL_SHA=$(sha256sum "$BLACKLIST" | cut -d' ' -f1)
case "$BL_SHA" in
  "$BLACKLIST_SHA256_PREFIX"*) ;;
  *) echo "blacklist sha256 $BL_SHA does not start $BLACKLIST_SHA256_PREFIX" >&2; exit 2 ;;
esac
KIT=$(cd "$KIT" && pwd)
COVARIATES="$KIT/tools/t118/covariates.tsv"
mkdir -p "$ALONE"

echo "=== t125 h_alone idx=$IDX job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX host=$(hostname) $(date -u)"
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || git -C "$KIT" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "blacklist sha256: $BL_SHA"

# Early exits before the venv, only from a table an earlier task already wrote with tasks.py.
if [ -f "$TASKS_TSV" ]; then
  ROW=$(row_from < "$TASKS_TSV")
  if [ -n "$ROW" ]; then
    H=${ROW%%$'\t'*}; SPEC=$(spec_of "$H" "${ROW#*$'\t'}")
    if [ -f "$ALONE/$H/ALONE_DONE" ]; then echo "skip $H: ALONE_DONE exists"; exit 0; fi
    case "$SPEC" in
      formula:*) ;;
      *) [ -f "$SPEC/TRAIN_DONE" ] || { echo "no $SPEC/TRAIN_DONE: h $H is not trained" >&2; exit 3; } ;;
    esac
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
if [ -f "$KIT/GIT_SHA" ]; then export T125_GIT_SHA=$(cat "$KIT/GIT_SHA"); fi
python3 -c "import candi.metrics, candi.bench.distributional, candi.store.genome; print('venv ok', candi.__file__)"
# the library must come from this kit, not from some other install (tests/test_slurm_kit_pin.py)
case "$(python3 -c 'import candi; print(candi.__file__)')" in
  "$KIT"/src/*) ;;
  *) echo "candi does not import from $KIT/src" >&2; exit 3 ;;
esac

# The authoritative task table, from tasks.py; a stale shared copy is an error, not a guess.
TMP_TABLE="$TASKS_TSV.tmp.${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX"
python3 "$KIT/tools/t125/hmap/tasks.py" alone "$MANIFEST" > "$TMP_TABLE"
# cmp exit 1 = the tables differ (an error); exit 2 = a read problem (not fatal).
ROW=$(row_from < "$TMP_TABLE")
if [ -f "$TASKS_TSV" ]; then
  CMP_RC=0; cmp -s "$TMP_TABLE" "$TASKS_TSV" || CMP_RC=$?
  if [ "$CMP_RC" = 1 ]; then
    echo "$TASKS_TSV differs from this kit's tasks.py table; remove it or fix the kit" >&2
    rm -f "$TMP_TABLE"; exit 2
  fi
else
  mv -n "$TMP_TABLE" "$TASKS_TSV" 2>/dev/null || true
fi
rm -f "$TMP_TABLE"
[ -n "$ROW" ] || { echo "no task $IDX in the tasks.py alone table" >&2; exit 2; }
H=${ROW%%$'\t'*}; SPEC=$(spec_of "$H" "${ROW#*$'\t'}")
echo "h: $H spec: $SPEC"
if [ -f "$ALONE/$H/ALONE_DONE" ]; then echo "skip $H: ALONE_DONE exists"; exit 0; fi
case "$SPEC" in
  formula:*) ;;
  *) [ -f "$SPEC/TRAIN_DONE" ] || { echo "no $SPEC/TRAIN_DONE: h $H is not trained" >&2; exit 3; } ;;
esac

python3 "$KIT/tools/t125/hmap/score.py" alone "$MANIFEST" "$COVARIATES" "$CACHE" "$CTL" \
    "$BLACKLIST" "$SPEC" "$ALONE" --workers 8

echo "=== done $H $(date -u)"
