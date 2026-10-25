#!/bin/bash
# t128 phase 2 reference: arm D via the base (tools/t128/viabase.py), from the NEW 12,000-step
# rung-D maps of phase 2. One array task = one phase-2 run D_<g>_counts_real_s<seed>, its two
# trained maps chained arm -> base -> arm on every never-trained arm->arm pair of its g, scored
# with the CRPS split. 24 tasks from `tools/t128/viabase.py tasks` (unchanged; index = g_i*3 + seed;
# index 9 = D_C19M16_counts_real_s0).
#
#   python3 $KIT/tools/t128/viabase.py run <manifest> <covariates.tsv> <cache_dir> <blacklist> \
#       <runs_dir>/<run_name> <out_dir>/viabase/<run_name> --workers 8
#                                              -> law.json, LAW_DONE, viabase.json
# The runs dir is READ ONLY. <out_dir>/viabase/<run_name>/LAW_DONE exists: exit 0. No
# <runs_dir>/<run_name>/SCORE_DONE: exit 3 (the phase-2 D run is not finished). An out dir not
# under a t128p2 directory is refused (exit 2).
#
# NO --export; positional arguments only. CPU only, 8 cores, 24 GB, 4 h; Fir nodes fc10713 and
# fc10715 excluded. DRY_RUN=1 prints the command line and exits 0.
#
# Usage (RUNS=/project/def-maxwl/mforooz/t128p2/runs, OUT=/project/def-maxwl/mforooz/t128p2):
#   smoke: sbatch --parsable --array=9 --output=$OUT/smoke/logs/%x_%A_%a.out \
#              $KIT/slurm/t128p2/viabase.sh $KIT $PRODUCTS $CACHE $BLACKLIST $OUT/smoke/runs $OUT/smoke
#   full:  sbatch --parsable --array=0-23%12 --output=$OUT/logs/%x_%A_%a.out \
#              $KIT/slurm/t128p2/viabase.sh $KIT $PRODUCTS $CACHE $BLACKLIST $RUNS $OUT
#SBATCH --account=def-maxwl
#SBATCH --job-name=t128p2_viabase
#SBATCH --exclude=fc10713,fc10715
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000M

set -euo pipefail

BLACKLIST_SHA256_PREFIX=31c69342
USAGE="usage: t128p2/viabase.sh <kit_dir> <products_dir> <cache_dir> <blacklist.bed> <runs_dir> <out_dir>"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
CACHE="${3:?$USAGE}"
BLACKLIST="${4:?$USAGE}"
RUNS="${5:?$USAGE}"
OUT="${6:?$USAGE}"
IDX="${SLURM_ARRAY_TASK_ID:?array task only}"
[[ "$IDX" =~ ^[0-9]+$ ]] || { echo "SLURM_ARRAY_TASK_ID '$IDX' is not an index" >&2; exit 2; }
case "$OUT" in
  */t118*|*/t125*|*/t126*|*/t128|*/t128/*) echo "out dir $OUT is under a t118/t125/t126/t128 directory; t128p2 writes only under t128p2/" >&2; exit 2 ;;
esac
case "$OUT" in
  */t128p2|*/t128p2/*) ;;
  *) echo "out dir $OUT is not under a t128p2 directory" >&2; exit 2 ;;
esac
PY_MODULES="python/3.10.13"
MANIFEST="$PRODUCTS/MANIFEST.tsv"
COVARIATES="$KIT/tools/t118/covariates.tsv"
VB="$OUT/viabase"

# The run_name of task $IDX, read from a viabase.py task table on stdin (columns found by header).
run_name_from() {
  awk -F'\t' -v i="$IDX" 'NR == 1 { for (c = 1; c <= NF; c++) h[$c] = c; next }
                          $h["index"] == i { print $h["run_name"] }'
}

if [ "${DRY_RUN:-0}" = 1 ]; then
  TOOLS="$KIT"
  [ -f "$TOOLS/tools/t128/viabase.py" ] || TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  TABLE=$(PYTHONPATH="$TOOLS/src${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-python3}" \
          "$TOOLS/tools/t128/viabase.py" tasks)
  RUN=$(run_name_from <<< "$TABLE")
  [ -n "$RUN" ] || { echo "no task $IDX in the viabase.py table" >&2; exit 2; }
  if [ -f "$VB/$RUN/LAW_DONE" ]; then echo "skip $RUN: LAW_DONE exists" >&2; exit 0; fi
  [ -f "$RUNS/$RUN/SCORE_DONE" ] || echo "note: no $RUNS/$RUN/SCORE_DONE; a real task would exit 3" >&2
  echo "python3 $KIT/tools/t128/viabase.py run $MANIFEST $COVARIATES $CACHE $BLACKLIST $RUNS/$RUN $VB/$RUN --workers 8"
  exit 0
fi

[ -f "$KIT/tools/t128/viabase.py" ] || { echo "no tools/t128/viabase.py under $KIT" >&2; exit 2; }
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
mkdir -p "$VB"

echo "=== t128p2 viabase idx=$IDX job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX host=$(hostname) $(date -u)"
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || git -C "$KIT" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "blacklist sha256: $BL_SHA"

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

RUN=$(python3 "$KIT/tools/t128/viabase.py" tasks | run_name_from)
[ -n "$RUN" ] || { echo "no task $IDX in the viabase.py table" >&2; exit 2; }
echo "run: $RUN"
if [ -f "$VB/$RUN/LAW_DONE" ]; then echo "skip $RUN: LAW_DONE exists"; exit 0; fi
[ -f "$RUNS/$RUN/SCORE_DONE" ] || { echo "no $RUNS/$RUN/SCORE_DONE: the phase-2 D run is not finished" >&2; exit 3; }

python3 "$KIT/tools/t128/viabase.py" run "$MANIFEST" "$COVARIATES" "$CACHE" "$BLACKLIST" \
    "$RUNS/$RUN" "$VB/$RUN" --workers 8

echo "=== done $RUN $(date -u)"
