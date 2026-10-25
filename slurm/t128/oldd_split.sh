#!/bin/bash
# t128 reference: old D's never-trained (law) pairs re-scored WITH the CRPS split
# (tools/t128/oldd_split.py). One array task = one old t126 rung-D run, D_<g>_counts_<model>_s<seed>
# (model real or its no-covariates twin nocov), the same direct law scoring as t126 plus the
# oracle_scaled / scale_error split. 48 tasks from `oldd_split.py tasks`
# (index = m_i*24 + g_i*3 + seed; index 9 = D_C19M16_counts_real_s0, 33 = its nocov twin).
#
#   python3 $KIT/tools/t128/oldd_split.py run <manifest> <covariates.tsv> <cache_dir> <blacklist> \
#       <old_runs_dir>/<run_name> <out_dir>/oldD_split/<run_name> --workers 8
#                                              -> law.json, LAW_DONE, oldd_split.json
# The old runs dir is READ ONLY. LAW_DONE exists: exit 0. No <old_runs_dir>/<run_name>/ckpt.pt:
# exit 3. An out dir not containing /t128 is refused (exit 2).
#
# NO --export; positional arguments only. CPU only, 8 cores, 24 GB, 4 h; c128, c166, c537, c371,
# c329 excluded. DRY_RUN=1 prints the command line and exits 0.
#
# Usage (OLD=/project/def-maxwl/mforooz/t126/D/runs, OUT=/project/def-maxwl/mforooz/t128):
#   smoke: sbatch --parsable --array=9 --output=$OUT/logs/%x_%A_%a.out \
#              $KIT/slurm/t128/oldd_split.sh $KIT $PRODUCTS $CACHE $BLACKLIST $OLD $OUT
#   full:  sbatch --parsable --array=0-47%12 --output=$OUT/logs/%x_%A_%a.out \
#              $KIT/slurm/t128/oldd_split.sh $KIT $PRODUCTS $CACHE $BLACKLIST $OLD $OUT
#SBATCH --account=def-maxwl
#SBATCH --job-name=t128_oldd_split
#SBATCH --exclude=c128,c166,c537,c371,c329
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000M

set -euo pipefail

BLACKLIST_SHA256_PREFIX=31c69342
USAGE="usage: t128/oldd_split.sh <kit_dir> <products_dir> <cache_dir> <blacklist.bed> <old_runs_dir> <out_dir>"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
CACHE="${3:?$USAGE}"
BLACKLIST="${4:?$USAGE}"
OLD="${5:?$USAGE}"
OUT="${6:?$USAGE}"
IDX="${SLURM_ARRAY_TASK_ID:?array task only}"
[[ "$IDX" =~ ^[0-9]+$ ]] || { echo "SLURM_ARRAY_TASK_ID '$IDX' is not an index" >&2; exit 2; }
case "$OUT" in
  */t128|*/t128/*) ;;
  *) echo "out dir $OUT does not contain /t128; t128 writes only under t128/" >&2; exit 2 ;;
esac
PY_MODULES="python/3.10.13"
MANIFEST="$PRODUCTS/MANIFEST.tsv"
COVARIATES="$KIT/tools/t118/covariates.tsv"
SPLIT="$OUT/oldD_split"

# The run_name of task $IDX, read from an oldd_split.py task table on stdin (columns found by header).
run_name_from() {
  awk -F'\t' -v i="$IDX" 'NR == 1 { for (c = 1; c <= NF; c++) h[$c] = c; next }
                          $h["index"] == i { print $h["run_name"] }'
}

if [ "${DRY_RUN:-0}" = 1 ]; then
  TOOLS="$KIT"
  [ -f "$TOOLS/tools/t128/oldd_split.py" ] || TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  TABLE=$(PYTHONPATH="$TOOLS/src${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-python3}" \
          "$TOOLS/tools/t128/oldd_split.py" tasks)
  RUN=$(run_name_from <<< "$TABLE")
  [ -n "$RUN" ] || { echo "no task $IDX in the oldd_split.py table" >&2; exit 2; }
  if [ -f "$SPLIT/$RUN/LAW_DONE" ]; then echo "skip $RUN: LAW_DONE exists" >&2; exit 0; fi
  [ -f "$OLD/$RUN/ckpt.pt" ] || echo "note: no $OLD/$RUN/ckpt.pt; a real task would exit 3" >&2
  echo "python3 $KIT/tools/t128/oldd_split.py run $MANIFEST $COVARIATES $CACHE $BLACKLIST $OLD/$RUN $SPLIT/$RUN --workers 8"
  exit 0
fi

[ -f "$KIT/tools/t128/oldd_split.py" ] || { echo "no tools/t128/oldd_split.py under $KIT" >&2; exit 2; }
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
mkdir -p "$SPLIT"

echo "=== t128 oldd_split idx=$IDX job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX host=$(hostname) $(date -u)"
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

RUN=$(python3 "$KIT/tools/t128/oldd_split.py" tasks | run_name_from)
[ -n "$RUN" ] || { echo "no task $IDX in the oldd_split.py table" >&2; exit 2; }
echo "run: $RUN"
if [ -f "$SPLIT/$RUN/LAW_DONE" ]; then echo "skip $RUN: LAW_DONE exists"; exit 0; fi
[ -f "$OLD/$RUN/ckpt.pt" ] || { echo "no $OLD/$RUN/ckpt.pt: old D run missing" >&2; exit 3; }

python3 "$KIT/tools/t128/oldd_split.py" run "$MANIFEST" "$COVARIATES" "$CACHE" "$BLACKLIST" \
    "$OLD/$RUN" "$SPLIT/$RUN" --workers 8

echo "=== done $RUN $(date -u)"
