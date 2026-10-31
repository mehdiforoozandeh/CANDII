#!/bin/bash
# t130: the nested-read oracle for the t112 depth pairs (tools/t130/nested_oracle.py; PI ruling
# 2026-10-30, plan_gaps.md A9). Mode `run` is an array task over the tracks with a depth arm
# (SLURM_ARRAY_TASK_ID = the track's index in `nested_oracle.py tracks <manifest>`; 7 tracks ->
# --array=0-6): it scores every depth pair of the track in counts on chr19 + chr21 (blacklist
# removed) and writes <out>/parts/<track>.{oracle,dispersion}.tsv + <track>.DONE. A task whose
# .DONE exists exits 0 without work. Mode `aggregate` (stdlib, no venv) concatenates the parts into
# <out>/nested_oracle.tsv, nested_dispersion.tsv, NESTED_DONE.
#
# NO --export: every input is a POSITIONAL argument. `run` imports baseline_rungs (imports candi,
# which imports torch): the venv is slurm/t137/floor.sh's recipe. Out dir must be t130 or under it.
# DRY_RUN=1 prints the python command line and exits before any module load.
#
# Usage, from the Nibi login node, after snapshotting the repo to $KIT:
#   OUT=/project/def-maxwl/mforooz/t130/nested; mkdir -p $OUT/logs
#   J=$(sbatch --parsable --array=0-6 --output=$OUT/logs/%x_%A_%a.out \
#       $KIT/slurm/t130/nested_oracle.sh run $KIT $PRODUCTS $BL $OUT)
#   sbatch --dependency=afterok:$J --time=0:15:00 --mem=2G --output=$OUT/logs/%x_%j.out \
#       $KIT/slurm/t130/nested_oracle.sh aggregate $KIT $PRODUCTS $BL $OUT
# Expected per run task: under 20 min (15 pair-predictives x 2 depth-ratio sources x 2 subsets; one
# scale search on 1 M synthetic bins took 0.5-1.5 s locally), under 4 GB; asked: 1 h, 8 GB.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t130_nested
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=none
#SBATCH --exclude=c128,c166,c537,c371,c329

set -euo pipefail

USAGE="usage: nested_oracle.sh run|aggregate <kit_dir> <products_dir> <blacklist.bed> <out_dir>"
MODE="${1:?$USAGE}"
KIT="${2:?$USAGE}"
PRODUCTS="${3:?$USAGE}"
BL="${4:?$USAGE}"
OUT="${5:?$USAGE}"

case "$OUT" in
  */t130|*/t130/*) ;;
  *) echo "out dir $OUT is not t130 or under it; t130 writes only there" >&2; exit 2 ;;
esac
case "$MODE" in
  run) IDX="${SLURM_ARRAY_TASK_ID:?mode run is an array task}" ;;
  aggregate) ;;
  *) echo "$USAGE" >&2; exit 2 ;;
esac

PY="$KIT/tools/t130/nested_oracle.py"
MAN="$PRODUCTS/MANIFEST.tsv"
if [ "${DRY_RUN:-0}" = 1 ]; then
  if [ "$MODE" = run ]; then echo "python3 $PY run $MAN $PRODUCTS $BL $OUT $IDX"
  else echo "python3 $PY aggregate $MAN $OUT"; fi
  exit 0
fi

[ -f "$PY" ] || { echo "no tools/t130/nested_oracle.py under $KIT" >&2; exit 2; }
[ -f "$MAN" ] || { echo "no $MAN" >&2; exit 2; }
mkdir -p "$OUT/parts"
echo "=== t130 nested $MODE job ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-none}}_${IDX:-} host=$(hostname) $(date -u)"
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || git -C "$KIT" rev-parse HEAD 2>/dev/null || echo unknown)"

if [ "$MODE" = aggregate ]; then
  set +u; module load scipy-stack; set -u
  python3 "$PY" aggregate "$MAN" "$OUT"
  ls -la "$OUT"
  echo "=== done $(date -u)"
  exit 0
fi

[ -s "$BL" ] || { echo "blacklist $BL missing or empty" >&2; exit 2; }
echo "blacklist sha256: $(sha256sum "$BL" | cut -d' ' -f1)"
KIT=$(cd "$KIT" && pwd)
set +u; module load python/3.10.13; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
ls "$KIT"/wheels/x_transformers-*-py3-none-any.whl >/dev/null 2>&1 \
  || { echo "no x_transformers wheel in $KIT/wheels" >&2; exit 2; }
pip install --no-index -r "$KIT/requirements-fir.txt"
pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
export PYTHONPATH="$KIT/src"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
case "$(python3 -c 'import candi; print(candi.__file__)')" in
  "$KIT"/src/*) ;;
  *) echo "candi does not import from $KIT/src" >&2; exit 3 ;;
esac

# the track lookup imports numpy, so it runs inside the venv
TRACK=$(python3 "$PY" tracks "$MAN" | awk -F'\t' -v i="$IDX" '$1 == i {print $2}')
[ -n "$TRACK" ] || { echo "no track at index $IDX" >&2; exit 2; }
if [ -f "$OUT/parts/$TRACK.DONE" ]; then
  echo "$TRACK already done ($OUT/parts/$TRACK.DONE); nothing to do"
  exit 0
fi

echo "track $TRACK"
python3 "$PY" run "$MAN" "$PRODUCTS" "$BL" "$OUT" "$IDX"
echo "=== done $(date -u)"
