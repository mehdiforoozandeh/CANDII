#!/bin/bash
# t142: the t137 resampling floor, WHOLE-READ thinned (PI 2026-10-30), of ONE product per array
# task (SLURM_ARRAY_TASK_ID = pid_index, the
# product's position in the sorted pid list of <pseudoreps_dir>/MANIFEST.tsv; 130 products ->
# --array=0-129). `tools/t137/floor.py product` thins the product's pr1/pr2 halves along the depth
# ladder (ratio 1, 0.5, 0.25, 0.125; seeds 0, 1), scores them on chr19 + chr21 (blacklist removed)
# and writes <out>/pairs/<pid>.tsv (14 rows), <out>/bins1k/<pid>.npz and <out>/pairs/<pid>.DONE.
# A task whose .DONE exists exits 0 without work, so a resubmission redoes only the missing ones.
# The thinned halves (ratio < 1) are READ from <thinned_dir>/<pid>/ (slurm/t142/readthin.sh, which
# must be done for this pid: chain with --dependency=aftercorr:<live readthin array>, same indices).
#
# NO --export: every input is a POSITIONAL argument. The venv is slurm/t118/qm_rungs.sh's recipe
# (baseline_rungs imports candi, which imports torch): %12. Out dir must be t142/floor or under it.
# DRY_RUN=1 prints the python command line and exits before any module load.
#
# Usage, from the Nibi login node, after snapshotting the repo to $KIT:
#   T142=/project/def-maxwl/mforooz/t142; mkdir -p $T142/logs
#   sbatch --parsable --array=0-129%12 --dependency=aftercorr:$RT --output=$T142/logs/%x_%A_%a.out \
#       $KIT/slurm/t142/floor.sh $KIT $PR $PRODUCTS $BL $T142/floor $T142/thinned
#SBATCH --account=def-maxwl
#SBATCH --job-name=t142_floor
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=none
#SBATCH --exclude=c128,c166,c537,c371,c329

set -euo pipefail

USAGE="usage: t142/floor.sh <kit_dir> <pseudoreps_dir> <products_dir> <blacklist.bed> <out_dir> <thinned_dir>"
KIT="${1:?$USAGE}"
PR="${2:?$USAGE}"
PRODUCTS="${3:?$USAGE}"
BL="${4:?$USAGE}"
OUT="${5:?$USAGE}"
THIN="${6:?$USAGE}"
IDX="${SLURM_ARRAY_TASK_ID:?floor.sh is an array task}"

case "$OUT" in
  */t142/floor|*/t142/floor/*) ;;
  *) echo "out dir $OUT is not t142/floor or under it; t142 writes only there" >&2; exit 2 ;;
esac

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "python3 $KIT/tools/t137/floor.py product $PR $PRODUCTS $BL $OUT $IDX --thinned $THIN"
  exit 0
fi

[ -f "$KIT/tools/t137/floor.py" ] || { echo "no tools/t137/floor.py under $KIT" >&2; exit 2; }
[ -f "$PR/MANIFEST.tsv" ] || { echo "no $PR/MANIFEST.tsv" >&2; exit 2; }
[ -f "$PRODUCTS/MANIFEST.tsv" ] || { echo "no $PRODUCTS/MANIFEST.tsv" >&2; exit 2; }
[ -s "$BL" ] || { echo "blacklist $BL missing or empty" >&2; exit 2; }
KIT=$(cd "$KIT" && pwd)

PID=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) if($i=="pid") c=i; next} {print $c}' "$PR/MANIFEST.tsv" \
      | LC_ALL=C sort -u | sed -n "$((IDX + 1))p")
[ -n "$PID" ] || { echo "no pid at index $IDX in $PR/MANIFEST.tsv" >&2; exit 2; }
if [ -f "$OUT/pairs/$PID.DONE" ]; then
  echo "$PID already done ($OUT/pairs/$PID.DONE); nothing to do"
  exit 0
fi
[ -f "$THIN/$PID/DONE" ] || { echo "no $THIN/$PID/DONE: run slurm/t142/readthin.sh first" >&2; exit 2; }
mkdir -p "$OUT/pairs" "$OUT/bins1k"

echo "=== t142 floor (read thinning) idx=$IDX pid=$PID job ${SLURM_ARRAY_JOB_ID:-}_${IDX} host=$(hostname) $(date -u)"
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || git -C "$KIT" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "blacklist sha256: $(sha256sum "$BL" | cut -d' ' -f1)"

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
if [ -f "$KIT/GIT_SHA" ]; then export GIT_SHA=$(cat "$KIT/GIT_SHA"); fi
case "$(python3 -c 'import candi; print(candi.__file__)')" in
  "$KIT"/src/*) ;;
  *) echo "candi does not import from $KIT/src" >&2; exit 3 ;;
esac

python3 "$KIT/tools/t137/floor.py" product "$PR" "$PRODUCTS" "$BL" "$OUT" "$IDX" --thinned "$THIN"

echo "=== done $(date -u)"
