#!/bin/bash
# t142: the t141 depth null redone by WHOLE-READ thinning of the base (PI 2026-10-30), one t112
# DEPTH arm product per array task (other arms are refused: their output does not change)
# (SLURM_ARRAY_TASK_ID = arm_index, the product's position in the sorted list of non-base pids
# of <products_dir>/MANIFEST.tsv; base products are skipped). `tools/t141/arms.py arm` sums base
# and arm to the 1 kb grid (counts and -log10 p; every
# chromosome but chrM/chrY), fits the one-warp (QuantileMatching base -> arm, genome-wide), writes
# the per-bin residual, and for the depth arm the whole-read-thinning null (seeds 0, 1), to
# <out>/arms1k/<pid>.npz (the DNase mapq arms get <pid>.DROPPED). A task whose output exists exits 0
# without work, so a resubmission redoes only the missing ones.
#
# The thinned base is READ from <null_dir>/<pid>/null_s{0,1}.npz (slurm/t142/readthin.sh; chain with
# --dependency=afterok:<live readthin array>: the arm index is not the readthin index).
# NO --export: every input is a POSITIONAL argument. The venv is slurm/t137/floor.sh's recipe
# (baseline_rungs, which holds the QuantileMatching rule, imports candi, which imports torch). Out
# dir must be t142/t141 or under it. DRY_RUN=1 prints the python command line and exits 0 before any
# module load. SLURM does not create the --output directory: mkdir -p $OUT/logs first.
#
# %12: each task imports torch (via candi) off the shared venv; >12 at once fail (Fir lesson).
# Usage, from the Nibi login node, after snapshotting the repo to $KIT:
#   T142=/project/def-maxwl/mforooz/t142; mkdir -p $T142/logs
#   IDX=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++){if($i=="pid")p=i; if($i=="arm")a=i}; next}
#         $a!="base"{print $p"\t"$a}' $PRODUCTS/MANIFEST.tsv | LC_ALL=C sort -u \
#         | awk -F'\t' '$2=="depth"{printf "%s%d", (n++ ? "," : ""), NR-1}')   # 21 depth arms
#   sbatch --parsable --array=$IDX%12 --dependency=afterok:$RT --output=$T142/logs/%x_%A_%a.out \
#       $KIT/slurm/t142/arms.sh $KIT $PRODUCTS $T142/t141 $T142/thinned
#SBATCH --account=def-maxwl
#SBATCH --job-name=t142_arms
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=none
#SBATCH --exclude=c128,c166,c537,c371,c329

set -euo pipefail

USAGE="usage: t142/arms.sh <kit_dir> <products_dir> <out_dir> <null_dir>"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
OUT="${3:?$USAGE}"
NULLD="${4:?$USAGE}"
IDX="${SLURM_ARRAY_TASK_ID:?arms.sh is an array task}"

case "$OUT" in
  */t142/t141|*/t142/t141/*) ;;
  *) echo "out dir $OUT is not t142/t141 or under it; t142 writes only there" >&2; exit 2 ;;
esac

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "python3 $KIT/tools/t141/arms.py arm $PRODUCTS $OUT $IDX $NULLD"
  exit 0
fi

[ -f "$KIT/tools/t141/arms.py" ] || { echo "no tools/t141/arms.py under $KIT" >&2; exit 2; }
[ -f "$PRODUCTS/MANIFEST.tsv" ] || { echo "no $PRODUCTS/MANIFEST.tsv" >&2; exit 2; }
KIT=$(cd "$KIT" && pwd)

LINE=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++){if($i=="pid")p=i; if($i=="arm")a=i}; next}
                  $a!="base"{print $p"\t"$a}' "$PRODUCTS/MANIFEST.tsv" | LC_ALL=C sort -u | sed -n "$((IDX + 1))p")
PID=${LINE%%$'\t'*}
[ -n "$PID" ] || { echo "no arm pid at index $IDX in $PRODUCTS/MANIFEST.tsv" >&2; exit 2; }
[ "${LINE#*$'\t'}" = depth ] || { echo "$PID is not a depth arm; t142 redoes the depth null only" >&2; exit 2; }
for s in 0 1; do
  [ -s "$NULLD/$PID/null_s$s.npz" ] || { echo "no $NULLD/$PID/null_s$s.npz: run readthin.sh first" >&2; exit 2; }
done
if [ -f "$OUT/arms1k/$PID.npz" ] || [ -f "$OUT/arms1k/$PID.DROPPED" ]; then
  echo "$PID already done under $OUT/arms1k; nothing to do"
  exit 0
fi
mkdir -p "$OUT/arms1k"
rm -f "$OUT/arms1k/$PID.tmp.npz"

echo "=== t142 arms (read-thinned null) idx=$IDX pid=$PID job ${SLURM_ARRAY_JOB_ID:-}_${IDX} host=$(hostname) $(date -u)"
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || git -C "$KIT" rev-parse HEAD 2>/dev/null || echo unknown)"

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

python3 "$KIT/tools/t141/arms.py" arm "$PRODUCTS" "$OUT" "$IDX" "$NULLD"
# the python and the shell must agree on which pid this index is
[ -f "$OUT/arms1k/$PID.npz" ] || [ -f "$OUT/arms1k/$PID.DROPPED" ] \
  || { echo "arms.py wrote no output for $PID" >&2; exit 4; }

echo "=== done $(date -u)"
