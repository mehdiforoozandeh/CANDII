#!/bin/bash
# t141 (q11): one t112 ARM product per array task (SLURM_ARRAY_TASK_ID = arm_index, the product's
# position in the sorted list of non-base pids of <products_dir>/MANIFEST.tsv; base products are
# skipped). `tools/t141/arms.py arm` sums base and arm to the 1 kb grid (counts and -log10 p; every
# chromosome but chrM/chrY), fits the one-warp (QuantileMatching base -> arm, genome-wide), writes
# the per-bin residual, and for the depth arm the binomial-thinning null (seeds 0, 1), to
# <out>/arms1k/<pid>.npz (the DNase mapq arms get <pid>.DROPPED). A task whose output exists exits 0
# without work, so a resubmission redoes only the missing ones.
#
# NO --export: every input is a POSITIONAL argument. The venv is slurm/t137/floor.sh's recipe
# (baseline_rungs, which holds the QuantileMatching rule, imports candi, which imports torch). Out
# dir must be t141 or under it. DRY_RUN=1 prints the python command line and exits 0 before any
# module load. SLURM does not create the --output directory: mkdir -p $OUT/logs first.
#
# %12: each task imports torch (via candi) off the shared venv; >12 at once fail (Fir lesson).
# Usage, from the Nibi login node, after snapshotting the repo to $KIT:
#   OUT=/project/def-maxwl/mforooz/t141; mkdir -p $OUT/logs
#   N=$(python3 $KIT/tools/t141/arms.py list $PRODUCTS | wc -l)     # 123 arm products
#   sbatch --parsable --array=0-$((N - 1))%12 --output=$OUT/logs/%x_%A_%a.out \
#       $KIT/slurm/t141/arms.sh $KIT $PRODUCTS $OUT
#SBATCH --account=def-maxwl
#SBATCH --job-name=t141_arms
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=none
#SBATCH --exclude=c128,c166,c537,c371,c329

set -euo pipefail

USAGE="usage: t141/arms.sh <kit_dir> <products_dir> <out_dir>"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
OUT="${3:?$USAGE}"
IDX="${SLURM_ARRAY_TASK_ID:?arms.sh is an array task}"

case "$OUT" in
  */t141|*/t141/*) ;;
  *) echo "out dir $OUT is not t141 or under it; t141 writes only there" >&2; exit 2 ;;
esac

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "python3 $KIT/tools/t141/arms.py arm $PRODUCTS $OUT $IDX"
  exit 0
fi

[ -f "$KIT/tools/t141/arms.py" ] || { echo "no tools/t141/arms.py under $KIT" >&2; exit 2; }
[ -f "$PRODUCTS/MANIFEST.tsv" ] || { echo "no $PRODUCTS/MANIFEST.tsv" >&2; exit 2; }
KIT=$(cd "$KIT" && pwd)

PID=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++){if($i=="pid")p=i; if($i=="arm")a=i}; next}
                  $a!="base"{print $p}' "$PRODUCTS/MANIFEST.tsv" | LC_ALL=C sort -u | sed -n "$((IDX + 1))p")
[ -n "$PID" ] || { echo "no arm pid at index $IDX in $PRODUCTS/MANIFEST.tsv" >&2; exit 2; }
if [ -f "$OUT/arms1k/$PID.npz" ] || [ -f "$OUT/arms1k/$PID.DROPPED" ]; then
  echo "$PID already done under $OUT/arms1k; nothing to do"
  exit 0
fi
mkdir -p "$OUT/arms1k"
rm -f "$OUT/arms1k/$PID.tmp.npz"

echo "=== t141 arms idx=$IDX pid=$PID job ${SLURM_ARRAY_JOB_ID:-}_${IDX} host=$(hostname) $(date -u)"
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

python3 "$KIT/tools/t141/arms.py" arm "$PRODUCTS" "$OUT" "$IDX"
# the python and the shell must agree on which pid this index is
[ -f "$OUT/arms1k/$PID.npz" ] || [ -f "$OUT/arms1k/$PID.DROPPED" ] \
  || { echo "arms.py wrote no output for $PID" >&2; exit 4; }

echo "=== done $(date -u)"
