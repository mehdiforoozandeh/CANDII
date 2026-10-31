#!/bin/bash
# t142: whole-read thinning of ONE product per array task (SLURM_ARRAY_TASK_ID = pid_index, the
# product's position in the sorted pid list of <pseudoreps_dir>/MANIFEST.tsv, as t137 floor.sh;
# 130 products -> --array=0-129). `tools/t142/readthin.py product` streams the product's pr1/pr2
# tagAligns (MANIFEST `tagalign`, on /scratch), checks their full count against the t117
# counts25.npz bit for bit, and writes under <out_dir>/<pid>/ the floor ladder (r0.5, r0.25,
# r0.125 x seeds 0, 1 x pr1, pr2) and, for a depth arm, the track base's read set thinned to the
# arm's depth (null_s0, null_s1), each with a .json sidecar, then product.json and DONE. A task whose
# DONE exists exits 0 without work, so a resubmission redoes only the missing ones.
#
# numpy + pandas only (no torch): the scipy-stack module, no venv. NO --export: positional args.
# Out dir must be t142/thinned or under it. DRY_RUN=1 prints the python line and exits 0 before any
# module load. SLURM does not create the --output directory: mkdir -p $T142/logs first.
#
#   T142=/project/def-maxwl/mforooz/t142; mkdir -p $T142/logs
#   sbatch --parsable --array=0-129%30 --output=$T142/logs/%x_%A_%a.out \
#       $K/slurm/t142/readthin.sh $K $PR $PRODUCTS $T142/thinned
#SBATCH --account=def-maxwl
#SBATCH --job-name=t142_readthin
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=none
#SBATCH --exclude=c128,c166,c537,c371,c329

set -euo pipefail

USAGE="usage: t142/readthin.sh <kit_dir> <pseudoreps_dir> <products_dir> <out_dir>"
KIT="${1:?$USAGE}"
PR="${2:?$USAGE}"
PRODUCTS="${3:?$USAGE}"
OUT="${4:?$USAGE}"
IDX="${SLURM_ARRAY_TASK_ID:?readthin.sh is an array task}"

case "$OUT" in
  */t142/thinned|*/t142/thinned/*) ;;
  *) echo "out dir $OUT is not t142/thinned or under it; t142 writes only there" >&2; exit 2 ;;
esac

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "python3 $KIT/tools/t142/readthin.py product $PR $PRODUCTS $OUT $IDX"
  exit 0
fi

[ -f "$KIT/tools/t142/readthin.py" ] || { echo "no tools/t142/readthin.py under $KIT" >&2; exit 2; }
[ -f "$PR/MANIFEST.tsv" ] || { echo "no $PR/MANIFEST.tsv" >&2; exit 2; }
[ -f "$PRODUCTS/MANIFEST.tsv" ] || { echo "no $PRODUCTS/MANIFEST.tsv" >&2; exit 2; }
KIT=$(cd "$KIT" && pwd)

PID=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) if($i=="pid") c=i; next} {print $c}' "$PR/MANIFEST.tsv" \
      | LC_ALL=C sort -u | sed -n "$((IDX + 1))p")
[ -n "$PID" ] || { echo "no pid at index $IDX in $PR/MANIFEST.tsv" >&2; exit 2; }
if [ -f "$OUT/$PID/DONE" ]; then
  echo "$PID already done ($OUT/$PID/DONE); nothing to do"
  exit 0
fi
mkdir -p "$OUT"
rm -f "$OUT/$PID"/*.tmp.npz "$OUT/$PID"/*.json.tmp

echo "=== t142 readthin idx=$IDX pid=$PID job ${SLURM_ARRAY_JOB_ID:-}_${IDX} host=$(hostname) $(date -u)"
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || git -C "$KIT" rev-parse HEAD 2>/dev/null || echo unknown)"

set +u; module load scipy-stack; set -u
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
python3 "$KIT/tools/t142/readthin.py" product "$PR" "$PRODUCTS" "$OUT" "$IDX"
[ -f "$OUT/$PID/DONE" ] || { echo "readthin.py wrote no DONE for $PID" >&2; exit 4; }

echo "=== done $(date -u)"
