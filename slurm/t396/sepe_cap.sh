#!/bin/bash
# t396 SE/PE cap test at 25 bp (C10a tool; C10b, C11 submit). numpy only.
#
#   stats: `sepe_cap.py stats` on pairs GROUP*i .. GROUP*(i+1)-1 of rows.tsv (file order),
#          i = SLURM_ARRAY_TASK_ID (0 outside an array).
#     export KIT=/project/def-maxwl/mforooz/onion_kit/SC10b UNIT_RULE=<UNIT_RULE.json> GROUP=2
#     sbatch --job-name=sc_sepe --export=ALL --array=0-<N-1>%12 --time=<probe> \
#         --output=<logs>/%x_%A_%a.out "$KIT/slurm/t396/sepe_cap.sh" stats
#   floor-count: `sepe_cap.py floor-count` on every pair (all are scored leave-one-experiment-
#          out), same grouping; needs FIT (sepe per_assay.tsv).
#     export KIT=... UNIT_RULE=... FIT=<per_assay.tsv> GROUP=2
#     sbatch --job-name=sc_sepe ... "$KIT/slurm/t396/sepe_cap.sh" floor-count
#
# Outputs: $OUT/<se id>.npz (stats), $OUT_FLOOR/<se id>.npz (floor-count). An existing output is
# skipped. DRY_RUN=1 prints the commands and runs nothing.
#SBATCH --account=def-maxwl
#SBATCH --job-name=sc_sepe
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --gres=none
#SBATCH --exclude=fc10713,fc10715

set -uo pipefail
MODE="${1:?usage: sepe_cap.sh stats|floor-count}"
: "${KIT:?export KIT}"
: "${UNIT_RULE:?export UNIT_RULE}"
ROWS="${ROWS:-$KIT/cruxvault/results/t396/sepe_cap/rows.tsv}"
OUT="${OUT:-/scratch/mforooz/sc396/caps/sepe}"
OUT_FLOOR="${OUT_FLOOR:-/scratch/mforooz/sc396/caps/sepe_floor}"
GROUP="${GROUP:-1}"
I="${SLURM_ARRAY_TASK_ID:-0}"
PY=/project/def-maxwl/mforooz/candi_venv/bin/python

echo "=== sc396 sepe_cap $MODE job ${SLURM_JOB_ID:-none} task $I host=$(hostname) $(date -u)"
cd "$KIT" || exit 2
[ -f "$ROWS" ] || { echo "no $ROWS" >&2; exit 2; }
source "$(dirname "$PY")/activate"
source "$KIT/slurm/_kit_pin.sh" || exit 1

case "$MODE" in
stats)       DEST="$OUT" ;;
floor-count) : "${FIT:?export FIT}"; DEST="$OUT_FLOOR" ;;
*) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
mkdir -p "$DEST"
pairs=$(awk -F'\t' -v g="$GROUP" -v i="$I" '
  NR == 1 {for (k = 1; k <= NF; k++) c[$k] = k; next}
  { if (int(n / g) == i) print $c["parent_id"], $c["se_id"]; n++ }' "$ROWS")
[ -n "$pairs" ] || { echo "no pairs for task $I" >&2; exit 1; }
rc=0
while read -r par se; do
  f="$DEST/$se.npz"
  [ -s "$f" ] && { echo "skip $f"; continue; }
  extra=()
  [ "$MODE" = floor-count ] && extra=(--fit "$FIT" --rows "$ROWS")
  cmd=("$PY" tools/t396/sepe_cap.py "$MODE" --pair "$par" "$se" --unit-rule "$UNIT_RULE" "${extra[@]}" --out "$f")
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "${cmd[*]}"; continue; fi
  /usr/bin/time -v "${cmd[@]}" || { echo "FAILED $se" >&2; rc=1; }
done <<< "$pairs"
echo "=== done rc=$rc $(date -u)"
exit $rc
