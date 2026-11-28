#!/bin/bash
# t396 dedup cap test at 25 bp (C10a tool; C10b, C10c submit). numpy only.
#
#   stats: `dedup_cap.py stats` on pairs GROUP*i .. GROUP*(i+1)-1 of rows.tsv (rows with
#          in_store = 1, file order), i = SLURM_ARRAY_TASK_ID (0 outside an array).
#     export KIT=/project/def-maxwl/mforooz/onion_kit/SC10b UNIT_RULE=<UNIT_RULE.json> GROUP=4
#     sbatch --job-name=sc_dedup --export=ALL --array=0-<N-1>%12 --time=<probe> \
#         --output=<logs>/%x_%A_%a.out "$KIT/slurm/t396/dedup_cap.sh" stats
#   floor-count: `dedup_cap.py floor-count` on the val pairs, same grouping; needs FIT.
#     export KIT=... UNIT_RULE=... FIT=<per_assay.tsv> GROUP=4
#     sbatch --job-name=sc_dedup ... "$KIT/slurm/t396/dedup_cap.sh" floor-count
#
# Outputs: $OUT/<on id>.npz (stats), $OUT_FLOOR/<on id>.npz (floor-count). An existing output is
# skipped. DRY_RUN=1 prints the commands and runs nothing.
#SBATCH --account=def-maxwl
#SBATCH --job-name=sc_dedup
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --gres=none
#SBATCH --exclude=fc10713,fc10715

set -uo pipefail
MODE="${1:?usage: dedup_cap.sh stats|floor-count}"
: "${KIT:?export KIT}"
: "${UNIT_RULE:?export UNIT_RULE}"
ROWS="${ROWS:-$KIT/cruxvault/results/t396/dedup_cap/rows.tsv}"
OUT="${OUT:-/scratch/mforooz/sc396/caps/dedup}"
OUT_FLOOR="${OUT_FLOOR:-/scratch/mforooz/sc396/caps/dedup_floor}"
GROUP="${GROUP:-1}"
I="${SLURM_ARRAY_TASK_ID:-0}"
PY=/project/def-maxwl/mforooz/candi_venv/bin/python

echo "=== sc396 dedup_cap $MODE job ${SLURM_JOB_ID:-none} task $I host=$(hostname) $(date -u)"
cd "$KIT" || exit 2
[ -f "$ROWS" ] || { echo "no $ROWS" >&2; exit 2; }
source "$(dirname "$PY")/activate"
source "$KIT/slurm/_kit_pin.sh" || exit 1

case "$MODE" in
stats)       VAL_ONLY=0; DEST="$OUT" ;;
floor-count) : "${FIT:?export FIT}"; VAL_ONLY=1; DEST="$OUT_FLOOR" ;;
*) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
mkdir -p "$DEST"
# pairs of this task: rows with in_store = 1 (and in_eic = val for floor-count), GROUP per task
pairs=$(awk -F'\t' -v g="$GROUP" -v i="$I" -v v="$VAL_ONLY" '
  NR == 1 {for (k = 1; k <= NF; k++) c[$k] = k; next}
  $c["in_store"] == "1" && (v == 0 || $c["in_eic"] == "val") {
    if (int(n / g) == i) print $c["off_id"], $c["on_id"]; n++ }' "$ROWS")
[ -n "$pairs" ] || { echo "no pairs for task $I" >&2; exit 1; }
rc=0
while read -r off on; do
  f="$DEST/$on.npz"
  [ -s "$f" ] && { echo "skip $f"; continue; }
  extra=(--rows "$ROWS")
  [ "$MODE" = floor-count ] && extra+=(--fit "$FIT")
  cmd=("$PY" tools/t396/dedup_cap.py "$MODE" --pair "$off" "$on" --unit-rule "$UNIT_RULE" "${extra[@]}" --out "$f")
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "${cmd[*]}"; continue; fi
  /usr/bin/time -v "${cmd[@]}" || { echo "FAILED $on" >&2; rc=1; }
done <<< "$pairs"
echo "=== done rc=$rc $(date -u)"
exit $rc
