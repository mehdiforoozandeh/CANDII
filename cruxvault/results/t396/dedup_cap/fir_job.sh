#!/bin/bash
# t396 C10c job script (kit SC10c, Fir, numpy only). Modes: fit | merge-fit | score.
#   fit:       fit_score.py fit-group, task = SLURM_ARRAY_TASK_ID (one assay per task; the last
#              task is the all/all point fit)
#   merge-fit: fit_*.tsv -> per_assay.tsv
#   score:     fit_score.py score on pairs GROUP*i .. GROUP*(i+1)-1 (rows with in_store = 1)
# The count-space floor runs with the C10a script: slurm/t396/dedup_cap.sh floor-count.
# Submit: export KIT=... ; sbatch --job-name=sc_dedup --export=ALL [--array=...] \
#   --output=/scratch/mforooz/sc396/logs/%x_%A_%a.out "$KIT/cruxvault/results/t396/dedup_cap/fir_job.sh" <mode>
#SBATCH --account=def-maxwl
#SBATCH --job-name=sc_dedup
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --gres=none
#SBATCH --exclude=fc10713,fc10715

set -uo pipefail
MODE="${1:?usage: fir_job.sh fit|merge-fit|score}"
: "${KIT:?export KIT}"
STATS=/scratch/mforooz/sc396/caps/dedup
FLOOR=/scratch/mforooz/sc396/caps/dedup_floor
OUTD="${OUTD:-/project/def-maxwl/mforooz/t396/dedup_cap}"
TABLES=/project/def-maxwl/mforooz/ONION/tables/eic_v1_2
HERE="$KIT/cruxvault/results/t396/dedup_cap"
ROWS="$HERE/rows.tsv"
GROUP="${GROUP:-30}"
I="${SLURM_ARRAY_TASK_ID:-0}"
PY=/project/def-maxwl/mforooz/candi_venv/bin/python

echo "=== sc396 C10c $MODE job ${SLURM_JOB_ID:-none} task $I host=$(hostname) $(date -u)"
cd "$KIT" || exit 2
source "$(dirname "$PY")/activate"
source "$KIT/slurm/_kit_pin.sh" || exit 1
mkdir -p "$OUTD"
case "$MODE" in
fit)       /usr/bin/time -v "$PY" "$HERE/fit_score.py" fit-group --stats-dir "$STATS" --rows "$ROWS" \
             --tables "$TABLES" --task "$I" --out-dir "$OUTD" ;;
merge-fit) "$PY" "$HERE/fit_score.py" merge-fit --out-dir "$OUTD" --out "$OUTD/per_assay.tsv" ;;
score)     /usr/bin/time -v "$PY" "$HERE/fit_score.py" score --stats-dir "$STATS" --rows "$ROWS" \
             --tables "$TABLES" --fit "${FIT:-$OUTD/per_assay.tsv}" --floor-dir "$FLOOR" --task "$I" \
             --group "$GROUP" --out-dir "$OUTD" ;;
*) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
rc=$?
echo "=== done rc=$rc $(date -u)"
exit $rc
