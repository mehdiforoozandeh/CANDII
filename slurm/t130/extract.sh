#!/bin/bash
# t130 blind check-set readings, step 1: flatten the h1/h2 check-set records of the 576 row-1
# ladder runs and the rungs_v2 baselines (tools/t130/extract_records.py, stdlib only). Read-only
# on the runs and the baselines; writes only under <out_dir>, which must be the t130 tree.
#
#   OUT=/project/def-maxwl/mforooz/t130; mkdir -p $OUT/logs
#   sbatch --output=$OUT/logs/%x_%j.out $KIT/slurm/t130/extract.sh \
#       $KIT /project/def-maxwl/mforooz/t118/ladder/runs /project/def-maxwl/mforooz/t118/rungs_v2 $OUT
#   DRY_RUN=1 bash slurm/t130/extract.sh <kit> <runs> <rungs_v2> <out>   # print the python line
#SBATCH --account=def-maxwl
#SBATCH --job-name=t130_extract
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --exclude=c128,c166,c537,c371,c329

set -euo pipefail
USAGE="usage: extract.sh <kit_dir> <ladder_runs_dir> <rungs_v2_dir> <out_dir>"
KIT="${1:?$USAGE}"; RUNS="${2:?$USAGE}"; RUNGS="${3:?$USAGE}"; OUT="${4:?$USAGE}"
case "$OUT" in
  */t130|*/t130/*) ;;
  *) echo "refusing out_dir $OUT: it must be the t130 tree (*/t130 or */t130/*)" >&2; exit 2 ;;
esac
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "python3 $KIT/tools/t130/extract_records.py $RUNS $RUNGS $OUT"
  exit 0
fi
echo "=== t130 extract job ${SLURM_JOB_ID:-none} host=$(hostname) $(date -u)"
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || echo unknown)"
set +u; module load scipy-stack; set -u
mkdir -p "$OUT"
python3 "$KIT/tools/t130/extract_records.py" "$RUNS" "$RUNGS" "$OUT"
ls -la "$OUT"
echo "=== done $(date -u)"
