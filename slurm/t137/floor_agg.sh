#!/bin/bash
# t137: the floor aggregate, one plain job after the 130-task floor array (chain it with
# --dependency=afterok:<live array id>). `tools/t137/floor.py aggregate` needs all 130 .DONE and
# writes floor_pairs.tsv, floor_curve.tsv, floor_at_30M.tsv, floor_p15M.tsv, floor_check.txt and
# config.json under <out_dir>. Same venv as slurm/t137/floor.sh. DRY_RUN=1 prints the python line.
#
#   sbatch --parsable --dependency=afterok:$FLOOR --output=$OUT/logs/%x_%j.out \
#       $KIT/slurm/t137/floor_agg.sh $KIT $OUT $RUNGS
#SBATCH --account=def-maxwl
#SBATCH --job-name=t137_floor_agg
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --exclude=c128,c166,c537,c371,c329

set -euo pipefail

USAGE="usage: floor_agg.sh <kit_dir> <out_dir> <rungs_v2_dir>"
KIT="${1:?$USAGE}"
OUT="${2:?$USAGE}"
RUNGS="${3:?$USAGE}"

case "$OUT" in
  */t137|*/t137/*) ;;
  *) echo "out dir $OUT is not t137 or under it; t137 writes only there" >&2; exit 2 ;;
esac

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "python3 $KIT/tools/t137/floor.py aggregate $OUT $RUNGS"
  exit 0
fi

[ -f "$KIT/tools/t137/floor.py" ] || { echo "no tools/t137/floor.py under $KIT" >&2; exit 2; }
[ -d "$OUT/pairs" ] || { echo "no $OUT/pairs" >&2; exit 2; }
[ -d "$RUNGS" ] || { echo "no rungs_v2 dir $RUNGS" >&2; exit 2; }
KIT=$(cd "$KIT" && pwd)
echo "=== t137 floor_agg job ${SLURM_JOB_ID:-} host=$(hostname) $(date -u)"

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

python3 "$KIT/tools/t137/floor.py" aggregate "$OUT" "$RUNGS"

echo "=== done $(date -u)"
