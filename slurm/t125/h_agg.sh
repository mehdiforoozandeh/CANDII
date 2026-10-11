#!/bin/bash
# t125 part 2: aggregate every h scoring (alone and in the chain) into the part-2 tables
# (complexity x inputs, push only against push + own noise, the repair reading, the chain against
# the old grid's p results, exploding pairs). One plain job (not an array); it reads t118's row-2
# results.json and writes nothing outside <out_root>/agg.
#
# WHAT THE JOB DOES (agg_dir = <out_root>/agg):
#   python3 $KIT/tools/t125/hmap/aggregate.py <manifest> <covariates.tsv> <out_root> <agg_dir> \
#       --old-grid-tsv <old_grid_tsv> \
#       --old-grid-json /project/def-maxwl/mforooz/t118/row2/agg/results.json \
#       --eta-runs <eta_runs_dir>
#           -> results_h.json, h_alone_summary.tsv, h_chain_summary.tsv, h_repair.tsv,
#              h_vs_oldgrid.tsv, grid_h.md, summary_data_h.json
# aggregate.py reads <out_root>/alone and <out_root>/chain. Re-runnable: every step overwrites.
# The old-grid TSV is the local cruxvault/results/row2/results_summary.tsv, rsynced up first; the
# old-grid JSON is t118's row-2 aggregate on Nibi (read-only here).
#
# NO --export. Every input is a POSITIONAL argument. Every output is under <out_root>/agg:
#   OUT_H=/project/def-maxwl/mforooz/t125/h
#   CTL=/project/def-maxwl/mforooz/t125/ctl_cache     (not read by this job)
#   ETA_RUNS=/project/def-maxwl/mforooz/t125/eta/runs  (read-only here)
# An <out_root> under a t118 directory is refused (exit 2).
#
# THE VENV is the pinned recipe of slurm/t118/row2_train.sh (python/3.10.13, requirements-fir.txt,
# then requirements-pypi.txt from $KIT/wheels), built in $SLURM_TMPDIR; PYTHONPATH is "$KIT/src"
# and the t125 CLIs put tools/t118 and tools/t125 on sys.path themselves.
#
# RESOURCES. CPU only, no GPU request. 4 cores, 16 GB, 2 h. Nodes c128, c166, c537 and c371 are
# excluded. Peak memory from `sacct -o MaxRSS` (no /usr/bin/time on compute nodes).
#
# DRY_RUN=1 prints the python command line and exits 0 before any module load; nothing is checked
# except that <out_root> is not under a t118 directory.
#
# Usage, from the Nibi login node, once the alone and chain scorings have their ALONE_DONE and
# CHAIN_DONE (check the files; never a dependency on an array that may have finished). Record the
# job id in plan/T125_STATUS.md:
#   OUT_H=/project/def-maxwl/mforooz/t125/h; mkdir -p $OUT_H/logs
#   OLD=/project/def-maxwl/mforooz/t125/old_grid_results_summary.tsv
#   ETA_RUNS=/project/def-maxwl/mforooz/t125/eta/runs
#   sbatch --test-only --exclude=c128,c166,c537,c371 --output=$OUT_H/logs/%x_%j.out \
#       $KIT/slurm/t125/h_agg.sh $KIT $PRODUCTS $OUT_H $OLD $ETA_RUNS
#   sbatch --parsable --exclude=c128,c166,c537,c371 --output=$OUT_H/logs/%x_%j.out \
#       $KIT/slurm/t125/h_agg.sh $KIT $PRODUCTS $OUT_H $OLD $ETA_RUNS
#SBATCH --account=def-maxwl
#SBATCH --job-name=t125_hagg
#SBATCH --exclude=c128,c166,c537,c371
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M

set -euo pipefail

OLD_GRID_JSON=/project/def-maxwl/mforooz/t118/row2/agg/results.json
USAGE="usage: h_agg.sh <kit_dir> <products_dir> <out_root> <old_grid_tsv> <eta_runs_dir>"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
OUT_ROOT="${3:?$USAGE}"
OLD_GRID_TSV="${4:?$USAGE}"
ETA_RUNS="${5:?$USAGE}"
case "$OUT_ROOT" in
  */t118|*/t118/*) echo "out root $OUT_ROOT is under a t118 directory; t125 writes only under t125/" >&2; exit 2 ;;
esac
PY_MODULES="python/3.10.13"
MANIFEST="$PRODUCTS/MANIFEST.tsv"
COVARIATES="$KIT/tools/t118/covariates.tsv"
AGG="$OUT_ROOT/agg"

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "python3 $KIT/tools/t125/hmap/aggregate.py $MANIFEST $COVARIATES $OUT_ROOT $AGG --old-grid-tsv $OLD_GRID_TSV --old-grid-json $OLD_GRID_JSON --eta-runs $ETA_RUNS"
  exit 0
fi

[ -f "$KIT/tools/t125/hmap/aggregate.py" ] || { echo "no tools/t125/hmap/aggregate.py under $KIT" >&2; exit 2; }
[ -f "$COVARIATES" ] || { echo "no $COVARIATES" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "no $MANIFEST" >&2; exit 2; }
[ -d "$OUT_ROOT/alone" ] || { echo "no alone dir $OUT_ROOT/alone" >&2; exit 2; }
[ -d "$OUT_ROOT/chain" ] || { echo "no chain dir $OUT_ROOT/chain" >&2; exit 2; }
[ -d "$ETA_RUNS" ] || { echo "no eta runs dir $ETA_RUNS" >&2; exit 2; }
[ -f "$OLD_GRID_TSV" ] || { echo "no old-grid TSV $OLD_GRID_TSV (rsync cruxvault/results/row2/results_summary.tsv up first)" >&2; exit 2; }
[ -f "$OLD_GRID_JSON" ] || { echo "no old-grid JSON $OLD_GRID_JSON" >&2; exit 2; }
KIT=$(cd "$KIT" && pwd)
COVARIATES="$KIT/tools/t118/covariates.tsv"
mkdir -p "$AGG"

echo "=== t125 h_agg job ${SLURM_JOB_ID:-none} host=$(hostname) $(date -u)"
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || git -C "$KIT" rev-parse HEAD 2>/dev/null || echo unknown)"

set +u; module load $PY_MODULES; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
ls "$KIT"/wheels/x_transformers-*-py3-none-any.whl >/dev/null 2>&1 \
  || { echo "no x_transformers wheel in $KIT/wheels" >&2; exit 2; }
pip install --no-index -r "$KIT/requirements-fir.txt"
pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
export PYTHONPATH="$KIT/src"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
if [ -f "$KIT/GIT_SHA" ]; then export T125_GIT_SHA=$(cat "$KIT/GIT_SHA"); fi
python3 -c "import candi.metrics, candi.bench.distributional, candi.store.genome; print('venv ok', candi.__file__)"
# the library must come from this kit, not from some other install (tests/test_slurm_kit_pin.py)
case "$(python3 -c 'import candi; print(candi.__file__)')" in
  "$KIT"/src/*) ;;
  *) echo "candi does not import from $KIT/src" >&2; exit 3 ;;
esac

python3 "$KIT/tools/t125/hmap/aggregate.py" "$MANIFEST" "$COVARIATES" "$OUT_ROOT" "$AGG" \
    --old-grid-tsv "$OLD_GRID_TSV" --old-grid-json "$OLD_GRID_JSON" --eta-runs "$ETA_RUNS"

echo "=== done $(date -u)"
ls -l "$AGG/results_h.json" "$AGG/grid_h.md"
