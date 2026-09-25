#!/bin/bash
# t118 ladder: aggregate every finished run, then write one rung's figures and report. One plain
# job (not an array) per rung.
#
# WHAT THE JOB DOES, in order (agg_dir = <out_dir>/agg):
#   1. python3 $KIT/tools/t118/ladder/aggregate.py <manifest> <covariates.tsv> <out_dir>/runs \
#          <refs_tsv> <agg_dir>                 -> results.json, results_summary.tsv, checks_*.json
#   2. python3 $KIT/tools/t118/ladder/aggregate.py qm-curves <manifest> <products_dir> \
#          <agg_dir>/qm_curves.json             -> the QuantileMatching curves for figure 5
#   3. python3 $KIT/tools/t118/ladder/figures.py <agg_dir> <rung> --refs-qm <agg_dir>/qm_curves.json
#                                               -> <agg_dir>/<rung>/figures/fig1..fig9 PNGs
#   4. python3 $KIT/tools/t118/ladder/report.py <agg_dir> <rung>   -> <agg_dir>/<rung>/report.md
# Re-runnable: every step always overwrites. Runs still missing are listed in results.json, not
# fatal. A missing refs_tsv gives refs = {} with a warning inside aggregate.py.
# The QM curves read the products dir, not the cache: the foreman deletes the cache once every
# law task is done, and this job may run after that.
#
# NO --export. Every input is a POSITIONAL argument. Every output is under <out_dir>/agg (the
# foreman passes /project/def-maxwl/mforooz/t118/ladder).
#
# THE VENV is the repo's pinned recipe, copied from slurm/t118/qm_rungs.sh (python/3.10.13,
# requirements-fir.txt, which carries matplotlib for figures.py, then requirements-pypi.txt from
# $KIT/wheels), built in $SLURM_TMPDIR.
#
# RESOURCES. CPU only, no GPU request. 4 cores, 16 GB, 2 h. Peak memory from `sacct -o MaxRSS`
# (no /usr/bin/time on compute nodes).
#
# DRY_RUN=1 prints the four python command lines and exits 0 before any module load; nothing is
# checked.
#
# Usage, from the Nibi login node, once all 144 runs of <rung> have their SCORE_DONE and LAW_DONE
# (check the files; do not chain onto arrays that may have finished):
#   OUT=/project/def-maxwl/mforooz/t118/ladder; mkdir -p $OUT/logs
#   REFS=/project/def-maxwl/mforooz/t118/rungs_v2/rungs_v2.tsv
#   sbatch --test-only --output=$OUT/logs/%x_%j.out \
#       $KIT/slurm/t118/ladder_agg.sh $KIT $PRODUCTS $OUT $REFS A
#   sbatch --parsable  --output=$OUT/logs/%x_%j.out \
#       $KIT/slurm/t118/ladder_agg.sh $KIT $PRODUCTS $OUT $REFS A
#SBATCH --account=def-maxwl
#SBATCH --job-name=t118L_agg
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M

set -euo pipefail

USAGE="usage: ladder_agg.sh <kit_dir> <products_dir> <out_dir> <refs_tsv> <rung A|B|C|D>"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
OUT="${3:?$USAGE}"
REFS="${4:?$USAGE}"
RUNG="${5:?$USAGE}"
case "$RUNG" in A|B|C|D) ;; *) echo "rung must be one of A B C D; got '$RUNG'" >&2; exit 2 ;; esac
PY_MODULES="python/3.10.13"
MANIFEST="$PRODUCTS/MANIFEST.tsv"
COVARIATES="$KIT/tools/t118/covariates.tsv"
RUNS="$OUT/runs"
AGG="$OUT/agg"

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "python3 $KIT/tools/t118/ladder/aggregate.py $MANIFEST $COVARIATES $RUNS $REFS $AGG"
  echo "python3 $KIT/tools/t118/ladder/aggregate.py qm-curves $MANIFEST $PRODUCTS $AGG/qm_curves.json"
  echo "python3 $KIT/tools/t118/ladder/figures.py $AGG $RUNG --refs-qm $AGG/qm_curves.json"
  echo "python3 $KIT/tools/t118/ladder/report.py $AGG $RUNG"
  exit 0
fi

for tool in aggregate.py figures.py report.py; do
  [ -f "$KIT/tools/t118/ladder/$tool" ] || { echo "no tools/t118/ladder/$tool under $KIT" >&2; exit 2; }
done
[ -f "$COVARIATES" ] || { echo "no $COVARIATES" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "no $MANIFEST" >&2; exit 2; }
[ -d "$RUNS" ] || { echo "no runs dir $RUNS" >&2; exit 2; }
[ -f "$REFS" ] || echo "warning: refs $REFS missing; aggregate.py will use refs = {}" >&2
KIT=$(cd "$KIT" && pwd)
COVARIATES="$KIT/tools/t118/covariates.tsv"
mkdir -p "$AGG"

echo "=== t118 ladder_agg rung=$RUNG job ${SLURM_JOB_ID:-none} host=$(hostname) $(date -u)"
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
if [ -f "$KIT/GIT_SHA" ]; then export T118_GIT_SHA=$(cat "$KIT/GIT_SHA"); fi
python3 -c "import candi.metrics, candi.bench.distributional, candi.store.genome; print('venv ok', candi.__file__)"
# the library must come from this kit, not from some other install (tests/test_slurm_kit_pin.py)
case "$(python3 -c 'import candi; print(candi.__file__)')" in
  "$KIT"/src/*) ;;
  *) echo "candi does not import from $KIT/src" >&2; exit 3 ;;
esac

python3 "$KIT/tools/t118/ladder/aggregate.py" "$MANIFEST" "$COVARIATES" "$RUNS" "$REFS" "$AGG"
python3 "$KIT/tools/t118/ladder/aggregate.py" qm-curves "$MANIFEST" "$PRODUCTS" "$AGG/qm_curves.json"
python3 "$KIT/tools/t118/ladder/figures.py" "$AGG" "$RUNG" --refs-qm "$AGG/qm_curves.json"
python3 "$KIT/tools/t118/ladder/report.py" "$AGG" "$RUNG"

echo "=== done rung $RUNG $(date -u)"
ls -l "$AGG/$RUNG/report.md" "$AGG/checks_$RUNG.json"
