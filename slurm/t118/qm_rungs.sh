#!/bin/bash
# t118: the two baseline rungs (noSolution, QuantileMatching) for one base<->arm pair per array task.
#
# WHAT ONE TASK DOES. `tools/t118/baseline_rungs.py run --index $SLURM_ARRAY_TASK_ID` fits both
# rungs on the training chromosomes of one (arm product, direction), in counts and in -log10 p,
# scores them on chr19 + chr21 (blacklist removed) and on chr22 (reported only), and writes
# <out>/<arm_pid>__<direction>.json. A task whose JSON already exists exits 0 without work, so a
# resubmission redoes only the missing pairs.
#
# THE PAIR TABLE. Its row order is a pure function of MANIFEST.tsv (sorted by track, pid,
# direction). Print it and take the array size from it; 130 products = 7 bases + 123 arms gives
# 246 pairs, i.e. --array=0-245:
#   python3 $KIT/tools/t118/baseline_rungs.py pairs --manifest $PRODUCTS/MANIFEST.tsv
#
# NO --export. Alliance arrays truncate a comma-valued --export variable, so every input is a
# POSITIONAL argument.
#
# THE VENV. `import candi.metrics` runs candi/__init__.py, which imports torch, einops, einx, loguru
# and x_transformers (eagerly, through candi.encoder), so the venv is the repo's own pinned recipe,
# as slurm/t112/build_store.sh builds it: python/3.10.13, requirements-fir.txt from the CVMFS
# wheelhouse, then requirements-pypi.txt (x-transformers only) from $KIT/wheels with --no-index
# --find-links (x-transformers is not in the wheelhouse; compute nodes have no internet). Built per
# task in $SLURM_TMPDIR, never on a shared filesystem.
#
# RESOURCES. One core. Measured locally (M-series laptop) on a synthetic pair with full-length
# chr19/21/22 and 17.6 M training bins (1/6.7 of the real 117.6 M): 135 s, peak RSS 2.6 GB. The
# p-space fit (distinct-value histograms, worst case every bin distinct) was 45 s of it and scales
# with training bins; scoring (~75 s) does not. Extrapolated per real pair: ~8-12 min, peak RSS
# ~3-6 GB. 3 h keeps the job in the b1 bin; 16G covers the worst-case distinct-value merge.
#
# Usage, from the Nibi login node, after snapshotting the repo to $KIT:
#   OUT=<results dir>; mkdir -p $OUT/logs   # SLURM opens --output before the body runs
#   sbatch --test-only --array=0-245%40 --output=$OUT/logs/%x_%A_%a.out \
#       $KIT/slurm/t118/qm_rungs.sh $KIT $PRODUCTS $BLACKLIST $OUT
#   sbatch --parsable  --array=0-245%40 --output=$OUT/logs/%x_%A_%a.out \
#       $KIT/slurm/t118/qm_rungs.sh $KIT $PRODUCTS $BLACKLIST $OUT
#   # when all tasks are done (a login-node python has no numpy, so run it in a job or salloc):
#   python3 $KIT/tools/t118/baseline_rungs.py aggregate --manifest $PRODUCTS/MANIFEST.tsv --out $OUT
#SBATCH --account=def-maxwl
#SBATCH --job-name=qm_rungs
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

set -euo pipefail

USAGE="usage: qm_rungs.sh <kit_dir> <products_dir> <blacklist.bed> <out_dir>"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
BLACKLIST="${3:?$USAGE}"
OUT="${4:?$USAGE}"
PY_MODULES="python/3.10.13"
: "${SLURM_ARRAY_TASK_ID:?array task only}"

[ -f "$KIT/tools/t118/baseline_rungs.py" ] || { echo "no tools/t118/baseline_rungs.py under $KIT" >&2; exit 2; }
[ -f "$PRODUCTS/MANIFEST.tsv" ] || { echo "no $PRODUCTS/MANIFEST.tsv" >&2; exit 2; }
[ -s "$BLACKLIST" ] || { echo "blacklist $BLACKLIST missing or empty" >&2; exit 2; }
KIT=$(cd "$KIT" && pwd)
mkdir -p "$OUT"

echo "=== t118 qm_rungs idx=$SLURM_ARRAY_TASK_ID job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} host=$(hostname) $(date -u)"
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || git -C "$KIT" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "blacklist sha256: $(sha256sum "$BLACKLIST" | cut -d' ' -f1)"

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

python3 "$KIT/tools/t118/baseline_rungs.py" run   # no /usr/bin/time on Nibi compute nodes; peak RSS is in sacct MaxRSS \
    --manifest "$PRODUCTS/MANIFEST.tsv" --products "$PRODUCTS" --blacklist "$BLACKLIST" \
    --out "$OUT" --index "$SLURM_ARRAY_TASK_ID"

echo "=== done $(date -u)"
