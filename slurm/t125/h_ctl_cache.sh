#!/bin/bash
# t125 part 2: build the CONTROL count cache that h reads. One array task = one product of the
# manifest (index over the rows sorted by pid, 0..129, as `ladder.train cache`). A product with a
# control (114 of 130) gets <ctl_cache>/<pid>__counts.npy + .json in the ladder.data cache format
# (space "counts", so ladder.data.Corpus(<ctl_cache>).get(pid, "counts", chrom) returns the CONTROL
# counts), md5-checked against the manifest's control_counts25_md5; a product without one is
# "absent" and writes nothing.
#
# WHAT ONE TASK DOES.
#   python3 $KIT/tools/t125/hmap/controls.py cache <manifest> <products_dir> <ctl_cache> <index>
# controls.py itself skips a product whose cache files already exist ("skipped"), so a
# resubmission of the whole array redoes only the missing products.
#
# NO --export. Every input is a POSITIONAL argument; the task index is $SLURM_ARRAY_TASK_ID.
# Every output is under <ctl_cache>:
#   CTL=/project/def-maxwl/mforooz/t125/ctl_cache
#   OUT_H=/project/def-maxwl/mforooz/t125/h          (logs under $OUT_H/logs)
# A <ctl_cache> under a t118 directory is refused (exit 2): t118's count cache holds TREATMENT
# counts under the same file names.
#
# THE VENV is the pinned recipe of slurm/t118/row2_train.sh (python/3.10.13, requirements-fir.txt,
# then requirements-pypi.txt from $KIT/wheels), built per task in $SLURM_TMPDIR; PYTHONPATH is
# "$KIT/src" and the t125 CLIs put tools/t118 and tools/t125 on sys.path themselves.
#
# RESOURCES. CPU only, no GPU request. 2 cores, 8 GB, 30 min (Nibi's default walltime is 15 min, so
# the time is always set). Nodes c128, c166, c537 and c371 are excluded (CVMFS I/O errors,
# plan/T118_STATUS.md). Peak memory from `sacct -o MaxRSS` (no /usr/bin/time on compute nodes).
#
# DRY_RUN=1 prints the python command line and exits 0 before any module load. It checks the index
# against the row count of <products_dir>/MANIFEST.tsv when it exists, else of this checkout's
# tests/fixtures/t112_meta/MANIFEST.tsv (the same file, md5-pinned); an index outside it exits 2.
#
# Usage, from the Nibi login node, after snapshotting the repo to $KIT (kit K9 and up). Record the
# job id in plan/T125_STATUS.md:
#   CTL=/project/def-maxwl/mforooz/t125/ctl_cache; OUT_H=/project/def-maxwl/mforooz/t125/h
#   mkdir -p $CTL $OUT_H/logs                     # SLURM opens --output first
#   sbatch --test-only --array=0-129%20 --exclude=c128,c166,c537,c371 --output=$OUT_H/logs/%x_%A_%a.out \
#       $KIT/slurm/t125/h_ctl_cache.sh $KIT $PRODUCTS $CTL
#   sbatch --parsable --array=0-129%20 --exclude=c128,c166,c537,c371 --output=$OUT_H/logs/%x_%A_%a.out \
#       $KIT/slurm/t125/h_ctl_cache.sh $KIT $PRODUCTS $CTL
# Check: `ls $CTL/*__counts.json | wc -l` -> 114. Never chain a job onto this array: verify the
# cache by its files.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t125_hctl
#SBATCH --exclude=c128,c166,c537,c371
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M

set -euo pipefail

USAGE="usage: h_ctl_cache.sh <kit_dir> <products_dir> <ctl_cache>"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
CTL="${3:?$USAGE}"
IDX="${SLURM_ARRAY_TASK_ID:?array task only}"
[[ "$IDX" =~ ^[0-9]+$ ]] || { echo "SLURM_ARRAY_TASK_ID '$IDX' is not an index" >&2; exit 2; }
case "$CTL" in
  */t118|*/t118/*) echo "control cache $CTL is under a t118 directory; t125 writes only under t125/" >&2; exit 2 ;;
esac
PY_MODULES="python/3.10.13"
MANIFEST="$PRODUCTS/MANIFEST.tsv"

# The number of products in a manifest (data rows after the header).
n_products() { awk 'NR > 1 && NF > 0 { n++ } END { print n + 0 }' "$1"; }

if [ "${DRY_RUN:-0}" = 1 ]; then
  TOOLS="$KIT"
  [ -f "$TOOLS/tools/t125/hmap/controls.py" ] || TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  TABLE_MANIFEST="$MANIFEST"
  [ -f "$TABLE_MANIFEST" ] || TABLE_MANIFEST="$TOOLS/tests/fixtures/t112_meta/MANIFEST.tsv"
  N=$(n_products "$TABLE_MANIFEST")
  [ "$IDX" -lt "$N" ] || { echo "index $IDX outside 0..$((N - 1)) of $TABLE_MANIFEST" >&2; exit 2; }
  echo "python3 $KIT/tools/t125/hmap/controls.py cache $MANIFEST $PRODUCTS $CTL $IDX"
  exit 0
fi

[ -f "$KIT/tools/t125/hmap/controls.py" ] || { echo "no tools/t125/hmap/controls.py under $KIT" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "no $MANIFEST" >&2; exit 2; }
[ -d "$PRODUCTS" ] || { echo "no products dir $PRODUCTS" >&2; exit 2; }
N=$(n_products "$MANIFEST")
[ "$IDX" -lt "$N" ] || { echo "index $IDX outside 0..$((N - 1)) of $MANIFEST" >&2; exit 2; }
KIT=$(cd "$KIT" && pwd)
mkdir -p "$CTL"

echo "=== t125 h_ctl_cache idx=$IDX job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX host=$(hostname) $(date -u)"
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

python3 "$KIT/tools/t125/hmap/controls.py" cache "$MANIFEST" "$PRODUCTS" "$CTL" "$IDX"

echo "=== done idx=$IDX $(date -u)"
