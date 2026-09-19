#!/bin/bash
# t112, chunk C11: export the product tree and build + verify CANDI_STORE corpus `cf` on Nibi.
#
#   # step A, on the LOGIN node (it has internet; compute nodes do not). Download only, no venv:
#   bash "$KIT/slurm/t112/build_store.sh" --fetch-wheels <WHEELS_DIR>
#   # step B, the job. Every install in it is --no-index:
#   mkdir -p /scratch/mforooz/t112_cf/logs/verify /scratch/mforooz/t112_cf/checks
#   KIT=/scratch/mforooz/t112_cf/code/C15 WHEELS=<WHEELS_DIR> sbatch "$KIT/slurm/t112/build_store.sh"
#
# <WHEELS_DIR> is the caller's (e.g. a dir inside the kit snapshot); this script creates nothing else.
#
# Steps, each refusing to continue on a failure:
#   1. tools/t112/export_store.py: products + MANIFEST.tsv -> $SRC (npz source tree, cf_metadata.csv,
#      signal_provenance.cf.json) and $STORE/genome/chrom_sizes.json. It runs HERE, not on the login
#      node, because it needs numpy and the login node has none. It refuses a non-empty $SRC.
#   2. build-biosample --kinds counts,pval, one process per biosample, $SLURM_CPUS_PER_TASK at once.
#      It refuses an existing h5 (no --overwrite): a rebuild starts from an empty corpus dir.
#   3. build-manifest, strict (a CSV/file_metadata.json or signal-units conflict fails the build).
#   4. verify > $CF/checks/store_verify.txt; its first line must be "$CORPUS: OK".
#
# The venv: `import candi.store` runs candi/__init__.py, which imports torch, einops, einx, loguru
# and x_transformers, so numpy/pandas/h5py/torch alone do not import. The venv is the repo's own
# documented recipe from the kit snapshot: requirements-fir.txt from the CVMFS wheelhouse, then
# requirements-pypi.txt (x-transformers only). x-transformers is not in the wheelhouse, so step A
# fetches its pure-python wheel (py3-none-any) with the same module python, and step B installs it
# with --no-index --find-links. No step of the job needs internet.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t112_build_store
#SBATCH --output=/scratch/mforooz/t112_cf/logs/verify/%x_%j.out
#SBATCH --error=/scratch/mforooz/t112_cf/logs/verify/%x_%j.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -uo pipefail

PY_MODULE=python/3.10.13
XT_PIN=x-transformers==2.11.23      # must match requirements-pypi.txt

if [ "${1:-}" = "--fetch-wheels" ]; then
  DIR="${2:?usage: build_store.sh --fetch-wheels WHEELS_DIR}"
  module load "$PY_MODULE" || exit 4
  mkdir -p "$DIR" || exit 4
  python -m pip download --no-deps "$XT_PIN" -d "$DIR" || { echo "!!! pip download $XT_PIN failed"; exit 4; }
  WHL=$(ls "$DIR"/x_transformers-2.11.23-py3-none-any.whl 2>/dev/null)
  [ -n "$WHL" ] || { echo "!!! no x_transformers-2.11.23-py3-none-any.whl in $DIR:"; ls -l "$DIR"; exit 4; }
  echo "fetched: $WHL md5 $(md5sum "$WHL" | cut -d' ' -f1)"
  exit 0
fi

CF=/scratch/mforooz/t112_cf
KIT="${KIT:-$CF/code/C15}"
WHEELS="${WHEELS:?set WHEELS to the dir step A (--fetch-wheels) filled}"
PRODUCTS=$CF/products
MANIFEST=$PRODUCTS/MANIFEST.tsv
CHRSZ=/scratch/mforooz/EIC_REPRO/003/refcache/c52f52c7bfa357f55a39b1de7e4d0b0c/GRCh38_EBV.chrom.sizes.tsv
SRC=$CF/store_src
STORE=$CF/CANDI_STORE
CORPUS=$STORE/cf
GENOME_JSON=$STORE/genome/chrom_sizes.json
CHECKS=$CF/checks
NPROC="${SLURM_CPUS_PER_TASK:-4}"

die() { echo "!!! HALT: $1 $(date -u)"; exit "${2:-1}"; }

echo "=== job ${SLURM_JOB_ID:-none} on $(hostname) KIT=$KIT $(date -u) ==="
[ -f "$MANIFEST" ] || die "no $MANIFEST (records.py manifest first)" 2
[ -f "$KIT/tools/t112/export_store.py" ] || die "no kit snapshot at $KIT" 2
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || git -C "$KIT" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "manifest md5: $(md5sum "$MANIFEST" | cut -d' ' -f1)  rows: $(($(wc -l < "$MANIFEST") - 1))"
mkdir -p "$CHECKS" || die "mkdir $CHECKS" 3

# --- venv in $SLURM_TMPDIR ------------------------------------------------------------------------
ls "$WHEELS"/x_transformers-*-py3-none-any.whl >/dev/null 2>&1 \
  || die "no x_transformers wheel in WHEELS=$WHEELS; run step A (--fetch-wheels) on the login node" 4
grep -qx "$XT_PIN" "$KIT/requirements-pypi.txt" || die "$XT_PIN is not the pin in $KIT/requirements-pypi.txt" 4
module load "$PY_MODULE" || die "module load $PY_MODULE" 4
virtualenv --no-download "$SLURM_TMPDIR/venv" || die "virtualenv" 4
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$KIT/requirements-fir.txt" || die "pip install --no-index requirements-fir.txt" 4
pip install --no-index --find-links "$WHEELS" -r "$KIT/requirements-pypi.txt" \
  || die "pip install --no-index --find-links $WHEELS requirements-pypi.txt" 4
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$KIT/src"
python -c "import candi.store.cli, pandas" || die "candi.store does not import" 4

# --- 1. export ------------------------------------------------------------------------------------
echo "=== export $(date -u) ==="
python "$KIT/tools/t112/export_store.py" --products "$PRODUCTS" --manifest "$MANIFEST" \
  --source-root "$SRC" --chrsz "$CHRSZ" --genome-json "$GENOME_JSON" || die "export_store.py" 5

# --- 2. build-biosample ---------------------------------------------------------------------------
tail -n +2 "$MANIFEST" | cut -f2 | sort -u > "$SLURM_TMPDIR/biosamples.txt"
echo "=== build-biosample: $(wc -l < "$SLURM_TMPDIR/biosamples.txt") biosamples, $NPROC at once $(date -u) ==="
xargs -a "$SLURM_TMPDIR/biosamples.txt" -P "$NPROC" -I{} \
  python -m candi.store build-biosample --source-root "$SRC" --corpus-root "$CORPUS" \
    --chrom-sizes "$GENOME_JSON" --kinds counts,pval --biosample {} \
  || die "build-biosample failed for at least one biosample (see lines above)" 6

# --- 3. build-manifest ----------------------------------------------------------------------------
echo "=== build-manifest $(date -u) ==="
python -m candi.store build-manifest --corpus-root "$CORPUS" --corpus cf \
  --metadata-csv "$SRC/cf_metadata.csv" --source-root "$SRC" \
  --signal-provenance "$SRC/signal_provenance.cf.json" || die "build-manifest" 7

# --- 4. verify ------------------------------------------------------------------------------------
echo "=== verify $(date -u) ==="
python -m candi.store verify --corpus-root "$CORPUS" > "$CHECKS/store_verify.txt"
RC=$?
cat "$CHECKS/store_verify.txt"
md5sum "$CORPUS/manifest.json"
du -sh "$SRC" "$CORPUS"
echo "=== DONE verify rc=$RC $(date -u) ==="
exit $RC
