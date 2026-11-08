#!/bin/bash
# t146 (h31): build the per-25 bp mappability channel, the DNA codes and the fixed mappability
# decile strata once (tools/t146/features.py), from t138's annotation files (the same Umap k36
# bedGraph and hg38 fasta t138/t141 used), cross-checked against t138's 1 kb map_<c>.
#
#   python3 $KIT/tools/t146/features.py build <cache_dir> <manifest> <annot_dir> <blacklist> \
#       <out_dir> --t138-features <t138_out>/features.npz     -> FEATURES_DONE, strata.json
#
# NO --export; positional only. CPU, 1 core, 48 GB, 6 h. The out dir must be t146 or under it.
#   sbatch --parsable --output=$OUT/logs/%x_%j.out $KIT/slurm/t146/features.sh $KIT $PRODUCTS \
#       $CACHE /project/def-maxwl/mforooz/t138/annot $BLACKLIST $OUT/features \
#       /project/def-maxwl/mforooz/t138/out/features.npz
#SBATCH --account=def-maxwl
#SBATCH --job-name=t146_features
#SBATCH --exclude=c128,c166,c537,c371,c329
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=48000M
#SBATCH --gres=none

set -euo pipefail
USAGE="usage: t146/features.sh <kit_dir> <products_dir> <cache_dir> <annot_dir> <blacklist.bed> <out_dir> [t138_features.npz]"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
CACHE="${3:?$USAGE}"
ANNOT="${4:?$USAGE}"
BLACKLIST="${5:?$USAGE}"
OUT="${6:?$USAGE}"
T138F="${7:-}"
case "$OUT" in
  */t146|*/t146/*) ;;
  *) echo "out dir $OUT is not t146 or under it" >&2; exit 2 ;;
esac
CMD=(python3 "$KIT/tools/t146/features.py" build "$CACHE" "$PRODUCTS/MANIFEST.tsv" "$ANNOT"
     "$BLACKLIST" "$OUT")
[ -z "$T138F" ] || CMD+=(--t138-features "$T138F")
if [ "${DRY_RUN:-0}" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
echo "=== t146 features job ${SLURM_JOB_ID:-local} host=$(hostname) $(date -u)"
set +u; module load python/3.10.13; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$KIT/requirements-fir.txt"
pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
export PYTHONPATH="$KIT/src"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
"${CMD[@]}"
echo "=== done $(date -u)"
