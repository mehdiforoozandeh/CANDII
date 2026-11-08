#!/bin/bash
# t150: the depth-only baseline (counts x d'/d, one fitted NB size per pair) and the identity
# reference, scored on the final CAC testbed's direct pairs exactly as score.sh scores a run
# (tools/t146/depth_baseline.py). One CPU job, no array, no GPU.
#
#   sbatch --output=$OUT/logs/%x_%j.out $KIT/slurm/t146/depth_baseline.sh \
#       $KIT $PRODUCTS $CACHE $OUT $FEATURES $BLACKLIST [law_stride]
#   -> $OUT/baselines/depth_only/scores.json, SCORE_DONE (exists: exit 0)
# Paths as slurm/t146/score.sh. 32 cores x 4000 MB (Nibi/Fir CPU rule).
#SBATCH --account=def-maxwl
#SBATCH --job-name=t150_depth_baseline
#SBATCH --exclude=c128,c166,c537,c371,c329
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000M
#SBATCH --gres=none

set -euo pipefail
BLACKLIST_SHA256_PREFIX=31c69342
USAGE="usage: t146/depth_baseline.sh <kit_dir> <products_dir> <cache_dir> <out_dir> <features_dir> <blacklist.bed> [law_stride]"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
CACHE="${3:?$USAGE}"
OUT="${4:?$USAGE}"
FEATURES="${5:?$USAGE}"
BLACKLIST="${6:?$USAGE}"
LAW_STRIDE="${7:-1}"
[[ "$LAW_STRIDE" =~ ^[0-9]+$ ]] || { echo "law_stride must be an integer" >&2; exit 2; }
case "$OUT" in
  */t146|*/t146/*) ;;
  *) echo "out dir $OUT is not t146 or under it" >&2; exit 2 ;;
esac
RD="$OUT/baselines/depth_only"
[ "$LAW_STRIDE" = 1 ] || RD="$RD.stride$LAW_STRIDE"
CMD=(python3 "$KIT/tools/t146/depth_baseline.py" run "$PRODUCTS/MANIFEST.tsv" "$CACHE" "$BLACKLIST"
     "$RD" --features "$FEATURES" --workers "${SLURM_CPUS_PER_TASK:-8}" --law-stride "$LAW_STRIDE")
if [ "${DRY_RUN:-0}" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
if [ -f "$RD/SCORE_DONE" ]; then echo "skip: $RD/SCORE_DONE exists"; exit 0; fi
case "$(sha256sum "$BLACKLIST" | cut -d' ' -f1)" in
  "$BLACKLIST_SHA256_PREFIX"*) ;;
  *) echo "blacklist sha256 does not start $BLACKLIST_SHA256_PREFIX" >&2; exit 2 ;;
esac
echo "=== t150 depth baseline job $SLURM_JOB_ID host=$(hostname) $(date -u)"
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
