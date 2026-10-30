#!/bin/bash
# t138: the regional map on Nibi. Step 1 builds the per-1 kb-bin features (GC, mappability, repeat
# fraction, peak-vs-domain context per track) unless <out_dir>/features.npz exists; step 2 fits
# signal level as the null on each product's split-half divergence and reads the residual against
# the features. numpy + scipy + pandas only: the scipy-stack module, no venv.
#
#   1. python3 $KIT/tools/t138/features.py build <annot_dir> <narrowpeak_dir> <bins1k_dir> \
#          <out_dir>/features.npz                       -> features.npz, features.json
#   2. python3 $KIT/tools/t138/map.py run <bins1k_dir> <out_dir>/features.npz \
#          <products_dir>/MANIFEST.tsv <out_dir>        -> map_*.tsv, map_check.txt, config.json
#
# NO --export; positional arguments only. The out dir must be t138 or under it. DRY_RUN=1 prints
# the two python lines and exits 0 before any module load. SLURM does not create the --output
# directory: the submitter runs `mkdir -p /project/def-maxwl/mforooz/t138/logs` first.
#   sbatch --parsable --output=/project/def-maxwl/mforooz/t138/logs/%x_%j.out \
#       $KIT/slurm/t138/map.sh $KIT $ANNOT $NARROWPEAK $BINS1K $PRODUCTS $OUT
#SBATCH --account=def-maxwl
#SBATCH --job-name=t138_map
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=none
#SBATCH --exclude=c128,c166,c537,c371,c329

set -euo pipefail
USAGE="usage: t138/map.sh <kit_dir> <annot_dir> <narrowpeak_dir> <bins1k_dir> <products_dir> <out_dir>"
KIT="${1:?$USAGE}"
ANN="${2:?$USAGE}"
NP="${3:?$USAGE}"
BINS="${4:?$USAGE}"
PRODUCTS="${5:?$USAGE}"
OUT="${6:?$USAGE}"
case "$OUT" in
  */t138|*/t138/*) ;;
  *) echo "out dir $OUT is not t138 or under it; t138 writes only there" >&2; exit 2 ;;
esac
FEAT="$OUT/features.npz"
L1="python3 $KIT/tools/t138/features.py build $ANN $NP $BINS $FEAT"
L2="python3 $KIT/tools/t138/map.py run $BINS $FEAT $PRODUCTS/MANIFEST.tsv $OUT"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "$L1"
  echo "$L2"
  exit 0
fi
echo "=== t138 map job ${SLURM_JOB_ID:-local} host=$(hostname) $(date -u)"
GIT_SHA="$(cat "$KIT/GIT_SHA" 2>/dev/null || echo unknown)"
export GIT_SHA
echo "kit git sha: $GIT_SHA"
set +u; module load scipy-stack; set -u
mkdir -p "$OUT"
# features.py writes $FEAT.tmp and renames it on success, so $FEAT exists only when complete; a
# killed step 1 leaves a stale .tmp, removed here before the rebuild.
if [[ -s "$FEAT" ]]; then
  echo "have $FEAT; step 1 skipped"
else
  rm -f "$FEAT.tmp"
  python3 "$KIT/tools/t138/features.py" build "$ANN" "$NP" "$BINS" "$FEAT"
fi
python3 "$KIT/tools/t138/map.py" run "$BINS" "$FEAT" "$PRODUCTS/MANIFEST.tsv" "$OUT"
echo "=== done $(date -u)"
