#!/bin/bash
# t141 (q11): read every arm's one-warp residual against the t138 features (GC, mappability,
# repeats, peak-vs-domain context) with signal level fitted first as the null, beside the t138
# pseudoreplicate floor map at matched depth. One job, after every arms.sh task is done.
# numpy + scipy only: the scipy-stack module, no venv.
#
#   python3 $KIT/tools/t141/map_knobs.py run <out_dir>/arms1k <features.npz> \
#       <products_dir>/MANIFEST.tsv <t138_out_dir> <out_dir>/out
#       -> knob_products.tsv knob_classes.tsv knob_context.tsv knob_deciles.tsv config.json
#
# NO --export; positional arguments only. The out dir must be t141 or under it. DRY_RUN=1 prints
# the python line and exits 0 before any module load. SLURM does not create the --output
# directory: mkdir -p /project/def-maxwl/mforooz/t141/logs first.
#   sbatch --parsable --output=/project/def-maxwl/mforooz/t141/logs/%x_%j.out \
#       $KIT/slurm/t141/map_knobs.sh $KIT $PRODUCTS /project/def-maxwl/mforooz/t138/out \
#       /project/def-maxwl/mforooz/t141
#SBATCH --account=def-maxwl
#SBATCH --job-name=t141_map_knobs
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=none
#SBATCH --exclude=c128,c166,c537,c371,c329

set -euo pipefail
USAGE="usage: t141/map_knobs.sh <kit_dir> <products_dir> <t138_out_dir> <out_dir>"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
T138="${3:?$USAGE}"
OUT="${4:?$USAGE}"
case "$OUT" in
  */t141|*/t141/*) ;;
  *) echo "out dir $OUT is not t141 or under it; t141 writes only there" >&2; exit 2 ;;
esac
L1="python3 $KIT/tools/t141/map_knobs.py run $OUT/arms1k $T138/features.npz $PRODUCTS/MANIFEST.tsv $T138 $OUT/out"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "$L1"
  exit 0
fi
[ -s "$T138/features.npz" ] || { echo "no $T138/features.npz (the t138 features build)" >&2; exit 2; }
[ -f "$PRODUCTS/MANIFEST.tsv" ] || { echo "no $PRODUCTS/MANIFEST.tsv" >&2; exit 2; }
echo "=== t141 map_knobs job ${SLURM_JOB_ID:-local} host=$(hostname) $(date -u)"
GIT_SHA="$(cat "$KIT/GIT_SHA" 2>/dev/null || echo unknown)"
export GIT_SHA
echo "kit git sha: $GIT_SHA"
[ -f "$T138/map_classes.tsv" ] || echo "WARN: no $T138/map_classes.tsv; floor columns will be NA"
set +u; module load scipy-stack; set -u
mkdir -p "$OUT/out"
python3 "$KIT/tools/t141/map_knobs.py" run "$OUT/arms1k" "$T138/features.npz" \
  "$PRODUCTS/MANIFEST.tsv" "$T138" "$OUT/out"
echo "=== done $(date -u)"
