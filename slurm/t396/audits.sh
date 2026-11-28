#!/bin/bash
# t396 C5: the table audits (h61 PE depth, tagAlign fragment counts, correction-pair census).
#
#   sbatch --job-name=sc_audits --output=<out>/%x_%j.out <kit>/slurm/t396/audits.sh <kit dir> <out dir>
#
# Writes <out>/pe_depth/{rows,summary,listed_rows,tag_fragments}.tsv and
# <out>/pair_census/{census,by_covariate,candidates}.tsv. Each step skips when its output exists.
# Reads only: the eic_v1_2 tables, the evaluation set, the read-start store, the read-count store
# and the rs392 tagAlign tars (streamed, nothing extracted to disk).
#SBATCH --account=def-maxwl
#SBATCH --job-name=sc_audits
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --gres=none
#SBATCH --exclude=fc10713,fc10715

set -euo pipefail
USAGE="usage: audits.sh <kit dir> <out dir>"
KIT="${1:?$USAGE}"
OUT="${2:?$USAGE}"
VENV=/project/def-maxwl/mforooz/candi_venv
TABLES=/project/def-maxwl/mforooz/ONION/tables/eic_v1_2
EVALSET=/project/def-maxwl/mforooz/ONION/eval/eic_v1_2/evalset_p16.json
RS_STORE=/scratch/mforooz/rs392/store
TARS=/scratch/mforooz/rs392/tars
MEMBERS=/scratch/mforooz/rs392/plan
KIT=$(cd "$KIT" && pwd)
echo "=== sc_audits job ${SLURM_JOB_ID:-none} host=$(hostname) kit=$KIT $(date -u)"
cat "$KIT/.kit_stamp" 2>/dev/null || true
cd "$KIT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
source "$VENV/bin/activate"
source "$KIT/slurm/_kit_pin.sh"
A="python $KIT/tools/t396/audits.py"
mkdir -p "$OUT/pe_depth" "$OUT/pair_census"

[ -s "$OUT/pair_census/candidates.tsv" ] ||
  $A pair-census --tables "$TABLES" --evalset "$EVALSET" --rs-store "$RS_STORE" --out "$OUT/pair_census"
[ -s "$OUT/pe_depth/listed_rows.tsv" ] ||
  $A pe-depth --tables "$TABLES" --rs-store "$RS_STORE" --out "$OUT/pe_depth"
[ -s "$OUT/pe_depth/tag_fragments.tsv" ] ||
  $A tag-fragments --tables "$TABLES" --rs-store "$RS_STORE" --tars "$TARS" --members "$MEMBERS" \
    --per-class 2 --tmp "${SLURM_TMPDIR:-/tmp}" --out "$OUT/pe_depth"
echo "=== done $(date -u)"
