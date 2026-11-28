#!/bin/bash
# t396 it2 B5 job script (kit SC2B5, Fir, numpy only): fit, floor-count and score of the testbed
# controlled pairs for one tool and one rate scenario, in that order, each step skipped when its
# output exists.
#
#   TOOL = dedup | sepe (covariate dedup | run_type); task i = SLURM_ARRAY_TASK_ID picks the
#   scenario: i = 0 -> `fitted` (rows.tsv of B3 as is: q_target measured, q_source fitted, D25);
#   i >= 1 -> `rho<x>`: q_source fixed to lines_source / (x * total_target), i.e. the source arm's
#   pre-subsample total is x times the target's (q_origin `fixed`). Grids are below.
#
# Submit (B5): export KIT=/project/def-maxwl/mforooz/onion_kit/SC2B5 TOOL=dedup
#   sbatch --job-name=sc_dedup --export=ALL --array=0-6%12 \
#     --output=/scratch/mforooz/sc396/it2/cf_caps/logs/%x_%A_%a.out "$KIT/cruxvault/results/t396/cf_caps/fir_job.sh"
# Outputs: $OUTROOT/<tool>/<scenario>/{rows.tsv,per_assay.tsv,floor/<target>.npz,pairs.tsv,table.tsv}
#SBATCH --account=def-maxwl
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --gres=none
#SBATCH --exclude=fc10713,fc10715

set -uo pipefail
: "${KIT:?export KIT}"
: "${TOOL:?export TOOL=dedup|sepe}"
I="${SLURM_ARRAY_TASK_ID:-0}"
OUTROOT="${OUTROOT:-/project/def-maxwl/mforooz/t396/cf_caps}"
STATS=/scratch/mforooz/sc396/it2/cf_caps/$TOOL
HERE="$KIT/cruxvault/results/t396/cf_caps"
RULE="$KIT/cruxvault/results/t396/dedup_1bp/UNIT_RULE.json"
PY=/project/def-maxwl/mforooz/candi_venv/bin/python
# rho = source pre-subsample lines / target pre-subsample lines.
# dedup: >= 1 (dedup-off holds every dedup-on read); 1.075 = 1 / 0.929, the iteration-1 median of
#   reads_on_over_off (dedup_cap/totals.tsv, 355 pairs); 6.69 = 1 / 0.1495, its minimum.
# sepe: PE lines / SE reads; 1.89 .. 2.82 = 1 / reads_se_over_parent on the 12 iteration-1 PE
#   parents (sepe_cap/rows.tsv); 2 = both arms keep every fragment.
case "$TOOL" in
dedup) COV=dedup;    GRID=(fitted 1.0 1.075 1.25 1.5 2.0 6.69) ;;
sepe)  COV=run_type; GRID=(fitted 1.5 1.89 2.0 2.4 2.82 4.0) ;;
*) echo "unknown TOOL $TOOL" >&2; exit 2 ;;
esac
RHO="${GRID[$I]:?no scenario $I}"
SCEN=$([ "$RHO" = fitted ] && echo fitted || echo "rho$RHO")
O="$OUTROOT/$TOOL/$SCEN"
echo "=== sc396 B5 $TOOL $SCEN job ${SLURM_JOB_ID:-none} task $I host=$(hostname) $(date -u)"
cd "$KIT" || exit 2
source "$(dirname "$PY")/activate"
source "$KIT/slurm/_kit_pin.sh" || exit 1
mkdir -p "$O/floor"

# 1. rows of this scenario
if [ ! -s "$O/rows.tsv" ]; then
  awk -F'\t' -v OFS='\t' -v rho="$RHO" -v cov="$COV" '
    NR == 1 {for (k = 1; k <= NF; k++) c[$k] = k; print; next}
    rho != "fitted" && $c["covariate"] == cov {
      $c["q_source"] = sprintf("%.8g", $c["lines_source"] / (rho * $c["total_target"]))
      $c["total_source"] = sprintf("%.0f", rho * $c["total_target"]); $c["q_origin"] = "fixed" }
    {print}' "$HERE/rows.tsv" > "$O/rows.tsv" || exit 1
fi
T="$PY tools/t396/${TOOL}_cap.py"
rc=0
# 2. fit: rates (fitted only when missing) and the leave-one-track-out a and kappa
[ -s "$O/per_assay.tsv" ] || /usr/bin/time -v $T fit --stats "$STATS"/*.npz --rows "$O/rows.tsv" \
  --out "$O/per_assay.tsv" || exit 1
# 3. floor-count of every pair of the covariate
while read -r src tgt; do
  f="$O/floor/$tgt.npz"
  [ -s "$f" ] && continue
  /usr/bin/time -v $T floor-count --pair "$src" "$tgt" --unit-rule "$RULE" --fit "$O/per_assay.tsv" \
    --rows "$O/rows.tsv" --out "$f" || { echo "FAILED floor $tgt" >&2; rc=1; }
done < <(awk -F'\t' 'NR == 1 {for (k = 1; k <= NF; k++) c[$k] = k; next}
  $c["covariate"] == cov && $c["in_store"] == "1" {print $c["source_id"], $c["target_id"]}' cov="$COV" "$O/rows.tsv")
[ "$rc" = 0 ] || exit 1
# 4. score
[ -s "$O/table.tsv" ] || /usr/bin/time -v $T score --stats "$STATS"/*.npz --fit "$O/per_assay.tsv" \
  --rows "$O/rows.tsv" --floor-count "$O"/floor/*.npz --out "$O" || exit 1
echo "=== done $(date -u)"
