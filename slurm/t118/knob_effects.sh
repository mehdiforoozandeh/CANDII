#!/bin/bash
# t133: rank the t112 processing knobs by how much each moves the signal (exploratory).
# One plain job over all 130 products; reads chr19 + chr21 only. Needs numpy + scipy only, so the
# scipy-stack module, not the candi venv.
#
#   OUT=<results dir>; mkdir -p $OUT
#   sbatch --output=$OUT/%x_%j.out $KIT/slurm/t118/knob_effects.sh $KIT $PRODUCTS $BLACKLIST $OUT [peakset]
# peakset: base (default, the t133 rule) or shared (t139: one peak set per track and space).
#SBATCH --account=def-maxwl
#SBATCH --job-name=knob_effects
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
USAGE="usage: knob_effects.sh <kit_dir> <products_dir> <blacklist.bed> <out_dir> [base|shared]"
KIT="${1:?$USAGE}"; PRODUCTS="${2:?$USAGE}"; BLACKLIST="${3:?$USAGE}"; OUT="${4:?$USAGE}"
PEAKSET="${5:-base}"
echo "=== t133 knob_effects (peakset $PEAKSET) job $SLURM_JOB_ID host=$(hostname) $(date -u)"
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || echo unknown)"
set +u; module load scipy-stack; set -u
python3 "$KIT/tools/t118/knob_effects.py" --products "$PRODUCTS" --blacklist "$BLACKLIST" --out "$OUT" \
    --peakset "$PEAKSET"
echo "=== done $(date -u)"
