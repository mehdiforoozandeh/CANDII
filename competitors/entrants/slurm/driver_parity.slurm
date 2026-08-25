#!/bin/bash
# Prove `score_entrant.py` scores identically to the vendored `fir_score.py`, before any entrant
# submission exists to score.
#
# The trick is that a "prediction" only has to be a bigwig on the official grid. So a DIFFERENT
# cell's track of the SAME assay stands in for one: it is a real, non-degenerate comparison against
# the blind truth -- the two cells genuinely differ -- and both files are already on disk. Scoring a
# track against itself would give mse 0 and gwcorr 1 and would prove nothing about the arithmetic.
#
# `--check-against-fir-score` makes the driver re-run vendor/fir_score.py in a subprocess on the same
# inputs and exit non-zero on any difference at all, exact rather than approximate: both paths call
# the same eic_metrics.score on arrays from the same loader, so a difference means the adaptation
# changed an input.
#
#   sbatch --array=0-2 slurm/driver_parity.slurm
#
#SBATCH --job-name=t54_parity
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=slurm_logs/t54_parity_%A_%a.out
#SBATCH --error=slurm_logs/t54_parity_%A_%a.err

set -uo pipefail
REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_t54}"
ENT="$REPO/competitors/entrants"
DATA="${DATA:-/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE}"
M001="${M001:-/project/def-maxwl/mforooz/MAX_EPI_IMPUTATION/experiments/001_2026-07-27_eic_leaderboard_repro}"
WORK="${WORK:-/project/def-maxwl/mforooz/CANDII_t54_work}"
OUTDIR="$WORK/parity"

# shellcheck disable=SC1091
source "$ENT/env_fir.sh"

# (target experiment, stand-in track of the same assay). Both sides must be BLIND-split experiments,
# since only those have a bigwig under blind_truth/.
#
#   C05M18 <- C06M18   H3K36me3, broad
#   C19M22 <- C28M22   H3K4me3, punctate -- the ONLY assay where prom_corr_h3k4me3 fires, so this is
#                      the task that exercises the P-block's promoter-window path
#   C12M01 <- C31M01   ATAC-seq, accessibility
#
# Three dynamic ranges, so the parity claim does not rest on one mark.
TARGETS=(C05M18 C19M22 C12M01)
STANDINS=(C06M18 C28M22 C31M01)
EXP=${TARGETS[$SLURM_ARRAY_TASK_ID]}
STAND=${STANDINS[$SLURM_ARRAY_TASK_ID]}

FARM="$SLURM_TMPDIR/pred"
mkdir -p "$FARM" "$OUTDIR"
# the stand-in's bytes, presented under the target's name -- which is the filename the scorer looks
# for. Nothing is copied and nothing on /project is touched.
ln -sfn "$DATA/blind_truth/$STAND.bigwig" "$FARM/$EXP.bigwig"
echo "[parity] target=$EXP prediction=$STAND" >&2

python3 "$ENT/score_entrant.py" \
    --experiment "$EXP" \
    --bridge "$M001/output/bridge/eic_bridge.csv" \
    --repo "$M001/output/inputs/imputation_challenge" \
    --truth-dir "$DATA/blind_truth" \
    --pred-dir "$FARM" \
    --label "parity_${STAND}_as_${EXP}" \
    --out-csv "$OUTDIR/parity__$EXP.csv" \
    --out-json "$OUTDIR/parity__$EXP.json" \
    --allow-dnase \
    --check-against-fir-score
