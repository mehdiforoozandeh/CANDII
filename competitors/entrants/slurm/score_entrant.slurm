#!/bin/bash
# Score one entrant's blind-test predictions: nine measures (minus msevar) plus the P-block.
# One array task per blind-test experiment, one submission per job submission.
#
#   sbatch --array=0-50 --export=ALL,LABEL=<team>,PREDDIR=<dir> slurm/score_entrant.slurm
#
# Set CHECK=1 to also re-run the vendored `fir_score.py` on the same inputs and require an exact
# match. That doubles the wall clock, so it is for the gate and for spot-checks, not the full grid.
#
# DNase-seq experiments exit 0 without scoring (decision B3): excluded from every Dataset-3 row.
#
#SBATCH --job-name=t54_score
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=slurm_logs/t54_score_%A_%a.out
#SBATCH --error=slurm_logs/t54_score_%A_%a.err

set -uo pipefail
REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_t54}"
ENT="$REPO/competitors/entrants"
DATA="${DATA:-/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE}"
M001="${M001:-/project/def-maxwl/mforooz/MAX_EPI_IMPUTATION/experiments/001_2026-07-27_eic_leaderboard_repro}"
WORK="${WORK:-/project/def-maxwl/mforooz/CANDII_t54_work}"
OUTDIR="${OUTDIR:-$WORK/entrant_scores/$LABEL}"

: "${LABEL:?set LABEL to the team name}"
: "${PREDDIR:?set PREDDIR to the directory of that team's C##M##.bigwig files}"

# shellcheck disable=SC1091
source "$ENT/env_fir.sh"

EXPLIST="${EXPLIST:-$M001/output/bridge/blind_experiments.txt}"
EXP=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$EXPLIST")
[ -z "$EXP" ] && { echo "no experiment at index $SLURM_ARRAY_TASK_ID"; exit 0; }

CSV="$OUTDIR/${LABEL}__${EXP}.csv"
JSON="$OUTDIR/${LABEL}__${EXP}.json"
if [ -s "$CSV" ] && [ -s "$JSON" ]; then echo "$CSV exists, skipping"; exit 0; fi
mkdir -p "$OUTDIR"

ARGS=(--experiment "$EXP"
      --bridge "$M001/output/bridge/eic_bridge.csv"
      --repo "$M001/output/inputs/imputation_challenge"
      --truth-dir "$DATA/blind_truth"
      --pred-dir "$PREDDIR"
      --label "$LABEL" --out-csv "$CSV" --out-json "$JSON")
[ "${CHECK:-0}" = "1" ] && ARGS+=(--check-against-fir-score)

python3 "$ENT/score_entrant.py" "${ARGS[@]}"
