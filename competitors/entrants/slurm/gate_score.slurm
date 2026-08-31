#!/bin/bash
# The Average validation gate: score the rebuilt Average against the Dataset-3 blind truth with the
# VENDORED scorer, unmodified. One array task per blind-test experiment (51).
#
# This is the run that proves the whole path before a single entrant is scored. It uses
# `vendor/fir_score.py` as a CLI rather than any adapter of ours, so a match against 001's recorded
# `our_average_tv.csv` is evidence about the path and the data staging, with our code contributing
# nothing but the argument list.
#
# `--var-source none` because msevar does not reproduce (001: no candidate variance vector gets
# closer than median ratio 0.19) and both 001 and 005 exclude it everywhere.
#
#   sbatch --array=0-50 slurm/gate_score.slurm
#
#SBATCH --job-name=t54_gate
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=6
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=slurm_logs/t54_gate_%A_%a.out
#SBATCH --error=slurm_logs/t54_gate_%A_%a.err

set -uo pipefail
REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_t54}"
ENT="$REPO/competitors/entrants"
DATA="${DATA:-/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE}"
M001="${M001:-/project/def-maxwl/mforooz/MAX_EPI_IMPUTATION/experiments/001_2026-07-27_eic_leaderboard_repro}"
WORK="${WORK:-/project/def-maxwl/mforooz/CANDII_t54_work}"
AVGDIR="${AVGDIR:-$WORK/avg_official}"
VARIANT="${VARIANT:-trainval}"
LABEL="${LABEL:-our_average_tv_t54}"
OUTDIR="${OUTDIR:-$WORK/gate_scores}"

# shellcheck disable=SC1091
source "$ENT/env_fir.sh"

EXPLIST="${EXPLIST:-$M001/output/bridge/blind_experiments.txt}"
EXP=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$EXPLIST")
[ -z "$EXP" ] && { echo "no experiment at index $SLURM_ARRAY_TASK_ID"; exit 0; }

OUT="$OUTDIR/${LABEL}__${EXP}.csv"
if [ -s "$OUT" ]; then echo "$OUT exists, skipping"; exit 0; fi
mkdir -p "$OUTDIR"

python3 "$ENT/vendor/fir_score.py" \
    --experiment "$EXP" \
    --bridge "$M001/output/bridge/eic_bridge.csv" \
    --repo "$M001/output/inputs/imputation_challenge" \
    --truth official --truth-dir "$DATA/blind_truth" \
    --pred average --avg-dir "$AVGDIR" --variant "$VARIANT" \
    --var-source none \
    --label "$LABEL" \
    --out "$OUT"
