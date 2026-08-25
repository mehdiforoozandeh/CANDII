#!/bin/bash
# Pull the organizers' own two baseline prediction sets, which the placement table needs beside the
# 23 entrants: `Average` (the published average-activity baseline, team_id 100) and `Avocado_p0`
# (team_id 0). Both are bigwigs on the official grid, so they go through the SAME
# score_entrant.py path as every entrant -- nine measures plus the P-block -- rather than a
# second, special-cased route.
#
#   BASE=average  -> baselines/average_predictor/test            syn20957925  56 files  51.6 GB
#   BASE=avocado  -> baselines/avocado_training_and_validation/test syn20606515  56 files  54.1 GB
#
# `--only-from blind_experiments.txt` cuts each to the 51 that matter. Both folders carry five extra
# C14* tracks -- C14M01, C14M02, C14M16, C14M17, C14M22 -- for a cell that is not in the EIC 363 at
# all, so they are predictions with no truth here and pulling them would be ~10 GB of dead weight.
#
# We ALSO hold our own rebuilt Average, which reproduced 001 bit-identically (see README §4). That
# one is npz and feeds the nine measures through vendor/fir_score.py; this download is what lets the
# published Average be carried with a P-block too, and lets the two be cross-checked against each
# other.
#
#   sbatch --array=0-3 --export=ALL,BASE=average slurm/fetch_baselines.slurm
#   sbatch --array=0-3 --export=ALL,BASE=avocado slurm/fetch_baselines.slurm
#
#SBATCH --job-name=t54_base
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=slurm_logs/t54_base_%A_%a.out
#SBATCH --error=slurm_logs/t54_base_%A_%a.err

set -uo pipefail
REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_t54}"
ENT="$REPO/competitors/entrants"
M001="${M001:-/project/def-maxwl/mforooz/MAX_EPI_IMPUTATION/experiments/001_2026-07-27_eic_leaderboard_repro}"
WORK="${WORK:-/project/def-maxwl/mforooz/CANDII_t54_work}"

: "${BASE:?set BASE to average or avocado}"
case "$BASE" in
    average) FOLDER=syn20957925; NAME=Average    ;;
    avocado) FOLDER=syn20606515; NAME=Avocado_p0 ;;
    *) echo "BASE must be average or avocado, got '$BASE'"; exit 1 ;;
esac
DEST="${DEST:-$HOME/scratch/t54_baselines/$NAME}"

module --force purge
module load StdEnv/2023 python/3.11

# PREFLIGHT -- a job that does nothing must not report COMPLETED.
# On 2026-08-25 four of eight baseline tasks exited 0 in six seconds having produced no stdout and
# downloaded nothing. All four had landed on node fc30560; every task on every other node worked.
# The module load left python unusable there, and without this check SLURM recorded the no-op as
# success -- the worst possible failure mode, because only a file count caught it.
command -v python3 >/dev/null 2>&1 || {
    echo "[preflight] FATAL: python3 absent after module load on $(hostname)" >&2; exit 1; }
python3 -c "import sys, json, urllib.request, hashlib; print('[preflight] python', sys.version.split()[0])" || {
    echo "[preflight] FATAL: python3 present but unusable on $(hostname)" >&2; exit 1; }

N=${NSHARD:-${SLURM_ARRAY_TASK_COUNT:-1}}
I=${SLURM_ARRAY_TASK_ID:-0}
echo "[shard] $NAME task $I of $N -> $DEST" >&2

mkdir -p "$WORK"
ARGS=(--token-file "$HOME/.synapse_pat" --folder-id "$FOLDER" --dest "$DEST"
      --shard "$I/$N" --only-from "$M001/output/bridge/blind_experiments.txt")
if [ "${SURVEY:-0}" = "1" ]; then
    ARGS+=(--survey --survey-out "$WORK/baseline_survey_$NAME.json")
elif [ "${VERIFY:-0}" = "1" ]; then
    ARGS+=(--verify --survey-in "$WORK/baseline_survey_$NAME.json")
else
    # each restricted folder is ~47 GB, well under the 500 GB guard, so no --yes-i-checked needed
    :
fi

python3 "$ENT/fetch_submissions.py" "${ARGS[@]}"
