#!/bin/bash
# Pull syn17083203/submissions_round2/ to the user's SCRATCH. Compute nodes have internet, so a batch
# job is the right vehicle for a transfer measured in hours -- a login-node pull dies with the SSH
# session.
#
# THREE STEPS, in this order:
#
#   1. survey, transferring nothing, and report the total before pulling anything:
#        sbatch --export=ALL,SURVEY=1 slurm/fetch_submissions.slurm
#
#   2. once the total is reported and the pull approved:
#        sbatch --array=0-7 --export=ALL,YES=1 slurm/fetch_submissions.slurm
#
#   3. per-team verification against the survey manifest, transferring nothing:
#        sbatch --export=ALL,VERIFY=1 slurm/fetch_submissions.slurm
#
# The array shards by file (`--shard i/n`, from the vendored downloader), so eight tasks share the
# tree without racing: each writes to a per-process `.part` and renames only after size and md5 both
# check out. Re-running is cheap -- a file already on disk at the right size is skipped -- so a
# timed-out task is resubmitted rather than restarted from zero.
#
# `--max-gb` in the driver refuses a tree over 500 GB without YES=1, so forgetting to report the
# size is an error rather than a surprise on someone else's quota.
#
#SBATCH --job-name=t54_fetch
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --account=def-maxwl_cpu
#SBATCH --output=slurm_logs/t54_fetch_%A_%a.out
#SBATCH --error=slurm_logs/t54_fetch_%A_%a.err

set -uo pipefail
REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_t54}"
ENT="$REPO/competitors/entrants"
# User scratch, NOT /project (PI ruling, 2026-08-25). One-shot, re-downloadable scoring inputs do
# not belong on group-charged quota, and scratch's 60-day purge is the correct lifetime for them.
# Only the DERIVED score tables are permanent; they live in the repo checkout under WORK and rsync
# down to the laptop's cruxvault/results/t54/ as small files.
DEST="${DEST:-$HOME/scratch/t54_submissions_round2}"
WORK="${WORK:-/project/def-maxwl/mforooz/CANDII_t54_work}"

# stdlib only -- no venv needed, which is why this job does not source env_fir.sh
module --force purge
module load StdEnv/2023 python/3.11

# NSHARD pins the partition denominator. Resubmitting only the failed shards -- say
# `--array=0,2,3,5,7` -- sets SLURM_ARRAY_TASK_COUNT to 5, which would silently RE-PARTITION the
# tree and send each task after a different slice than it was originally assigned. Pass NSHARD=8 on
# any partial resubmit so each task resumes exactly its own slice.
N=${NSHARD:-${SLURM_ARRAY_TASK_COUNT:-1}}
I=${SLURM_ARRAY_TASK_ID:-0}
echo "[shard] task $I of $N -> $DEST" >&2

ARGS=(--token-file "$HOME/.synapse_pat" --dest "$DEST" --shard "$I/$N")
if [ "${SURVEY:-0}" = "1" ]; then
    mkdir -p "$WORK"
    ARGS+=(--survey --survey-out "$WORK/submissions_survey.json")
elif [ "${VERIFY:-0}" = "1" ]; then
    # reads the saved manifest, so it needs neither the token nor an API round-trip
    ARGS+=(--verify --survey-in "$WORK/submissions_survey.json")
elif [ "${YES:-0}" = "1" ]; then
    ARGS+=(--yes-i-checked)
fi

python3 "$ENT/fetch_submissions.py" "${ARGS[@]}"
