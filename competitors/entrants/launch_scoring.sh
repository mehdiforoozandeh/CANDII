#!/bin/bash
# Submit the whole 25-method scoring grid as dependency-chained WAVES.
#
# 25 methods x 51 array tasks at 48 GB each, submitted at once, would starve the CPU partition that
# ChromImpute's arrays and t49's P2 generation also run on. So each wave holds until the previous
# one finishes: SLURM enforces the stagger, and everything can be submitted in a single pass without
# anyone having to come back and launch the next batch by hand.
#
# `afterany` rather than `afterok` is deliberate. A wave contains DNase tasks that exit 0 without
# scoring, and may contain a genuine per-experiment failure; either way the next wave should still
# run, because a failure is something to find in the verification pass and resubmit for one method,
# not a reason to strand the other twenty.
#
#   ./launch_scoring.sh            # submit every wave
#   WAVE_SIZE=3 ./launch_scoring.sh
#   DRY=1 ./launch_scoring.sh      # print the sbatch lines and submit nothing
#
set -uo pipefail

REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_t54}"
ENT="$REPO/competitors/entrants"
SUB="${SUB:-$HOME/scratch/t54_submissions_round2}"
BASE="${BASE:-$HOME/scratch/t54_baselines}"
WAVE_SIZE="${WAVE_SIZE:-5}"
WORK="${WORK:-/project/def-maxwl/mforooz/CANDII_t54_work}"
EXPECT_CSV="${EXPECT_CSV:-48}"   # 51 blind experiments minus the 3 DNase, never scored (B3)
UNSUBMITTED=()
DRY="${DRY:-0}"

cd "$ENT" || exit 1
mkdir -p slurm_logs

# Parse the job script BEFORE submitting 1275 tasks against it. A shell syntax error costs a whole
# wave: on 2026-08-25 an apostrophe inside `${PREDDIR:?...}` opened a single quote that swallowed
# the file, and 255 tasks failed with exit 2 in three seconds each before it was caught. `sbatch`
# does not parse the script, and `--test-only` does not either -- only bash does.
for f in slurm/score_entrant.slurm env_fir.sh; do
    bash -n "$f" || { echo "[launch] FATAL: $f does not parse -- nothing submitted" >&2; exit 1; }
done
echo "[launch] syntax ok: slurm/score_entrant.slurm, env_fir.sh"

# The two published baselines go FIRST. They are the rows every entrant is read against, and
# `Average` is the one method whose nine measures we can already check against the gate -- so if
# something is wrong with the scoring path, it surfaces in wave 1 rather than after 23 teams.
ALL=()
for d in "$BASE"/*/; do [ -d "$d" ] && ALL+=("$d"); done
for d in "$SUB"/*/;  do [ -d "$d" ] && ALL+=("$d"); done

# RESUMABLE. A method whose CSVs are all present is skipped, so re-running after a partial launch
# submits only what is missing. Not a nicety: the first relaunch stopped after 9 of 25 sbatch calls
# -- the queue was still draining 1275 tasks from the previous attempt and sbatch began refusing --
# and without this, fixing it would mean re-scoring the nine that had already succeeded.
METHODS=()
skipped=0
for d in "${ALL[@]}"; do
    label=$(basename "$d")
    have=$(ls "$WORK/entrant_scores/$label"/*.csv 2>/dev/null | wc -l)
    if [ "$have" -ge "$EXPECT_CSV" ]; then
        skipped=$((skipped + 1))
    else
        METHODS+=("$d")
    fi
done

echo "[launch] ${#ALL[@]} methods total, $skipped already complete, ${#METHODS[@]} to submit"
[ ${#METHODS[@]} -eq 0 ] && { echo "[launch] nothing to do"; exit 0; }
echo "[launch] waves of $WAVE_SIZE"

dep=""
wave=0
i=0
while [ $i -lt ${#METHODS[@]} ]; do
    wave=$((wave + 1))
    ids=()
    n=0
    while [ $n -lt "$WAVE_SIZE" ] && [ $i -lt ${#METHODS[@]} ]; do
        d="${METHODS[$i]}"
        label=$(basename "$d")
        args=(--array=0-50 --export=ALL,LABEL="$label",PREDDIR="$d")
        [ -n "$dep" ] && args+=(--dependency=afterany:"$dep")
        if [ "$DRY" = "1" ]; then
            echo "  wave $wave: sbatch ${args[*]} slurm/score_entrant.slurm"
            ids+=("000$wave$n")
        else
            # sbatch itself can refuse under queue pressure, which is how nine of twenty-five got
            # submitted last time. Retry before giving up, and record the miss so the run cannot
            # end quietly short.
            id=""
            for attempt in 1 2 3 4 5; do
                out=$(sbatch "${args[@]}" slurm/score_entrant.slurm 2>&1 | tail -1)
                id=$(echo "$out" | grep -oE '[0-9]+$')
                [ -n "$id" ] && break
                echo "  wave $wave: submit attempt $attempt/5 failed for $label -- $out" >&2
                sleep $((10 * attempt))
            done
            if [ -z "$id" ]; then
                echo "  wave $wave: SUBMIT FAILED for $label after 5 attempts" >&2
                UNSUBMITTED+=("$label")
            else
                echo "  wave $wave: $label -> $id"
                ids+=("$id")
            fi
        fi
        i=$((i + 1))
        n=$((n + 1))
    done
    # the next wave waits on every job in this one
    dep=$(IFS=:; echo "${ids[*]}")
done

if [ ${#UNSUBMITTED[@]} -gt 0 ]; then
    echo "[launch] ERROR: ${#UNSUBMITTED[@]} method(s) never submitted: ${UNSUBMITTED[*]}" >&2
    echo "[launch] re-run this script once the queue drains -- it skips what is already complete" >&2
    exit 1
fi
echo "[launch] $wave waves submitted; each holds until the previous finishes"
echo "[launch] watch:  squeue -u \$USER -h -o '%.12i %.10j %.2t %.8M' | grep t54_score | head"
