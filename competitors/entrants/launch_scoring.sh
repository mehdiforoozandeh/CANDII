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
DRY="${DRY:-0}"

cd "$ENT" || exit 1
mkdir -p slurm_logs

# The two published baselines go FIRST. They are the rows every entrant is read against, and
# `Average` is the one method whose nine measures we can already check against the gate -- so if
# something is wrong with the scoring path, it surfaces in wave 1 rather than after 23 teams.
METHODS=()
for d in "$BASE"/*/; do [ -d "$d" ] && METHODS+=("$d"); done
for d in "$SUB"/*/;  do [ -d "$d" ] && METHODS+=("$d"); done

echo "[launch] ${#METHODS[@]} methods, waves of $WAVE_SIZE"

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
            out=$(sbatch "${args[@]}" slurm/score_entrant.slurm 2>&1 | tail -1)
            id=$(echo "$out" | grep -oE '[0-9]+$')
            if [ -z "$id" ]; then
                echo "  wave $wave: SUBMIT FAILED for $label -- $out" >&2
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

echo "[launch] $wave waves submitted; each holds until the previous finishes"
echo "[launch] watch:  squeue -u \$USER -h -o '%.12i %.10j %.2t %.8M' | grep t54_score | head"
