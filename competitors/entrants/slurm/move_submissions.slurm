#!/bin/bash
# t90 -- carry the 2019 field's blind-test predictions off /scratch before the purge takes them.
#
# Two trees, both on Fir scratch and both single-copy:
#
#   /scratch/mforooz/t54_submissions_round2   960 GB, 23 entrant teams, 51 C##M##.bigwig each
#   ~/scratch/t54_baselines                    90 GB, the organizers' own Average and Avocado_p0
#
# Oldest file 2026-08-25, so the 60-day purge clock expires around 2026-10-24. /project is a
# Lustre PROJECT quota (project id 6014832 = def-maxwl, 30 TB / 5 M files, directory-inherited),
# so the group a file carries does not change what it is charged to -- `rsync -a` may preserve
# group `mforooz` here without any quota consequence.
#
# THIS SCRIPT ONLY COPIES. It has no `--delete` and no `rm`, on purpose and permanently: the
# scratch copy stays where it is until a human deletes it, having read a VERIFIED.txt.
#
# Verification is three independent checks per tree, all of which must pass before VERIFIED.txt
# is written:
#
#   1. a second `rsync -a -c --dry-run` that re-reads BOTH sides and checksums every file. Its
#      output must be EMPTY. `-c` is what makes this a byte-equality proof rather than the
#      size+mtime heuristic the copy itself used.
#   2. file counts match.
#   3. `du -s` of both sides, recorded (not asserted -- Lustre block accounting can differ).
#
#   sbatch /project/def-maxwl/mforooz/tools/move_submissions.slurm
#
# No `--partition`: the Alliance job-submit plugin derives it from --time (36 h -> the b4 bin,
# cpubase_bycore_b4, MaxTime 3 days). This job runs ON Fir, Lustre to Lustre, so there is no ssh
# and no BatchMode question anywhere in it.
#
#SBATCH --account=def-maxwl
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=t90_move
#SBATCH --output=/project/def-maxwl/mforooz/tools/t90_move_%j.out

set -euo pipefail

SRCS=(/scratch/mforooz/t54_submissions_round2 /home/mforooz/scratch/t54_baselines)
DSTS=(/project/def-maxwl/mforooz/t54_submissions_round2 /project/def-maxwl/mforooz/t54_baselines)

echo "[t90] job ${SLURM_JOB_ID:-none} on $(hostname) starting $(date -Is)"

for i in 0 1; do
    src="${SRCS[$i]}"
    dst="${DSTS[$i]}"

    test -d "$src" || { echo "[t90] FATAL: source $src is not a directory" >&2; exit 1; }

    echo "[t90] === copy  $src -> $dst   $(date -Is)"
    mkdir -p "$dst"
    rsync -a --info=progress2 "$src/" "$dst/"

    # Check 1. Re-read and checksum both sides. Anything rsync would still send is a difference,
    # so silence here is the proof. A non-empty list is a hard stop -- we do not retry, because a
    # mismatch after a clean rsync means something is wrong with the media, not with the copy.
    echo "[t90] === verify $dst by checksum   $(date -Is)"
    resend=$(rsync -a -c --dry-run --out-format='%n' "$src/" "$dst/")
    if [ -n "$resend" ]; then
        echo "[t90] FATAL: checksum dry-run still wants to send these paths:" >&2
        printf '%s\n' "$resend" >&2
        exit 1
    fi

    # Check 2. VERIFIED.txt is excluded on the destination side so that re-running this script
    # over an already-verified tree stays green.
    src_files=$(find "$src" -type f | wc -l)
    dst_files=$(find "$dst" -type f ! -name VERIFIED.txt | wc -l)
    if [ "$src_files" -ne "$dst_files" ]; then
        echo "[t90] FATAL: file count differs -- src=$src_files dst=$dst_files" >&2
        exit 1
    fi

    # Check 3. Recorded, not asserted.
    src_kb=$(du -s "$src" | cut -f1)
    dst_kb=$(du -s "$dst" | cut -f1)

    {
        echo "verified  $(date -Is)"
        echo "job       ${SLURM_JOB_ID:-none}"
        echo "script    competitors/entrants/slurm/move_submissions.slurm (t90)"
        echo "src       $src"
        echo "dst       $dst"
        echo "files     src=$src_files dst=$dst_files"
        echo "du_kb     src=$src_kb dst=$dst_kb"
        echo "proof     rsync -a -c --dry-run src/ dst/ printed nothing"
        echo "note      copy only -- the scratch tree was not modified and nothing was deleted"
    } > "$dst/VERIFIED.txt"

    echo "[t90] --- $dst/VERIFIED.txt"
    cat "$dst/VERIFIED.txt"
done

echo "[t90] all trees verified $(date -Is)"
