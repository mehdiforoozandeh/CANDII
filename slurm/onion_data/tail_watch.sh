#!/bin/bash
# Tail watchdog of the EIC build (PI ruling 38). One run writes <tail dir>/TAIL_STATUS.tsv from
# sacct, the build list and the build logs (tools/onion_data/tail_watch.py status). While a tail
# row is still queued or running it resubmits itself, at most once per 6 h, after every live tail
# job has ended (--dependency=afterany on the jobs still in the queue, --begin=now+6hours); when
# every tail row is verified or terminal it writes TAIL_DONE and stops. It never submits a build
# row and never deletes anything; resubmitting a failed row stays the foreman's.
#
#   sbatch [--dependency=afterany:<tail job ids>] slurm/onion_data/tail_watch.sh <kit> <build list> [tail dir]
#   (the tail dir holds tail_rows.tsv, written by `tail_watch.py init`)
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_tailwatch
#SBATCH --output=/project/def-maxwl/mforooz/ONION/logs/tailwatch_%j.out
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000M
#SBATCH --gres=none

set -uo pipefail
KIT="${1:?usage: tail_watch.sh <kit> <build list> [tail dir]}"
BL="${2:?build list}"
TD="${3:-/project/def-maxwl/mforooz/ONION/eic/tail}"
SELF="$KIT/slurm/onion_data/tail_watch.sh"
echo "=== onion_tailwatch job ${SLURM_JOB_ID:-none} $(date -u)"
OUT=$(python3 "$KIT/tools/onion_data/tail_watch.py" status --build-list "$BL" --out-dir "$TD")
rc=$?
echo "$OUT"
[ "$rc" -eq 0 ] && { echo "tail done"; exit 0; }
[ "$rc" -eq 3 ] || { echo "tail_watch.py status failed (rc=$rc); not resubmitting"; exit "$rc"; }
# one watcher at a time: another queued watcher (not this job) means nothing to do
OTHER=$(squeue -u "$USER" -h -n onion_tailwatch -o '%i' | grep -vx "${SLURM_JOB_ID:-x}" | head -1)
[ -z "$OTHER" ] || { echo "watcher $OTHER already queued; not resubmitting"; exit 0; }
LIVE=$(echo "$OUT" | sed -n 's/^live //p')
QUEUED=()
for j in ${LIVE//,/ }; do
  squeue -h -j "$j" -o '%i' 2>/dev/null | grep -q . && QUEUED+=("$j")
done
DEP=()
[ "${#QUEUED[@]}" -gt 0 ] && DEP=(--dependency="afterany:$(IFS=:; echo "${QUEUED[*]}")")
NEXT=$(sbatch --parsable --begin=now+6hours ${DEP[@]+"${DEP[@]}"} "$SELF" "$KIT" "$BL" "$TD") || exit 4
echo "$NEXT" > "$TD/WATCHER_JOB"
echo "next watcher $NEXT (begin now+6h${DEP[*]:+, ${DEP[*]}})"
