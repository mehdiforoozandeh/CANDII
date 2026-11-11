#!/bin/bash
# D7 (t168), KF (t180): stage reads from the ENCODE portal (compute nodes have internet),
# md5-checked against the portal, mode 0444, under /scratch/mforooz/onion/fastq/<experiment>/.
# Also copies the hg38 reference bundle to /project/def-maxwl/mforooz/ONION/pipelines/hg38/ once
# (idempotent; the source refcache on /scratch is purgeable).
#
#   sbatch [--array=<rows>%N] slurm/onion_data/stage_fastq.sh <kit dir> <build list> [row]
#       what one build-list row reads: its FASTQs (any pipeline) or its native redacted BAMs
#   sbatch slurm/onion_data/stage_fastq.sh <kit dir> <experiment> [<experiment> ...]
#       every FASTQ of these experiments
#
# Each file is fetched as ONION_RANGES byte ranges (default 64), ONION_PARALLEL_FILES files at a
# time (default 4): about 256 connections per job. Nibi -> ENCODE S3 runs ~0.14 MB/s per flow and
# scales linearly with flows (NETPROBE 2026-11-11), so throughput is set by the connection count.
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_stage
#SBATCH --output=/project/def-maxwl/mforooz/ONION/logs/stage_%A_%a.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --gres=none

set -uo pipefail
KIT="${1:?usage: stage_fastq.sh <kit dir> <build list> [row] | <kit dir> <experiment>...}"
shift
[ $# -ge 1 ] || { echo "usage: stage_fastq.sh <kit dir> <build list> [row] | <kit dir> <experiment>..."; exit 2; }
EIC=/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline   # read only
REFS=/project/def-maxwl/mforooz/ONION/pipelines/hg38
RANGES="${ONION_RANGES:-64}"
PFILES="${ONION_PARALLEL_FILES:-4}"

echo "=== onion_stage job ${SLURM_JOB_ID:-none} array ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} host $(hostname) args: $* ranges=$RANGES files=$PFILES $(date -u)"
source "$EIC/runner/env.sh"
export PYTHONUNBUFFERED=1
PY="$KIT/tools/onion_data/fastq_uniform.py"
if [ ! -s "$REFS/hg38.local.tsv" ]; then
  python "$PY" prepare-refs --dest "$REFS" || { echo "[onion-stage] exit=4 (refs)"; exit 4; }
fi
if [ -f "$1" ]; then
  ROW="${2:-${SLURM_ARRAY_TASK_ID:-}}"
  [ -n "$ROW" ] || { echo "stage_fastq.sh: build list given but no row (argument or array index)"; exit 2; }
  python "$PY" stage --build-list "$1" --row "$ROW" --ranges "$RANGES" --parallel-files "$PFILES"
else
  args=()
  for e in "$@"; do args+=(--experiment "$e"); done
  python "$PY" stage "${args[@]}" --ranges "$RANGES" --parallel-files "$PFILES"
fi
rc=$?
echo "[onion-stage] exit=$rc $(date -u)"
exit "$rc"
