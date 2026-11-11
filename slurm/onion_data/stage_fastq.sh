#!/bin/bash
# D7 (t168): stage an experiment's FASTQs from the ENCODE portal (compute nodes have internet),
# md5-checked against the portal, mode 0444, under /scratch/mforooz/onion/fastq/<experiment>/.
# Also copies the hg38 reference bundle to /project/def-maxwl/mforooz/ONION/pipelines/hg38/ once
# (idempotent; the source refcache on /scratch is purgeable).
#
#   sbatch slurm/onion_data/stage_fastq.sh <kit dir> <experiment> [<experiment> ...]
#
# fastq_uniform.py run stages by itself when a FASTQ is missing; this job exists so a build can
# pre-stage cheaply (2 cores) and so the alignment job's sacct record holds the alignment alone.
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_stage
#SBATCH --output=/project/def-maxwl/mforooz/ONION/logs/stage_%j.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --gres=none

set -uo pipefail
KIT="${1:?usage: stage_fastq.sh <kit dir> <experiment>...}"
shift
[ $# -ge 1 ] || { echo "usage: stage_fastq.sh <kit dir> <experiment>..."; exit 2; }
EIC=/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline   # read only
REFS=/project/def-maxwl/mforooz/ONION/pipelines/hg38

echo "=== onion_stage job ${SLURM_JOB_ID:-none} host $(hostname) experiments: $* $(date -u)"
source "$EIC/runner/env.sh"
export PYTHONUNBUFFERED=1
PY="$KIT/tools/onion_data/fastq_uniform.py"
if [ ! -s "$REFS/hg38.local.tsv" ]; then
  python "$PY" prepare-refs --dest "$REFS" || { echo "[onion-stage] exit=4 (refs)"; exit 4; }
fi
args=()
for e in "$@"; do args+=(--experiment "$e"); done
python "$PY" stage "${args[@]}"
rc=$?
echo "[onion-stage] exit=$rc $(date -u)"
exit "$rc"
