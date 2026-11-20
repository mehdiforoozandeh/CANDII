#!/bin/bash
# DNase-seq FASTQ kit job: dnase-seq-pipeline v3.0.0, alignment only (t152, chunk D7d).
# Runs tools/onion_data/fastq_dnase.py from a pushed ONION kit. Cromwell (Caper local backend,
# Apptainer) works in $SLURM_TMPDIR; only the products reach /scratch/mforooz/onion/work.
#
# Setup (once; writes /project/def-maxwl/mforooz/ONION/pipelines/dnase, about 6.6 GB in 11 files,
# and the unpacked alignment references under /scratch/mforooz/onion/pipelines/dnase/refs):
#   sbatch --time=03:00:00 --cpus-per-task=2 --mem=8000M \
#     --output=/project/def-maxwl/mforooz/ONION/logs/dnase/setup_%j.out \
#     $KIT/slurm/onion_data/fastq_dnase.sh $KIT setup
# One unit and arm:
#   sbatch --job-name=onion_dnase --time=24:00:00 --cpus-per-task=16 --mem=64000M \
#     --output=/project/def-maxwl/mforooz/ONION/logs/dnase/run_%j.out \
#     $KIT/slurm/onion_data/fastq_dnase.sh $KIT run --experiment ENCSR000EMD --unit experiment --arm base
# Last log line: `[onion-dnase] exit=<rc>`.
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_dnase
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --gres=none

set -uo pipefail
KIT="${1:?usage: fastq_dnase.sh <kit dir> setup|run|units [args...]}"
shift
echo "=== onion dnase job ${SLURM_JOB_ID:-} host=$(hostname) kit=$KIT $(date -u)"
cat "$KIT/.kit_stamp" 2>/dev/null || true
df -h "${SLURM_TMPDIR:-/tmp}" | tail -1
set +u
# never pipe `module load`: a subshell loses the environment
module load StdEnv/2023 python/3.11.5 java/17.0.6 apptainer/1.3.5
# Caper 2.3.2 venv of the EIC reprocessing (read-only here)
source /project/def-maxwl/mforooz/EIC_REPRO/003_pipeline/caper_venv/bin/activate
set -u
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export APPTAINER_CACHEDIR="${SLURM_TMPDIR:-/tmp}/apptainer_cache"
export APPTAINER_TMPDIR="${SLURM_TMPDIR:-/tmp}/apptainer_tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
cd "$KIT"
echo "=== python tools/onion_data/fastq_dnase.py $*"
python tools/onion_data/fastq_dnase.py "$@"
rc=$?
echo "=== done $(date -u)"
echo "[onion-dnase] exit=$rc"
exit $rc
