#!/bin/bash
# t117: pseudoreplicates of the t112 counterfactual-arm products. One array task per read set
# (mode `readset`) or per (product, half) (mode `half`); `check` and `smoke` are single tasks.
# All the logic is in tools/t117/pseudoreps.py; this file only builds the environment.
#
# The environment is t112's, unchanged (slurm/t112/bam_arm.sh): apptainer 1.3.5 for the pipeline
# SIFs, and a throwaway venv in $SLURM_TMPDIR with pyBigWig + h5py from the Alliance wheelhouse
# (--no-index) on top of scipy-stack. Nothing is installed anywhere else.
#
# The kit root is an argument (SLURM runs a spool copy of this file), and nothing is passed with
# --export (Alliance arrays truncate comma-valued exports). The log dir must exist before sbatch.
#
# Usage, from the Nibi login node:
#   KIT=$CF/code/P1; PR=$CF/pseudoreps; mkdir -p $PR/logs
#   sbatch --array=0-79 --time=3:00:00 --mem=24G $KIT/slurm/t117/pseudoreps.sh $KIT readset $PR/readsets.tsv
#   sbatch --array=...  $KIT/slurm/t117/pseudoreps.sh $KIT half $PR/halves.tsv
#   sbatch $KIT/slurm/t117/pseudoreps.sh $KIT check
#   sbatch --time=2:00:00 $KIT/slurm/t117/pseudoreps.sh $KIT smoke C19M16__base__base chr21
#SBATCH --account=def-maxwl
#SBATCH --job-name=prep_pseudoreps
#SBATCH --output=/scratch/mforooz/t112_cf/pseudoreps/logs/%x_%A_%a.out
#SBATCH --error=/scratch/mforooz/t112_cf/pseudoreps/logs/%x_%A_%a.out
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G

set -euo pipefail

USAGE="usage: pseudoreps.sh <kit_dir> readset|half <tasks.tsv> [--ratio-mode M] | check [pids] | smoke <pid> <chrom>"
KIT="${1:?$USAGE}"
MODE="${2:?$USAGE}"
shift 2
[ -f "$KIT/tools/t117/pseudoreps.py" ] || { echo "kit_dir '$KIT' has no tools/t117/pseudoreps.py" >&2; exit 2; }
KIT=$(cd "$KIT" && pwd)
[ -f "$KIT/GIT_SHA" ] || { echo "kit_dir '$KIT' has no GIT_SHA" >&2; exit 2; }
export KIT T117_GIT_SHA=$(tr -d '[:space:]' < "$KIT/GIT_SHA")

echo "=== t117 $MODE $*  job ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-none}}_${SLURM_ARRAY_TASK_ID:-}  host=$(hostname)  $(date -u)"
df -h "$SLURM_TMPDIR" | tail -1

set +u; module load apptainer/1.3.5; module load StdEnv/2023 python/3.11 scipy-stack; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index pyBigWig h5py
python3 -c "import numpy, pandas, h5py, pyBigWig; print('venv ok', numpy.__version__, pyBigWig.__version__)"
export PYTHONPATH="$KIT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$KIT"

PY="$KIT/tools/t117/pseudoreps.py"
case "$MODE" in
    readset|half)
        TASKS="${1:?$USAGE}"; shift
        : "${SLURM_ARRAY_TASK_ID:?array task only}"
        python3 "$PY" "$MODE" --tasks "$TASKS" --index "$SLURM_ARRAY_TASK_ID" --tmp "$SLURM_TMPDIR" "$@" ;;
    check)
        if [ $# -gt 0 ]; then python3 "$PY" check --pids "$1"; else python3 "$PY" check; fi ;;
    smoke)
        python3 "$PY" smoke --pid "${1:?$USAGE}" --chrom "${2:?$USAGE}" --tmp "$SLURM_TMPDIR" ;;
    *)
        echo "$USAGE" >&2; exit 2 ;;
esac
echo "=== t117 $MODE done $(date -u)"
