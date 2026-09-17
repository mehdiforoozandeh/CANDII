#!/bin/bash
# t112, chunk C8: one BAM-route counterfactual arm per array task.
#
# WHAT ONE TASK DOES. It reads its row from a rows TSV by index, then hands it to
# `tools/t112/bam_arm.py run`, which builds the arm's input tagAlign with the pipeline's own
# scripts inside the pipeline's own SIF, runs the recorded `macs2_signal_track` command with only
# the tagAligns, `--fraglen`, `--chrsz` and `--out-dir` substituted, bins both the tagAligns and
# the produced bigwig at 25 bp, and publishes the product. Everything heavy happens in
# $SLURM_TMPDIR (~3.3 TB local disk) and is copied out only after it exists and is non-empty.
#
# NO --export. Alliance arrays lose a comma-valued --export variable, so the rows TSV and the
# output roots are POSITIONAL arguments and the task index is $SLURM_ARRAY_TASK_ID. That is also
# why $KIT is derived from this script's own path: a snapshot in $CF/code/C9 runs its own copy.
#
# THE VENV. `bin25.py pval` imports `tools/dnase_macs2_pval.py`, which imports h5py and pandas at
# module level, and reads the bigwig with pyBigWig. scipy-stack supplies numpy and pandas;
# pyBigWig and h5py come from the Alliance wheelhouse with --no-index, because compute nodes have
# no internet. The venv is built in $SLURM_TMPDIR: never on the login node, which has no numpy.
#
# RESOURCES. The pipeline's own macs2_signal_track for C19M16 took 64 min at 14.9 GB on 1 core;
# 8 h and 48 GB leaves room for the two genome-wide bigwig binnings this task adds.
#
# THE KIT ROOT IS AN ARGUMENT. SLURM runs its own spool copy of this file, so under sbatch
# `dirname "${BASH_SOURCE[0]}"` is /var/spool/... and not the snapshot: deriving $KIT from it gave
# every task `/var/spool/tools/t112/bam_arm.py: No such file` (C9, 2026-09-17). It is now the first
# positional and the script refuses to start without a kit that actually holds the tool.
#
# Usage, from the Nibi login node, after snapshotting the repo to $CF/code/<chunk>:
#   KIT=$CF/code/C9
#   mkdir -p $CF/logs/bamarms $CF/bamarms   # THE LAUNCHER MUST DO THIS. SLURM opens --output
#                                           # before the script body runs, so a job cannot create
#                                           # its own log directory: without it every task dies at
#                                           # launch with no log at all.
#   R=$CF/bamarms/rows_chip.tsv
#   python3 $KIT/tools/t112/arms.py rows --route bam --dnase none --ratio yes > $R
#   sbatch --test-only --array=0-83 $KIT/slurm/t112/bam_arm.sh $KIT $R
#   sbatch --parsable --array=0-83 $KIT/slurm/t112/bam_arm.sh $KIT $R
#SBATCH --account=def-maxwl
#SBATCH --job-name=t112_bam_arm
#SBATCH --output=/scratch/mforooz/t112_cf/logs/bamarms/%x_%A_%a.out
#SBATCH --error=/scratch/mforooz/t112_cf/logs/bamarms/%x_%A_%a.out
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail

USAGE="usage: bam_arm.sh <kit_dir> <rows.tsv> [products_dir] [work_dir] [ta_dir]"
KIT="${1:?$USAGE — kit_dir is the code snapshot, e.g. \$CF/code/C9}"
ROWS="${2:?$USAGE}"
CF=/scratch/mforooz/t112_cf
PRODUCTS="${3:-$CF/products}"
WORK="${4:-$CF/bamarms}"
TA_DIR="${5:-$CF/ta}"
PY_MODULES="StdEnv/2023 python/3.11 scipy-stack"

# Fail here, not 10 s into the venv build, if the kit is wrong or the old argument order was used.
[ -f "$KIT/tools/t112/bam_arm.py" ] || {
    echo "kit_dir '$KIT' has no tools/t112/bam_arm.py. $USAGE" >&2; exit 2; }
KIT=$(cd "$KIT" && pwd)

: "${SLURM_ARRAY_TASK_ID:?array task only}"
mkdir -p "$CF/logs/bamarms"

# The row, by the pinned index rule: +2 skips the header and turns 0-based into sed's 1-based.
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$ROWS")
[ -n "$LINE" ] || { echo "no row at index $SLURM_ARRAY_TASK_ID in $ROWS" >&2; exit 2; }
PID=$(cut -f1 <<< "$LINE")
echo "=== $PID  idx=$SLURM_ARRAY_TASK_ID  job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}  host=$(hostname)  $(date -u)"
df -h "$SLURM_TMPDIR" | tail -1

set +u; module load apptainer/1.3.5; module load $PY_MODULES; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index pyBigWig h5py
python3 -c "import numpy, pandas, h5py, pyBigWig; print('venv ok', numpy.__version__, pyBigWig.__version__)"

cd "$KIT"
# PREPEND, never replace: `module load scipy-stack` puts numpy and pandas on PYTHONPATH, and
# overwriting it makes `import numpy` fail inside the job (seen 2026-09-17, job 22149497_0)
# even though the same import worked one line earlier.
export PYTHONPATH="$KIT/src${PYTHONPATH:+:$PYTHONPATH}"
export KIT
if [ -f "$KIT/GIT_SHA" ]; then export T112_GIT_SHA=$(cat "$KIT/GIT_SHA"); fi

python3 "$KIT/tools/t112/bam_arm.py" run \
    --rows "$ROWS" --index "$SLURM_ARRAY_TASK_ID" \
    --cf "$CF" --products "$PRODUCTS" --work "$WORK" --ta-dir "$TA_DIR" \
    --tmp "$SLURM_TMPDIR"

echo "=== $PID done $(date -u)"
ls -l "$PRODUCTS/$PID"
