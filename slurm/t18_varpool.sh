#!/bin/bash
# t18 / D7 — build the `msevar` cross-biosample variance pools on the store, where the store is.
#
#   sbatch slurm/t18_varpool.sh                       # eic, chr21 chr22 chrX, T_ pool, pval space
#   CHROMS="chr21" sbatch slurm/t18_varpool.sh        # one chromosome
#
# COST, MEASURED, NOT GUESSED. On Fir a full chr21 pval track reads in 0.17 s and one 30-member
# assay stack in 8.2 s, so 35 assays of chr21 is about 5 minutes. chr21+chr22+chrX is 9.7 M bins
# against chr21's 1.87 M, so roughly half an hour. The whole genome would be ~5.5 h; it is not
# built here because nothing scores it yet, and a pool nobody reads is 12 GB of scratch.
#
# WHY --gres ON A JOB WITH NO GPU COMPUTE: see slurm/bake.sh. Submitted without it this routes to
# def-maxwl_cpu (fairshare 0.088 against the gpu account's 0.435) and effectively never starts.
# The slice is idle for the half hour this takes; that is the intended trade, and the spec is the
# project hard-rule MIG slice and never any other.
#SBATCH --account=def-maxwl
#SBATCH --job-name=candi_t18_varpool
#SBATCH --output=slurm-logs/t18_varpool_%j.out
#SBATCH --error=slurm-logs/t18_varpool_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

set -uo pipefail

KIT="${KIT:-$HOME/projects/def-maxwl/$USER/CANDII_t18}"
VENV="${VENV:-$HOME/projects/def-maxwl/$USER/candi_venv}"
STORE="${STORE:-$HOME/projects/def-maxwl/$USER/CANDI_STORE/eic}"
OUT="${OUT:-/scratch/$USER/candi_kit/varpool}"
CORPUS="${CORPUS:-eic}"
CHROMS="${CHROMS:-chr21 chr22 chrX}"
PREFIX="${PREFIX:-T_}"
SPACE="${SPACE:-pval}"
BLOCK="${BLOCK:-2000000}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export MPLBACKEND=Agg
module load StdEnv/2023 python/3.10.13 >/dev/null 2>&1
source "$VENV/bin/activate" || { echo "[error] no venv at $VENV" >&2; exit 1; }
cd "$KIT"
echo "[t18] host=$(hostname) commit=$(git rev-parse --short HEAD) store=$STORE chroms=$CHROMS"

PYTHONPATH="$KIT/src" python tools/build_varpool.py \
    --store "$STORE" --out "$OUT" --corpus "$CORPUS" \
    --chroms $CHROMS --prefix "$PREFIX" --space "$SPACE" --block-bins "$BLOCK"
rc=$?
echo "[t18] exit=$rc"
ls -la "$OUT/$CORPUS/" || true
exit $rc
