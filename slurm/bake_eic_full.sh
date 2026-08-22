#!/bin/bash
# candi — bake the FULL EIC panel for the cell-identity conditioning experiment.
#
# Differs from bake.sh only in panel, output name, walltime and memory. Everything else -- the flags,
# the type2 loci counts, --allow-missing-control -- is bake.sh's, deliberately, so the resulting h5 is
# the same shape as the q19 one at a different scale.
#
# SCALE, and why the walltime and memory move:
#   89 biosample entries (51 base cell types under T_/V_/B_) x 35 assays x 7,485 windows
#   (chr19 3,053 train + chr21 2,432 eval + 1,000 cCRE + 1,000 non-cCRE type2).
#   bake.sh's 1 h / 30 G are calibrated on ~15 biosamples x 8 assays; this is ~6x the biosamples and
#   ~4.4x the assay width, and the handler caches a whole chromosome per biosample while it works.
#   ~354 GB uncompressed before the type2 windows, so tens of GB on disk after gzip-1. /scratch has
#   18 TB free; /project is 82% full, which is why OUT lives on scratch (and is purge-eligible --
#   re-bake rather than archive it).
#
#   mkdir -p slurm-logs
#   VENV=~/projects/def-maxwl/mforooz/EpiDenoise/candi_venv \
#     sbatch candi/slurm/bake_eic_full.sh
#
# Logs go to ./slurm-logs/ RELATIVE TO WHERE YOU SUBMIT FROM (SLURM resolves --output against the
# submitting cwd, not the script's location).
#
# WHY --gres ON A CPU-ONLY JOB: see bake.sh. The def-maxwl_cpu fairshare is 0.088 vs the GPU account's
# 0.435, and a plain CPU bake effectively never starts. The MIG slice idles for the duration; that is
# the intended trade. Never use any other gres spec (project hard rule).
#SBATCH --account=def-maxwl
#SBATCH --job-name=bake_eic_full
#SBATCH --output=slurm-logs/bake_eic_full_%j.out
#SBATCH --error=slurm-logs/bake_eic_full_%j.err
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2

set -uo pipefail

KIT="${KIT:-/project/6014832/mforooz/EpiDenoise/candi}"
VENV="${VENV:-${VIRTUAL_ENV:-}}"
ROOT="${ROOT:-/project/6014832/mforooz/DATA_CANDI_EIC}"
SIDE="${SIDE:-/project/6014832/mforooz/EpiDenoise/data}"
PANEL="${PANEL:-$KIT/configs/panel.eic_full.json}"
OUT="${OUT:-/scratch/$USER/candi_kit/eic_full.h5}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
export WANDB_MODE=disabled
if [ -n "$VENV" ]; then
  source "$VENV/bin/activate"
else
  echo "[error] no environment: set VENV=/path/to/venv, or sbatch from an active venv" >&2; exit 1
fi

cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
mkdir -p "$(dirname "$OUT")"

echo "[bake] host=$(hostname) panel=$PANEL out=$OUT"
echo "[bake] started $(date -Is)"

python -m candi.prep.bake \
  --root "$ROOT" \
  --panel "$PANEL" \
  --out "$OUT" \
  --fasta "$SIDE/hg38.fa" \
  --chrom-sizes "$SIDE/hg38.chrom.sizes" \
  --ccres "$SIDE/GRCh38-cCREs.bed" \
  --type2-ccre 1000 --type2-non 1000 \
  --allow-missing-control \
  --seed 42
rc=$?

echo "[bake] finished $(date -Is)"
echo "[bake] DONE rc=$rc"
ls -lh "$OUT" 2>/dev/null
exit $rc
