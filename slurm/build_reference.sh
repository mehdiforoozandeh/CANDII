#!/bin/bash
# candi — build the h74 per-assay, per-position average reference. ONE-OFF, CPU-only.
#
#   mkdir -p slurm-logs
#   VENV=~/projects/def-maxwl/mforooz/EpiDenoise/candi_venv sbatch candi/slurm/build_reference.sh
#
# This is NOT a re-bake. It streams the existing h5 once, accumulating per (assay, window, bin) the
# sum and count of DEPTH-NORMALIZED dsf1 counts over the T_ biosamples, and writes
# `<h5 stem>.reference.h5` (~0.8 GB on the 35-assay / 7,485-window / 768-bin panel).
#
# THE GPU IS NOT USED and the request is not a mistake. The venv's Python comes from the cvmfs
# `x86-64-v4` tree, so it segfaults on import on a plain CPU node that lacks those instructions
# (numpy dies with a bogus "circular import" from `numpy.core._internal`). Requesting the MIG slice
# pins the job to the same node class the training arms run on, which is where the venv works. Same
# reason `bake_eic_full.sh` carries it, and it satisfies the project's blanket gres constraint.
#
# /scratch IS PURGE-ELIGIBLE. The reference lives beside the h5 and is regenerable by re-running this
# in ~30 min, but check it exists before launching the arms — a missing file aborts the run at load.
#SBATCH --account=def-maxwl
#SBATCH --job-name=ref74
#SBATCH --output=slurm-logs/ref74_%j.out
#SBATCH --error=slurm-logs/ref74_%j.err
#SBATCH --time=02:30:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=2

set -uo pipefail

KIT="${KIT:-/project/6014832/mforooz/EpiDenoise/candi}"
VENV="${VENV:-${VIRTUAL_ENV:-}}"
H5="${H5:-/scratch/$USER/candi_kit/eic_full.h5}"
OUT="${OUT:-/scratch/$USER/candi_kit/eic_full.reference.h5}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export MPLBACKEND=Agg
if [ -n "$VENV" ]; then
  source "$VENV/bin/activate"
else
  echo "[error] no environment: set VENV=/path/to/venv, or sbatch from an active venv" >&2; exit 1
fi

cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
echo "[ref] host=$(hostname) h5=$H5 out=$OUT"
echo "[ref] started $(date -Is)"

python -m candi.reference build \
  --h5 "$H5" \
  --out "$OUT" \
  --summary-json "${OUT%.h5}.summary.json"
rc=$?

if [ $rc -eq 0 ]; then
  # Gates L1/L2/L3 against the REAL artifacts, not the synthetic panel the unit tests use. A green
  # unit suite proves the arithmetic; this proves THIS file was built the way the arithmetic assumes.
  python -m candi.reference verify --h5 "$H5" --ref "$OUT"
  rc=$?
fi

echo "[ref] finished $(date -Is)"
echo "[ref] DONE rc=$rc"
exit $rc
