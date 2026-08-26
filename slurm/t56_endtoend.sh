#!/bin/bash
# t56 — one whole `bench.external` run on the real t49 P1 root, exact vs --crps-approx K.
#
#   METHOD=avg        sbatch slurm/t56_endtoend.sh    # the closed form, as t49 P1 ran it
#   METHOD=avg K=100  sbatch slurm/t56_endtoend.sh    # the same run with the sampled estimator
#
# Same regime, same varpool, same seed as the t49 P1 run, so the score json this writes is
# comparable key-for-key with cruxvault/results/t49/p1_n1e4/<method>.json. The k-curve
# (tools/t56_crps_sweep.py) measures the estimator alone; this measures the WHOLE run, which is the
# number a P2 re-launch is sized on — the CRPS family is most of a count-arm track's cost, and it is
# not all of it.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t56_e2e
#SBATCH --output=slurm-logs/t56_e2e_%j.out
#SBATCH --error=slurm-logs/t56_e2e_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
set -uo pipefail
KIT=/project/def-maxwl/$USER/CANDII_t56
VENV=/project/def-maxwl/$USER/candi_venv
METHOD="${METHOD:-avg}"; K="${K:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
module load StdEnv/2023 python/3.10.13 >/dev/null 2>&1
source "$VENV/bin/activate"
cd "$KIT"; source "$KIT/slurm/_kit_pin.sh"; export PYTHONPATH="$KIT/src:$KIT"
OUT=/project/def-maxwl/$USER/t56_crps/e2e; mkdir -p "$OUT"
TAG="${METHOD}_${K:-exact}"
echo "[t56-e2e] host=$(hostname) method=$METHOD k=${K:-exact} start=$(date +%s)"
/usr/bin/time -v python -m candi.bench.external \
  --store configs/regime.eic_val.json \
  --pred /project/def-maxwl/$USER/t49_baselines/p1v2/n1e4/preds/$METHOD \
  --out "$OUT/$TAG.json" --chroms chr21 \
  --varpool /scratch/$USER/candi_kit/varpool ${K:+--crps-approx $K --crps-seed 0}
echo "[t56-e2e] done=$(date +%s) rc=$?"
