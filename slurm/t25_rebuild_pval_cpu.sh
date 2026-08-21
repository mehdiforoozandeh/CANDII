#!/bin/bash
# t25, CPU-routed half. IDENTICAL to slurm/t25_rebuild_pval.sh except that it requests no GPU.
#
# WHY THIS DEPARTS FROM INVARIANT 13, AND WHY THAT IS THE INVARIANT'S OWN LOGIC.
# AGENTS.md invariant 13 fixes every `#SBATCH --gres` line to the smallest MIG slice, and states the
# reason in the same breath: "The bake does no GPU compute; it requests the smallest MIG slice only
# to route through the GPU account." It is a fairshare-routing hack, not a compute requirement --- the
# def-maxwl_cpu fairshare is 0.032 against the GPU account's 0.39, so a CPU job normally never starts.
#
# On 2026-08-21 that stopped being true: 74,607 jobs were pending cluster-wide, every GPU node read
# `mix-`, and the rebuild was managing 1.3 tasks/min. `sbatch --test-only` projected an IMMEDIATE
# start for the same job with no gres, in cpubase_bycore_b1. So this half is routed through the CPU
# account and the other half stays on the MIG slice.
#
# THE INDEX RANGES ARE DISJOINT BY CONSTRUCTION. Two tasks writing one pval.h5 concurrently would
# corrupt it; that is the only real hazard here and it is handled by partitioning, not by luck.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t25_pval_cpu
#SBATCH --output=/project/def-maxwl/mforooz/CANDI_STORE/t25/logs/pval_%A_%a.out
#SBATCH --error=/project/def-maxwl/mforooz/CANDI_STORE/t25/logs/pval_%A_%a.err
#SBATCH --time=30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -uo pipefail

STORE=/project/def-maxwl/mforooz/CANDI_STORE
CORPUS="${CORPUS:?set CORPUS}"
SRC="${SRC:?set SRC}"
KIT="${KIT:-/home/mforooz/projects/def-maxwl/mforooz/CANDII}"
LIST="${LIST:-$STORE/t25/${CORPUS}_biosamples.txt}"
PVAL_SCALE="${PVAL_SCALE:-2000}"
PVAL_TRANSFORM="${PVAL_TRANSFORM:-arcsinh}"

B=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
if [ -z "$B" ]; then
  echo "[t25] no biosample at index $SLURM_ARRAY_TASK_ID in $LIST" >&2; exit 2
fi

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export MPLBACKEND=Agg
source /project/6014832/mforooz/EpiDenoise/candi_venv/bin/activate
cd "$KIT"; export PYTHONPATH="$KIT/src"

echo "[t25] host=$(hostname) corpus=$CORPUS biosample=$B idx=$SLURM_ARRAY_TASK_ID route=cpu"
echo "[t25] codec=${PVAL_TRANSFORM} scale=${PVAL_SCALE} kit=$(git -C "$KIT" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "[t25] started $(date -Is)"
t0=$(date +%s)

python -m candi.store build-biosample \
  --source-root "$SRC" --corpus-root "$STORE/$CORPUS" \
  --chrom-sizes "$STORE/genome/chrom_sizes.json" \
  --biosample "$B" --kinds pval \
  --pval-scale "$PVAL_SCALE" --pval-transform "$PVAL_TRANSFORM" --overwrite
rc=$?

t1=$(date +%s)
echo "[t25] corpus=$CORPUS biosample=$B rc=$rc elapsed_s=$((t1-t0)) finished $(date -Is)"
exit $rc
