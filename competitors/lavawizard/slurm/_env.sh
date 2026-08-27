# Shared environment for every Lavawizard job on Fir. Sourced, never run.
#
# `candi_venv` and not the port's own `torch_env`: `store_eic.py` imports `candi`, and `candi`
# imports `x_transformers`, which `torch_env` does not carry. The Dataset-3 side still runs in
# `torch_env` — see the README's split.
set -uo pipefail
REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_t53}"
VENV="${VENV:-/project/def-maxwl/mforooz/candi_venv}"
RUNS="${RUNS:-/project/def-maxwl/mforooz/rivals_src/lavawizard_runs}"
REGIME="${REGIME:-$REPO/configs/regime.eic_val.json}"
CACHE="${CACHE:-$RUNS/eic_cache}"
CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 \
        chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX)
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
module load StdEnv/2023 python/3.11 >/dev/null 2>&1 || true
source "$VENV/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/competitors"
mkdir -p "$RUNS" slurm-logs
