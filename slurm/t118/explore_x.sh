#!/bin/bash
# t118 exploratory: design X (g reads the bin value as well as C, C'), outside the pre-registered
# 576-run ladder. One array task = one (g, space, model, seed): train, then score the trained pairs.
# Index = g_i*12 + space_i*6 + model_i*3 + seed over G=(C19M16 C12M02), SPACES=(counts pval),
# MODELS=(real nocov), SEEDS=(0 1 2) -> 24 tasks. Positional args only; outputs under <out_dir>/runs.
#   sbatch --array=0-23%24 --output=$OUT/logs/%x_%A_%a.out \
#       $KIT/slurm/t118/explore_x.sh $KIT $PRODUCTS $CACHE $OUT
#SBATCH --account=def-maxwl
#SBATCH --job-name=t118L_xplore
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M

set -euo pipefail
USAGE="usage: explore_x.sh <kit_dir> <products_dir> <cache_dir> <out_dir> [blacklist.bed]"
KIT="${1:?$USAGE}"; PRODUCTS="${2:?$USAGE}"; CACHE="${3:?$USAGE}"; OUT="${4:?$USAGE}"
BLACKLIST="${5:-/project/def-maxwl/mforooz/EIC_REPRO/002/scripts/hg38_blacklist_v2.bed}"
IDX="${SLURM_ARRAY_TASK_ID:?array task only}"
GS=(C19M16 C12M02); SPACES=(counts pval); MODELS=(real nocov)
G=${GS[$((IDX / 12))]}; SPACE=${SPACES[$(((IDX / 6) % 2))]}; MODEL=${MODELS[$(((IDX / 3) % 2))]}; SEED=$((IDX % 3))
RUN="X_${G}_${SPACE}_${MODEL}_s${SEED}"; RUNS="$OUT/runs"; MANIFEST="$PRODUCTS/MANIFEST.tsv"
COVARIATES="$KIT/tools/t118/covariates.tsv"
echo "=== explore_x idx=$IDX run=$RUN host=$(hostname) $(date -u)"
if [ -f "$RUNS/$RUN/SCORE_DONE" ]; then echo "skip $RUN: SCORE_DONE exists"; exit 0; fi
mkdir -p "$RUNS"

set +u; module load python/3.10.13; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$KIT/requirements-fir.txt"
pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
export PYTHONPATH="$KIT/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

python3 "$KIT/tools/t118/ladder/train.py" train "$MANIFEST" "$COVARIATES" "$CACHE" "$RUNS" X "$G" "$SPACE" "$MODEL" "$SEED"
python3 "$KIT/tools/t118/ladder/score.py" trained "$MANIFEST" "$COVARIATES" "$CACHE" "$BLACKLIST" "$RUNS/$RUN" --workers 4
echo "=== done $RUN $(date -u)"
