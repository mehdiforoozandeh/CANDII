#!/bin/bash
# Frozen DNA-model embeddings: build the SEPARATE DNA venv (requirements-dnaemb.txt; PI rulings 10, 22) in
# $SLURM_TMPDIR, then run tools/minicandi/dna_embed.py with the given arguments. The main kit venv is not touched.
#
#   GPU form (precompute; the fixed slice; 2 CPUs on Nibi):
#     sbatch --time=02:00:00 --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=2 --mem=32000M \
#         --output=$OUT/dna_embed_%j.out $KIT/slurm/minicandi/dna_embed.sh $KIT \
#         precompute --source flashzoi --genome $O/genome --out /scratch/mforooz/onion/dna_emb/flashzoi --chroms chr21
#   CPU form (project, verify, folds; the header below):
#     sbatch --time=04:00:00 --output=$OUT/dna_embed_%j.out $KIT/slurm/minicandi/dna_embed.sh $KIT \
#         project --dir /scratch/mforooz/onion/dna_emb/flashzoi --fit-chroms chr1 --dim 128
#   Array form: CHROMS_ARRAY="chr1 chr2 ... chrX" (space separated; export it in the shell, use --export=ALL) and
#     --array=0-22%12: task i appends "--chroms <word i>" to the arguments.
#
# Weights: precompute downloads the pinned revision once into --weights-dir (default
# /project/def-maxwl/mforooz/ONION/dna_models) and checks the pinned sha256; HF_HOME points under $SLURM_TMPDIR.
# DNAEMB_WHEELS: optional find-links directory for the "# pypi" lines (e.g. Max's Fir copy of
# borzoi_pytorch-0.5.1 at /project/def-maxwl/maxwl/epi_imputation/s02/encoder/wheels). DRY_RUN=1 prints the
# install and run commands and exits.
#SBATCH --account=def-maxwl
#SBATCH --job-name=dna_embed
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000M
#SBATCH --gres=none

set -euo pipefail
USAGE="usage: minicandi/dna_embed.sh <kit_dir> <precompute|project|verify|folds> [dna_embed.py args...]"
KIT="${1:?$USAGE}"
shift
[ $# -ge 1 ] || { echo "$USAGE" >&2; exit 2; }
ARGS=("$@")
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -n "${CHROMS_ARRAY:-}" ]; then
  read -r -a CH <<< "$CHROMS_ARRAY"
  [ "$SLURM_ARRAY_TASK_ID" -lt "${#CH[@]}" ] || { echo "array task $SLURM_ARRAY_TASK_ID: no chromosome" >&2; exit 2; }
  ARGS+=(--chroms "${CH[$SLURM_ARRAY_TASK_ID]}")
fi
REQ="$KIT/requirements-dnaemb.txt"
TMP="${SLURM_TMPDIR:-${TMPDIR:-/tmp}/dna_embed_$$}"
WH_REQ="$TMP/req_wheelhouse.txt"
PY_REQ="$TMP/req_pypi.txt"
mkdir -p "$TMP"
grep -v -E '^[^#].*# pypi' "$REQ" > "$WH_REQ"
# "name==ver  # pypi  (wheel sha256 <hex>)" -> "name==ver --hash=sha256:<hex>"
grep -E '^[^#].*# pypi' "$REQ" | sed -E 's/^([^ #]+).*sha256 ([0-9a-f]{64}).*/\1 --hash=sha256:\2/' > "$PY_REQ"
FIND=()
[ -n "${DNAEMB_WHEELS:-}" ] && FIND=(--find-links "$DNAEMB_WHEELS")
INSTALL1=(pip install --no-index -r "$WH_REQ")
INSTALL2=(pip install --no-deps --require-hashes ${FIND[@]+"${FIND[@]}"} -r "$PY_REQ")
RUN=(python3 "$KIT/tools/minicandi/dna_embed.py" "${ARGS[@]}")
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "${INSTALL1[*]}"
  echo "${INSTALL2[*]}"
  cat "$PY_REQ"
  echo "${RUN[*]}"
  exit 0
fi
echo "=== dna_embed job ${SLURM_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none} host=$(hostname) $(date -u)"
set +u; module load StdEnv/2023 gcc/12.3 cuda/12.9 python/3.11; set -u
virtualenv --no-download "$TMP/venv_dnaemb"
source "$TMP/venv_dnaemb/bin/activate"
pip install --no-index --upgrade pip
"${INSTALL1[@]}"
"${INSTALL2[@]}"
pip list 2>/dev/null | grep -iE '^(torch|triton|flash.attn|transformers|borzoi.pytorch|numpy|h5py) ' || true
export HF_HOME="$TMP/hf" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd "$KIT"
"${RUN[@]}"
echo "=== done $(date -u)"
