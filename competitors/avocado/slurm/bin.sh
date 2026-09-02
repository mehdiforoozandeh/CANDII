#!/bin/bash
# Avocado stage 1 -- CANDI_STORE to one (n_bins, 267) float32 matrix per chromosome.
#
# FOUR array tasks, not 23. Avocado only ever touches the regime's train_chroms (the joint fit,
# §3.2) and its eval_chroms (the genomic-factor fits and the only scored scope, §4's blanking
# ruling), so the other 19 matrices are 129 GB of /scratch nothing reads. The list comes off the
# regime -- see _env.sh.
#
# Each task is pure h5 reading. The MIG slice is requested for the same reason `slurm/bake.sh`
# requests one: no GPU compute happens here, the smallest slice only routes the job through the GPU
# account, and AGENTS.md 3.13 permits no other gres spec.
#
#   mkdir -p slurm-logs && sbatch --array=0-3 competitors/avocado/slurm/bin.sh
#
#SBATCH --account=def-maxwl
#SBATCH --job-name=t81_avo_bin
#SBATCH --output=slurm-logs/t81_avo_bin_%A_%a.out
#SBATCH --error=slurm-logs/t81_avo_bin_%A_%a.err
# MEASURED, not guessed: chr21 (1.87M bins x 267 tracks) took 1444 s. The cost is close to linear
# in bins. chr19 is 2.34M bins and chr20 is 2.58M, so every task in the new four-chromosome list is
# under 40 min -- the 06:00:00 below was sized for chr1 (9.96M bins), which this list never touches.
# It is left at 6 h anyway: an over-ask on a 40-minute job costs queue position, not compute, and a
# task that times out leaves a .tmp behind and restarts from nothing.
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source "${SLURM_SUBMIT_DIR:-$PWD}/competitors/avocado/slurm/_env.sh"

# The shared fit's scope first, then eval_chroms: index 0 is what stage 2 needs before anything
# else can start. Under a `regions` regime index 0 is the whole BED scope in ONE task, not one task
# per train chromosome -- the pilot scope is 1,023,489 bins over 18 chromosomes, 1.1 GB against a
# whole chromosome's 2.7 GB, so splitting it would cost more in re-reads than it saves.
NEEDED=("$SHARED_SCOPE" "${EVAL_CHROMS[@]}")
SCOPE="${NEEDED[${SLURM_ARRAY_TASK_ID:-0}]:-}"
[ -z "$SCOPE" ] && { echo "no scope for index ${SLURM_ARRAY_TASK_ID:-0}; this regime needs ${#NEEDED[@]} (${NEEDED[*]})"; exit 1; }

echo "[avo_bin] regime=$REGIME_NAME scope=$SCOPE host=$(hostname) ws=$WS"
if [ "$SCOPE" = "regions" ]; then
    python "$AVO/bin_store.py" --regime "$REGIME" --out "$WS/binned" --regions
else
    python "$AVO/bin_store.py" --regime "$REGIME" --out "$WS/binned" --chrom "$SCOPE"
fi
