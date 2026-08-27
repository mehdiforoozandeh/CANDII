#!/bin/bash
# t49 / RIVALS_PLAN.md §2 — protocol P2: the same declared pairs, GENERATED genome-wide.
#
#   mkdir -p slurm-logs && sbatch --array=0-22 slurm/t49_baselines_p2.sh
#
# One array task per chromosome, all writing into the SAME prediction roots — a track directory
# collects one `chr*.npz` per task and is only complete when every task has finished, which is why
# scoring is a separate, dependent job (`t49_baselines_p2_score.sh`). `bench.external` refuses a
# track that covers only some of the scored chromosomes, so a partial array is a loud failure at
# score time rather than a quiet one.
#
# THE POISSON FLOOR HERE IS 1e4, NOT §5.1's 1e6, and that is a deviation on the record.
# `candi.metrics.nb_crps` cannot evaluate 1e6 (see `competitors/baselines/README.md`), so at the
# pre-registered value P2's count arm would ship with no CRPS tier at all. P1 is generated at BOTH
# values so the PI can see the difference cheaply; P2 reads ~370 GB off the store per pass and is
# not run twice to make the same point. The value is stamped in every manifest as `poisson_n`.
#
# READ P2 NUMBERS WITH §2's ASYMMETRY IN MIND, plus one that is specific to these baselines: P2
# covers chr19, which is the regime's TRAIN chromosome, so the kNN similarity ranking and the
# per-assay marginal are IN-SAMPLE on that chromosome. `avg` is unaffected — its exclusion rule is
# over cells, not positions.
#
# WHY --gres ON A CPU-ONLY JOB: see slurm/bake.sh.
#SBATCH --account=def-maxwl
#SBATCH --job-name=candi_t49_p2gen
#SBATCH --output=slurm-logs/t49_p2gen_%A_%a.out
#SBATCH --error=slurm-logs/t49_p2gen_%A_%a.err
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

set -uo pipefail

KIT="${KIT:-/project/def-maxwl/$USER/CANDII_t49}"
VENV="${VENV:-/project/def-maxwl/$USER/candi_venv}"
REGIME="${REGIME:-configs/regime.eic_val.json}"
PRED="${PRED:-/project/def-maxwl/$USER/t49_baselines/p2/preds}"
METHODS="${METHODS:-avg,avg-arcsinh,knn1,knn5,marginal}"
POISSON_N="${POISSON_N:-1e4}"

CHROM_LIST=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 \
            chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX)
CHROM="${CHROM_LIST[${SLURM_ARRAY_TASK_ID:-20}]}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export MPLBACKEND=Agg
module load StdEnv/2023 python/3.10.13 >/dev/null 2>&1
source "$VENV/bin/activate" || { echo "[error] no venv at $VENV" >&2; exit 1; }
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
export PYTHONPATH="$KIT/src:$KIT"

case "$REGIME" in *eic_test*) echo "[t49] REFUSING: $REGIME is the B-pair regime (A4)"; exit 2;; esac
echo "[t49-p2] host=$(hostname) commit=$(git rev-parse --short HEAD) chrom=$CHROM n=$POISSON_N"

mkdir -p "$PRED"
python -m competitors.baselines.generate \
    --store "$REGIME" --out "$PRED" --chroms "$CHROM" --methods "$METHODS" \
    --poisson-n "$POISSON_N"
rc=$?
echo "[t49-p2] chrom=$CHROM exit=$rc"
exit $rc
