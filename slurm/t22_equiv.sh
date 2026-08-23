#!/bin/bash
# t22 — one checkpoint, scored by BOTH harnesses, so the cutover's equivalence report compares a
# measurement change and nothing else.
#
#   sbatch slurm/t22_equiv.sh train      # train q19 + eval.py's own M1/M2/M3/S14 -> <tag>.json
#   sbatch slurm/t22_equiv.sh bench      # candi.bench over the SAME checkpoint  -> <tag>.bench.json
#
# WHY THE SAME CHECKPOINT IS GUARANTEED. `train.py` writes the last checkpoint (line 1300) and then
# calls `evaluate()` on the model still in memory (line 1339). So `<tag>.ckpt` and the `M1/M2/M3/S14`
# blocks inside `<tag>.json` are the same weights, and `bench` reloading that file scores those
# weights again. Nothing in the report is a model difference.
#
# WHY q19 AND NOT eic_full. The report is about which keys moved and by what mechanism, and the
# dominant mechanism — position scope, whole chr21 against the old `--eval-budget` subsample — shows
# up identically on 8 assays and on 35. q19 is 1.8 GB against 24 GB and one 85-minute job against
# many. eic_full remains the run to make when the numbers are to be QUOTED rather than explained.
#
# --gres is the HARD-RULE MIG slice — never any other spec.
#SBATCH --account=def-maxwl
#SBATCH --job-name=candi_t22
#SBATCH --output=slurm-logs/t22_%x_%j.out
#SBATCH --error=slurm-logs/t22_%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

set -uo pipefail
MODE="${1:-train}"

KIT="${KIT:-$HOME/projects/def-maxwl/$USER/CANDII}"
VENV="${VENV:-$HOME/projects/def-maxwl/$USER/candi_venv}"
H5="${H5:-/scratch/$USER/candi_kit/q19.h5}"
OUT="${OUT:-/scratch/$USER/candi_kit/runs_t22}"
SEED="${SEED:-0}"
# TAG defaults to the seed, so a second seed cannot overwrite the first by forgetting a variable.
TAG="${TAG:-t22_on_s$SEED}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg WANDB_MODE=disabled
module load StdEnv/2023 python/3.10.13 >/dev/null 2>&1
source "$VENV/bin/activate" || { echo "[error] no venv at $VENV" >&2; exit 1; }
cd "$KIT"; mkdir -p "$OUT"
source "$KIT/slurm/_kit_pin.sh"
echo "[t22] mode=$MODE tag=$TAG seed=$SEED host=$(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi -L || true

if [ "$MODE" = "train" ]; then
  # --heads count,signal is REQUIRED, and is the ONE departure from slurm/train.sh's recipe: D1's
  # `pval` arm scores the Gaussian signal head against the -log10 p-value track, and a count-only
  # checkpoint has no such head, so half the new suite would be structurally absent. `peak` is left
  # OFF deliberately — it would add a third term to the objective, and the fewer changes to the
  # recorded recipe the better for a report whose whole job is to isolate the MEASUREMENT change.
  # Every eval flag below is the recorded one, so eval.py's side of the report is its standard
  # measurement rather than a variant invented for this comparison.
  python -m candi.train \
    --h5 "$H5" --out-dir "$OUT" \
    --offset on --seed "$SEED" --tag "$TAG" \
    --heads count,signal \
    --weight-decay 0.0 \
    --dsf-sampling uniform --epochs 25 --batch-size 8 --full-coverage \
    --eval-batch-size 4 --eval-max-batches 0 --eval-budget 50000000 --m3-regions 40 \
    --fg-frac 0.02 --n-boot 1000
  rc=$?
elif [ "$MODE" = "bench" ]; then
  # --arch-from rebuilds the EXACT model that wrote the checkpoint from the run's own JSON, so no
  # architecture flag is retyped here and none of them can be retyped wrong.
  # VARPOOL is optional and OFF by default. Without it `msevar` is absent rather than the
  # organizers' bare 0.0, which is the honest outcome and the one this report first shipped with;
  # with it the E-block is nine measures instead of eight. The pool is D7's, built by
  # slurm/t18_varpool.sh from the eic store's own training biosamples, in PVAL space -- so it may
  # only ever weight the pval arm (EVAL_PLAN.md section 9, item 7).
  #   VARPOOL=/scratch/$USER/candi_kit/varpool BENCH_TAG=bench_varpool sbatch ... bench
  BENCH_TAG="${BENCH_TAG:-bench}"
  VARPOOL="${VARPOOL:-}"
  EXTRA=()
  [ -n "$VARPOOL" ] && EXTRA=(--varpool "$VARPOOL" --varpool-corpus "${VARPOOL_CORPUS:-eic}")
  python -m candi.bench \
    --h5 "$H5" --ckpt "$OUT/$TAG.ckpt" --arch-from "$OUT/$TAG.json" \
    --out "$OUT/$TAG.$BENCH_TAG.json" \
    --heads count,signal --kinds impute,denoise --blocks E,P,D,B,C \
    --seed "$SEED" --batch-windows 8 ${EXTRA[@]+"${EXTRA[@]}"}
  rc=$?
else
  echo "[error] mode must be train or bench, got '$MODE'" >&2; exit 2
fi

echo "[t22] DONE mode=$MODE rc=$rc"; exit $rc
