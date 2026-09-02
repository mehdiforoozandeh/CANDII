#!/bin/bash
# eDICE's σ table, fit on TRAINING residuals. BENCHMARK_DESIGN.md §7 and Rule 1.
#
#   N_TARGETS=31 REGIME=$REPO/configs/regime.eic_19.json    sbatch competitors/edice/slurm/sigma.sh
#   N_TARGETS=31 REGIME=$REPO/configs/regime.eic_pilot.json SIGMA_CHROMS=chr1,chr2 \
#       sbatch --time=12:00:00 competitors/edice/slurm/sigma.sh
#
# WHY THIS JOB EXISTS. eDICE emits a point in -log10 p and no spread, so the pval arm can only be
# scored as a distribution against a σ handed in from outside. Every σ this tree used to produce
# was squared off V_ eval-pair residuals, which Rule 1 forbids and §12.2 declared VOID -- so the
# table is new work and not a flag. A training-residual σ needs predictions on TRAINING tracks,
# which no board prediction root contains, and making them is what the three steps below do:
#
#   1. derive a training regime -- 12 seeded training cells, self-paired (T_x, T_x), scored on the
#      source regime's TRAIN chromosomes. tools/sigma_training_regime.py.
#   2. predict those self-pairs with the SELECTED checkpoint, into the pinned train_pred root.
#      A self-pair's target panel is "every assay the truth cell holds" (harness.StoreSource.
#      targets), so a cell contributes all its own marks and no V_ or B_ track is opened.
#   3. fit: competitors.sigma_pass, which refuses any pair whose target starts V_ or B_ and stamps
#      `fitted_on` with the "training-residuals:" prefix every score stage checks for.
#
# COST. Step 2 writes a FULL-LENGTH array per (track, chromosome): the training grid is the
# regime's train_chroms, so eic_19 is one chromosome (~1.9 h on this slice) and eic_pilot is
# eighteen. A pilot fit is not a whole-genome pass -- narrow it with SIGMA_CHROMS and let the
# regime's own Pilot Regions restrict the residuals inside what is written.
#
# --gres is the AGENTS.md hard-rule MIG slice and is never any other spec.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t81_edice_sigma
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

set -uo pipefail

if [ -z "${N_TARGETS:-}" ]; then
  echo "[error] N_TARGETS is required -- it names which trained model to fit σ from." >&2; exit 2
fi

REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_main}"
VENV="${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}"
REGIME="${REGIME:-$REPO/configs/regime.eic_19.json}"
REGIME_NAME="$(basename "$REGIME" .json)"; REGIME_NAME="${REGIME_NAME#regime.}"
RUNS="${RUNS:-/scratch/mforooz/t81_rivals/eDICE}"
WS="${WS:-$RUNS/${REGIME_NAME}_nt${N_TARGETS}}"
MODEL="${MODEL:-$WS/model.selected.pt}"
# The seeded sample is pinned: 12 cells, seed 890217, the same two numbers for every method, so a
# σ difference between methods is a difference of method and not of which cells were drawn.
N_CELLS="${N_CELLS:-12}"
SIGMA_SEED="${SIGMA_SEED:-890217}"
# Rows per forward pass in step 2. NOT a science knob: the predict pass is `model.eval()` with no
# normalisation layer anywhere in the net and one independent row per bin, so the batch only sets
# how much VRAM one decoder activation costs. run_eic.py's own default of 4096 does not fit the σ
# panel on the MIG slice -- σ is self-paired over 98 declared tracks against V_'s 45, and the
# decoder is 2048 wide, so 4096 x 98 x 2048 x 4 B = 3.06 GiB per activation and the job OOM'd at
# 30 s. 1024 is 0.77 GiB and costs no measurable time, because the pass is bound by the bin count.
PREDICT_BATCH="${PREDICT_BATCH:-1024}"
TRAIN_PRED="${TRAIN_PRED:-/scratch/mforooz/t81_sigma/eDICE/${REGIME_NAME}/train_pred}"
SIGMA_OUT="${SIGMA_OUT:-/project/def-maxwl/mforooz/t81_sigma/eDICE/sigma_${REGIME_NAME}.json}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
source "$VENV/bin/activate"
# The venv's editable install does NOT put `candi` on the path -- its .pth names a checkout that
# carries `candi_kit` instead. $REPO/src is not optional.
export PYTHONPATH="$REPO/competitors/edice:$REPO/src"
echo "[banner] code=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) kit=$REPO"

if [ ! -f "$MODEL" ]; then
  echo "[error] no model at $MODEL -- run eic_train.sh for N_TARGETS=$N_TARGETS first." >&2
  echo "        σ must come from the SELECTED checkpoint: it is the one that gets scored, and a" >&2
  echo "        spread fit from a different epoch describes a different model." >&2
  exit 2
fi

mkdir -p "$WS" "$(dirname "$SIGMA_OUT")"
SIGMA_REGIME="$WS/regime.${REGIME_NAME}.sigma.json"

echo "[edice-σ] regime=$REGIME_NAME  n_targets=$N_TARGETS  n_cells=$N_CELLS  seed=$SIGMA_SEED"
echo "[edice-σ] predict_batch=$PREDICT_BATCH"
echo "[edice-σ] host=$(hostname)"; nvidia-smi -L || true

# --- 1. the training regime ----------------------------------------------------------------------
python "$REPO/tools/sigma_training_regime.py" \
    --regime "$REGIME" --n-cells "$N_CELLS" --seed "$SIGMA_SEED" --out "$SIGMA_REGIME" || exit $?
echo "[edice-σ] training regime: $SIGMA_REGIME"

# What step 2 is about to write, said out loud before it is written. A pilot regime's train_chroms
# are eighteen whole chromosomes and the predict pass writes full-length arrays, so the number
# below is the one that decides whether SIGMA_CHROMS is needed.
python - "$SIGMA_REGIME" <<'PYEOF'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
print(f"[edice-σ] self-pairs: {len(d['eval_pairs'])} | chroms: {d['eval_chroms']}")
if d.get("regions"):
    print(f"[edice-σ] regions: {d['regions']['bed']} — residuals are restricted to it at fit time,")
    print(f"[edice-σ]   but the predict pass still writes whole chromosomes. Narrow SIGMA_CHROMS")
    print(f"[edice-σ]   if {len(d['eval_chroms'])} chromosomes is past the walltime band.")
PYEOF

# --- 2. predictions on the training self-pairs ---------------------------------------------------
CHROMS=()
if [ -n "${SIGMA_CHROMS:-}" ]; then CHROMS=(--chroms ${SIGMA_CHROMS//,/ }); fi

cd "$REPO/competitors/edice"
# Skip a root that already carries a manifest: this pass is hours, and a failed fit downstream is
# not a reason to spend them twice. FORCE_PREDICT=1 overrides.
if [ -z "${FORCE_PREDICT:-}" ] && [ -f "$TRAIN_PRED/manifest.json" ]; then
  echo "[edice-σ] $TRAIN_PRED already has a manifest.json -- reusing it (FORCE_PREDICT=1 to redo)."
else
  python run_eic.py predict \
      --regime "$SIGMA_REGIME" --model "$MODEL" --out "$TRAIN_PRED" \
      --batch-size "$PREDICT_BATCH" "${CHROMS[@]}" || exit $?
fi

# --- 3. the fit ----------------------------------------------------------------------------------
# The chromosomes actually written, read back out of the manifest rather than re-derived, so the
# fit scope is exactly the prediction scope.
FIT_CHROMS=$(python -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['chroms']))" \
               "$TRAIN_PRED/manifest.json") || exit $?
# The regime's own BED, passed explicitly. Redundant if sigma_pass reads `regions` off the regime,
# and load-bearing if it does not -- and under D32 a pilot σ fit outside the Pilot Regions would be
# fit on loci the model never trained on.
EXTRA=()
BED=$(python -c "
import json,sys
d=json.load(open(sys.argv[1]))
print((d.get('regions') or {}).get('bed',''))" "$SIGMA_REGIME") || exit $?
[ -n "$BED" ] && EXTRA+=(--eval-regions "$BED")

cd "$REPO"
python -m competitors.sigma_pass \
    --regime "$SIGMA_REGIME" --pred "$TRAIN_PRED" --out "$SIGMA_OUT" \
    --method eDICE --chroms "$FIT_CHROMS" "${EXTRA[@]}"
rc=$?

if [ $rc -eq 0 ]; then
  python -c "
import json,sys
d=json.load(open(sys.argv[1]))
print('[edice-σ] fitted_on =', d['fitted_on'])
print('[edice-σ] assays    =', len(d['sigma']))
" "$SIGMA_OUT"
fi
echo "[edice-σ] DONE regime=$REGIME_NAME rc=$rc  sigma=$SIGMA_OUT"
exit $rc
