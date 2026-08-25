#!/bin/bash
# eDICE retrained on our EIC (RIVALS_PLAN §7.3), after the Roadmap gate is recorded.
#
# N_TARGETS IS REQUIRED AND HAS NO DEFAULT. That is the point: eDICE masked 120 of Roadmap's 1032
# tracks per bin (11.6%), and our training panel holds 267, so the same ABSOLUTE count masks 45%
# while the same RATE is 31. "Paper defaults" (§6.3) does not settle it, and it changes the training
# signal rather than tuning it, so it is pre-registered by the PI, never defaulted by a script.
#
#   mkdir -p slurm-logs
#   N_TARGETS=31  sbatch --time=24:00:00 competitors/edice/slurm/eic_train.sh   # the RATE reading
#   N_TARGETS=120 sbatch --time=48:00:00 competitors/edice/slurm/eic_train.sh   # the ABSOLUTE reading
#
# Wall: PROJECTED, not measured. The Roadmap gate measured 0.255 s/batch at 120 targets over a
# 1032-track panel on this MIG slice. chr19 gives 2,344,704 bins = 9159 batches/epoch, and cost is
# dominated by batch x n_targets through the decoder, so 31 targets projects to ~10 h and 120 to
# ~30 h over 50 epochs. Set --time on the command line to match the reading you chose; the header's
# value below is only the floor for the b1 bin and WILL be too short for either run.
#
# --gres is the AGENTS.md hard-rule MIG slice and is never any other spec.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t52_edice_eic
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

set -uo pipefail

if [ -z "${N_TARGETS:-}" ]; then
  echo "[error] N_TARGETS is required and has no default." >&2
  echo "        31  = eDICE's Roadmap masking RATE (11.6% of our 267-track panel)" >&2
  echo "        120 = eDICE's Roadmap ABSOLUTE count (masks 45% of our panel)" >&2
  echo "        This is the PI's pre-registered call, not a knob. See competitors/edice/README.md." >&2
  exit 2
fi

REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_t52}"
VENV="${VENV:-/project/def-maxwl/mforooz/candi_venv}"
REGIME="${REGIME:-$REPO/configs/regime.eic_val.json}"
OUT="${OUT:-/project/def-maxwl/mforooz/rivals_src/edice_runs/eic_nt${N_TARGETS}}"
EPOCHS="${EPOCHS:-50}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
source "$VENV/bin/activate"

cd "$REPO/competitors/edice"
export PYTHONPATH="$PWD:$REPO/src"
mkdir -p "$OUT"

echo "[edice] EIC train  n_targets=$N_TARGETS  epochs=$EPOCHS  regime=$REGIME  out=$OUT"
echo "[edice] host=$(hostname)"; nvidia-smi -L || true

# --train-chroms defaults to the regime's train_chroms (chr19), matching CANDI's own recipe.
# eDICE carries no positional parameters, so this is a budget choice and not a leakage one -- see
# the README's "The one open decision".
python run_eic.py train \
  --regime "$REGIME" --out "$OUT" \
  --n-targets "$N_TARGETS" --epochs "$EPOCHS"
rc=$?

echo "[edice] DONE eic-train n_targets=$N_TARGETS rc=$rc"
exit $rc
