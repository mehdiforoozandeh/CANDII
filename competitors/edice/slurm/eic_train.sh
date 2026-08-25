#!/bin/bash
# eDICE retrained on our EIC (RIVALS_PLAN §7.3), after the Roadmap gate is recorded.
#
# DECIDED (PI, 2026-08-25): N_TARGETS=31 — the paper's 11.6% RATE, not its absolute 120.
# eDICE masked 120 of Roadmap's 1032 tracks per bin; our panel holds 267, so the absolute count
# would mask 45% and make our variant structurally harder than the published one. The flag stays
# REQUIRED even though the value is settled, so the number appears on the launch line and in this
# job's log — which is where the record of what was actually run lives.
#
#   mkdir -p slurm-logs
#   N_TARGETS=31  sbatch --time=24:00:00 competitors/edice/slurm/eic_train.sh   # THE DECIDED RUN
#   N_TARGETS=120 sbatch --time=48:00:00 competitors/edice/slurm/eic_train.sh   # absolute reading
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
  echo "        31  = eDICE's Roadmap masking RATE (11.6% of our 267-track panel)  <-- DECIDED," >&2
  echo "              PI 2026-08-25. This is the run to launch." >&2
  echo "        120 = eDICE's Roadmap ABSOLUTE count (masks 45% of our panel)" >&2
  echo "        See competitors/edice/README.md, 'Masking rate on our EIC'." >&2
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
