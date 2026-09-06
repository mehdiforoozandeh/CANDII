#!/bin/bash
# Blind-set (B_) preview: whole-genome eDICE prediction from the checkpoint that already exists.
#
#   sbatch competitors/edice/slurm/blind_predict.sh
#   BP_OUT=/somewhere/else sbatch competitors/edice/slurm/blind_predict.sh
#
# THROWAWAY, PI-AUTHORISED (2026-09-01). BENCHMARK_DESIGN.md §13.3 rules that the blind set is
# touched ONCE, at the very end; `competitors/chromimpute/slurm/submit.sh` repeats that as its
# blocker #2. The PI has overridden it knowingly for a side-by-side preview. Nothing produced here
# is scored, and every array is regenerated from retrained models later.
#
# No retraining and no scoring: this runs `predict` only, against
# `rivals_src/edice_runs/eic_nt31/model.pt` (n_targets=31, seed 211, trained on chr19 under
# `regime.eic_val.json`). The regime below is `regime.eic_test.json`, whose `biosamples.train` and
# `assays` are identical to the val regime's, so the checkpoint's vocabulary matches; `cmd_predict`
# refuses the checkpoint otherwise, and `eic_panel.assert_no_eval_leakage` refuses any panel
# carrying a V_ or B_ biosample before the checkpoint is even loaded.
#
# NO RESUME. `cmd_predict` writes one npz per (track, chrom) as it goes, but it recomputes
# everything from the start on a restart — there is no skip-if-present path in `run_eic.py`
# itself (`eic_score.sh`'s "COMPLETE, skipping predict" check is in that wrapper, not here, and it
# is all-or-nothing anyway). A failed or timed-out job is simply resubmitted; a partial root is
# redone, never resumed, because a half-written npz is exactly what the §4.1 grid assertion exists
# to catch.
#
# WALLTIME. ~6.9 GPU-h expected (the earlier genome-wide P2 pass over 45 tracks took 6.1 h; this is
# 51 tracks over the same 23 chromosomes). 16 h is asked because that earlier job hit its 12 h
# wall. 16 h lands in the b3 walltime bin (≤24 h) — do NOT add `--partition`, the plugin picks it
# from `--time`.
#
# The `jobs/` directory below must already exist at submission time — SLURM will not create the
# log directory, and a missing one fails the submission. The launch step creates
# $BP_OUT/{eDICE,jobs,...} before calling sbatch.
#
# --gres is the AGENTS.md hard-rule MIG slice and is never any other spec.
#SBATCH --account=def-maxwl
#SBATCH --job-name=bp_edice_predict
#SBATCH --output=/project/def-maxwl/mforooz/blind_preview_2026-09-01/jobs/%x_%j.out
#SBATCH --error=/project/def-maxwl/mforooz/blind_preview_2026-09-01/jobs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

set -euo pipefail

REPO=/project/def-maxwl/mforooz/CANDII_t52
VENV=/project/def-maxwl/mforooz/candi_venv
REGIME=$REPO/configs/regime.eic_test.json
MODEL=/project/def-maxwl/mforooz/rivals_src/edice_runs/eic_nt31/model.pt
OUTROOT=${BP_OUT:-/project/def-maxwl/mforooz/blind_preview_2026-09-01}
PRED=$OUTROOT/eDICE

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
source "$VENV/bin/activate"

cd "$REPO/competitors/edice"
export PYTHONPATH="$PWD:$REPO/src"

# The reviewed code is one commit. A different HEAD is a different experiment, so refuse rather
# than quietly predict with something else.
HEAD_SHA=$(git -C "$REPO" rev-parse --short HEAD)
if [ "$HEAD_SHA" != "04be648" ]; then
  echo "[error] $REPO is at $HEAD_SHA, expected 04be648" >&2
  exit 2
fi

if [ ! -f "$MODEL" ]; then
  echo "[error] no checkpoint at $MODEL" >&2
  exit 2
fi

echo "[bp-edice] host=$(hostname)  repo=$HEAD_SHA"
echo "[bp-edice] regime=$REGIME"
echo "[bp-edice] model=$MODEL"
echo "[bp-edice] out=$PRED  chroms=all"
nvidia-smi -L || true

python run_eic.py predict \
  --regime "$REGIME" \
  --model "$MODEL" \
  --out "$PRED" \
  --chroms all
rc=$?

echo "[bp-edice] DONE rc=$rc  pred=$PRED"
exit $rc
