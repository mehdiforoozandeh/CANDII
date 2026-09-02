# Shared environment for every Lavawizard job on Fir. Sourced, never run.
#
# `candi_venv` and not the port's own `torch_env`: `store_eic.py` imports `candi`, and `candi`
# imports `x_transformers`, which `torch_env` does not carry. The Dataset-3 side still runs in
# `torch_env` — see the README's split.
#
# RETARGETED 2026-08-31 for plan/BENCHMARK_DESIGN.md's two live regimes.
#
#   * REGIME was configs/regime.eic_val.json — train chr19, eval chr21, 26 V_ pairs. The live
#     regimes are eic_19 and eic_pilot, and they eval on chr20+chr21+chr22 (§4).
#   * CHROMS was a hard-coded list of all 23. §4 blanks Lavawizard's `genome-wide` cell and rules
#     that a blanked cell is NOT COMPUTED, so the only scope this method is ever predicted or
#     scored on is the regime's eval_chroms — three chromosomes, not 23. The list is now read off
#     the regime, so it cannot drift from it.
#
# THE REGIME AXIS IS REAL AS OF 2026-09-01, AND IT WAS NOT BEFORE.
#
# `train.py` used to fit ONE INDEPENDENT Guacamole PER CHROMOSOME — cell factors, assay factors,
# the dense network and the three position tables, all from a fresh init, all on that chromosome's
# own bins. The cell and assay factors are TRANSFERABLE parameters by BENCHMARK_DESIGN.md §2's own
# definition, so they were fit on chr20/21/22, the chromosomes the method is scored on. That broke
# Rule 2 in BOTH regimes, and it also meant `train_chroms` and `regions` were read by no lavawizard
# module at all: every key this method DID read was identical between eic.19 and eic.pilot, so the
# two rows would have been the same run twice.
#
# PI ruling 2026-09-01: a transferable stage, on Avocado's scheme (§3.2, §12.2).
#
#   STAGE=shared   ONE task. Fits everything on the regime's own training scope — chr19 under
#                  eic.19, the Pilot Regions of eighteen chromosomes under eic.pilot, packed onto
#                  one axis by `store_eic.shared_layout`. Neither scope holds a bin of chr20, 21
#                  or 22. This is the whole regime axis.
#   STAGE=genome   ONE TASK PER EVAL CHROMOSOME (three). Loads the shared half, FREEZES it, and
#                  fits that chromosome's position tables alone — inference under Rule 2, and open
#                  to every method. This is the stage that selects on V_ and the only stage whose
#                  checkpoint may be predicted from.
#
# This is new code, so the board row is OUR TWO-STAGE VARIANT of Lavawizard and not the published
# one. The 2019 submission stays on the board unmodified as one of the 23 anchor entrants, so both
# readings are available — see README.md.
#
# RETARGETED AGAIN 2026-09-01 for the t81 programme. REPO moves off the dead t78 checkout, parked
# on the t77 branch, and onto CANDII_main -- the one clone the programme's banner check reads.
# Every OUTPUT path a job defaults to is now one of the programme's pinned roots and is written
# here rather than in each script, so a root cannot drift between stages.
set -uo pipefail
REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_main}"
VENV="${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}"
RUNS="${RUNS:-/project/def-maxwl/mforooz/rivals_src/lavawizard_runs}"
REGIME="${REGIME:-$REPO/configs/regime.eic_19.json}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
module load StdEnv/2023 python/3.11 >/dev/null 2>&1 || true
source "$VENV/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/competitors"

[ -s "$REGIME" ] || { echo "[env] no regime at $REGIME" >&2; exit 1; }
REGIME_NAME="$(basename "$REGIME" .json)"; REGIME_NAME="${REGIME_NAME#regime.}"
RUNS="$RUNS/$REGIME_NAME"
CACHE="${CACHE:-$RUNS/eic_cache}"

# The eval chromosomes, read off the regime. Plain json — no `import candi`, because this is
# sourced by the caching array too and concurrent torch imports off /project are what the %3 array
# cap exists for.
#
# THE `regions` REFUSAL THAT USED TO LIVE HERE IS GONE, and the transferable stage is why. It
# refused a BED regime whose eval chromosomes were not in its `train_chroms`, because a
# per-chromosome fit could only be fit on the chromosome it predicts and under eic.pilot that meant
# fitting the four Pilot Regions the regime CUT — the complement of the declared scope, wearing its
# name. Nothing is fit on an eval chromosome any more except position tables, so the case the
# refusal guarded cannot arise.
_lava_chroms() {
    python - "$REGIME" <<'PYJSON'
import json, sys
print(" ".join(json.load(open(sys.argv[1]))["eval_chroms"]))
PYJSON
}
CHROMS=($(_lava_chroms)) || exit 1
NCHROM=${#CHROMS[@]}

# The cache stem the shared stage trains on — `store_eic.SHARED_STEM`, never a chromosome name,
# because under eic.pilot the axis is a packing of eighteen of them. Written out here rather than
# read from Python for the same reason as above: this file must not import candi.
SHARED_STEM=shared
METHOD=Lavawizard

mkdir -p "$RUNS" slurm-logs

# -------------------------------------------------------------------------------------------------
# The panel axis, and the programme's pinned roots.
#
# PANEL is V_ or B_ and NOTHING ELSE. BENCHMARK_DESIGN.md §5 rules that B_ is read ONCE, from the
# selected checkpoint, at the very end — so V_ is the default and B_ additionally needs B_ONCE=1
# and a root that does not exist yet. predict.sh holds that guard; the default here only decides
# which panel a stage means when nobody said.
PANEL="${PANEL:-V_}"
case "$PANEL" in
  V_|B_) ;;
  *) echo "[env] PANEL must be V_ or B_, got '$PANEL'" >&2; exit 2 ;;
esac

# Written once, read by predict.sh, score.sh and sigma.sh. Overridable one at a time, but the
# default of every one is a programme root and nothing else writes there.
PRED_V="/scratch/mforooz/t81_pred/$METHOD/$REGIME_NAME/V_"
PRED_B="/project/def-maxwl/mforooz/t81_pred_B/$METHOD/$REGIME_NAME/B_"
PRED_PANEL="$PRED_V"; [ "$PANEL" = "B_" ] && PRED_PANEL="$PRED_B"
TRAIN_PRED="${TRAIN_PRED:-/scratch/mforooz/t81_sigma/$METHOD/$REGIME_NAME/train_pred}"
SIGMA_JSON="/project/def-maxwl/mforooz/t81_sigma/$METHOD/sigma_$REGIME_NAME.json"
SCORES_DIR="/project/def-maxwl/mforooz/t81_scores/$METHOD/$REGIME_NAME"
CKPT_KEEP="${CKPT_KEEP:-/project/def-maxwl/mforooz/t81_checkpoints/$METHOD/$REGIME_NAME}"
TRUTH_CHALLENGE="${TRUTH_CHALLENGE:-/project/def-maxwl/mforooz/t81_truth_challenge/B_}"

# A σ table is accepted only when it says, in its own file, that it was fit on TRAINING residuals.
# BENCHMARK_DESIGN.md §7: "sigma is fit on training-set residuals only — never on V_, never on B_".
# The prefix is the whole contract with `competitors.sigma_pass`; scripts compare against this name
# rather than retyping the string.
SIGMA_FITTED_ON_PREFIX="training-residuals:"

# Derive the PANEL-only regime this stage predicts or scores. The filter lives in ONE tool for every
# method (tools/declare_eval_pairs.py split) so that a panel cannot mean one set of pairs for
# Lavawizard and another for CANDI, and the derived file sits beside the run rather than in
# configs/, where it could drift from the regime it came from.
#
# `store_eic.derive_v_only` still exists and still serves the SELECTOR, which runs inside training
# and must not depend on a file some launcher wrote. This is the launcher-side path, and the two
# agree on what V_ means because both filter on the target's prefix.
PANEL_REGIME="$RUNS/regime.${REGIME_NAME}.${PANEL}.json"
derive_panel_regime() {
    python "$REPO/tools/declare_eval_pairs.py" split \
        --regime "$REGIME" --panel "$PANEL" --out "$PANEL_REGIME" || return 1
    echo "[env] panel regime: $PANEL_REGIME (PANEL=$PANEL)"
}
