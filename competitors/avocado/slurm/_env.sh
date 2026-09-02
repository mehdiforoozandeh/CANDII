# Sourced by every Avocado job. Paths first, so a job that runs somewhere else fails at the source
# line rather than three stages later inside python.
#
# WS is /scratch on purpose: the binned matrices and the per-chromosome checkpoints are tens of GB
# and none of it is a result. What comes back to /project is small -- the training logs, the scores,
# the sigma table. /scratch purges after 60 days; anything still wanted then must be re-derived,
# which is why the commands are all in the README rather than in anybody's history.
#
# RETARGETED 2026-08-31 for plan/BENCHMARK_DESIGN.md's two live regimes. Three things changed and
# each one was a stale value, not a preference:
#
#   * REGIME was configs/regime.eic_val.json -- train chr19, eval chr21, 26 V_ pairs. The live
#     regimes are configs/regime.eic_19.json and configs/regime.eic_pilot.json, and they eval on
#     chr20+chr21+chr22 (§4).
#   * The chromosome lists are now READ OFF THE REGIME instead of competitors/avocado/chroms.txt.
#     §3.2 moves the joint (shared-parameter) fit to the regime's train_chroms, and §4's blanking
#     ruling means Avocado is only ever predicted on the regime's eval_chroms -- three of them, not
#     23 (§12.2, corrected). chroms.txt stays on disk because bin_store.py can still be pointed at
#     any single chromosome by hand.
#   * WS is keyed by the regime name, so the two regimes cannot overwrite each other's matrices.
#
# RETARGETED AGAIN 2026-09-01 for the t81 programme. REPO moves off the dead t78 checkout, parked
# on the t77 branch, and onto CANDII_main -- the one clone the programme's banner check reads.
# Every OUTPUT path a job defaults to is now one of the programme's pinned roots and is written
# here rather than in each script, so a root cannot drift between stages.
set -uo pipefail

REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_main}"
VENV="${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}"
REGIME="${REGIME:-$REPO/configs/regime.eic_19.json}"
AVO="$REPO/competitors/avocado"
METHOD=Avocado

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg

[ -d "$VENV" ] || { echo "[env] no venv at $VENV" >&2; exit 1; }
source "$VENV/bin/activate"
[ -d "$AVO" ] || { echo "[env] no competitors/avocado at $AVO" >&2; exit 1; }
[ -s "$REGIME" ] || { echo "[env] no regime at $REGIME" >&2; exit 1; }

# Regime name, for the workspace key: configs/regime.eic_19.json -> eic_19.
REGIME_NAME="$(basename "$REGIME" .json)"; REGIME_NAME="${REGIME_NAME#regime.}"
WS="${WS:-/scratch/$USER/t81_avocado/$REGIME_NAME}"

# The two chromosome lists, read off the regime. Plain json -- no `import candi`, because this is
# sourced by the binning array too and forty concurrent torch imports off /project do not work
# (see the %12 caps on the other rivals' arrays).
#
# JOINT_CHROMS is `train_chroms`: where the shared cell/assay/network parameters are fit (§3.2).
# EVAL_CHROMS is `eval_chroms`: the only place Avocado is ever predicted (§4's blanking ruling), so
# it is also the complete list of per-chromosome genomic-factor fits (§12.2: three, not 23).
#
# SHARED_SCOPE names what the shared fit trains on, and it is the third thing the regime decides.
# Without a `regions` block that is one whole chromosome, `$JOINT_CHROM`. With one it is the string
# `regions`: the D32 BED scope over EVERY train chromosome, packed onto one compact axis by
# `bin_store.py --regions`. §3.2 asks for exactly that -- "under eic.pilot Avocado's stage 1 returns
# to the Pilot Regions" -- and it is why the whole-chromosome refusal that stood here until
# 2026-08-31 is gone: the support is real now, and tests/test_avocado_regions.py holds it to the BED.
_avo_read_regime() {
    python - "$REGIME" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(" ".join(d["train_chroms"]))
print(" ".join(d["eval_chroms"]))
print("regions" if d.get("regions") else "")
PYEOF
}
_AVO_RG="$(_avo_read_regime)" || { echo "[env] cannot read $REGIME" >&2; exit 1; }
# `read` three times rather than `mapfile`: bash 3.2 has no mapfile, and this file is also read by
# hand on machines that are not Fir.
{ read -r _AVO_TRAIN; read -r _AVO_EVAL; read -r _AVO_SCOPE; } <<< "$_AVO_RG"
JOINT_CHROMS=($_AVO_TRAIN)
EVAL_CHROMS=($_AVO_EVAL)
JOINT_CHROM="${JOINT_CHROMS[0]}"
SHARED_SCOPE="${_AVO_SCOPE:-$JOINT_CHROM}"
unset _AVO_RG _AVO_TRAIN _AVO_EVAL _AVO_SCOPE

mkdir -p "$WS/binned" "$WS/ckpt" "$WS/pred" "$WS/logs"

# -------------------------------------------------------------------------------------------------
# The panel axis, and the programme's pinned roots.
#
# PANEL is V_ or B_ and NOTHING ELSE. BENCHMARK_DESIGN.md §5 rules that B_ is read ONCE, from the
# selected checkpoint, at the very end -- so V_ is the default and B_ additionally needs B_ONCE=1
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
# BENCHMARK_DESIGN.md §7: "sigma is fit on training-set residuals only -- never on V_, never on B_".
# The prefix is the whole contract with `competitors.sigma_pass`; scripts compare against this name
# rather than retyping the string.
SIGMA_FITTED_ON_PREFIX="training-residuals:"

# Derive the PANEL-only regime this stage predicts or scores. The filter lives in ONE tool for
# every method (tools/declare_eval_pairs.py split) so that a panel cannot mean one set of pairs for
# Avocado and another for CANDI, and the derived file sits beside the run rather than in configs/,
# where it could drift from the regime it came from.
#
# `regions.bed` resolves against the regime file's own directory (store/regime.py:52), which is why
# the tool rewrites it absolute -- a derived copy in $WS would otherwise fail the pilot regime's
# sha256 check.
PANEL_REGIME="$WS/regime.${REGIME_NAME}.${PANEL}.json"
derive_panel_regime() {
    python "$REPO/tools/declare_eval_pairs.py" split \
        --regime "$REGIME" --panel "$PANEL" --out "$PANEL_REGIME" || return 1
    echo "[env] panel regime: $PANEL_REGIME (PANEL=$PANEL)"
}
