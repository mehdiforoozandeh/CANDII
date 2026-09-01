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
set -uo pipefail

REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_t78_code}"
VENV="${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}"
REGIME="${REGIME:-$REPO/configs/regime.eic_19.json}"
AVO="$REPO/competitors/avocado"

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
_avo_read_regime() {
    python - "$REGIME" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("regions"):
    sys.exit("REGIONS")
print(" ".join(d["train_chroms"]))
print(" ".join(d["eval_chroms"]))
PYEOF
}
_AVO_RG="$(_avo_read_regime)" || {
    echo "[env] $REGIME declares a \`regions\` BED (the eic.pilot Pilot-Region scope, D32)." >&2
    echo "[env] bin_store.py writes one whole-chromosome matrix and train.py trains on all of it;" >&2
    echo "[env] neither can express a BED-restricted training scope, so Avocado CANNOT run this" >&2
    echo "[env] regime without a BED-restricted binner. Raise it -- do not train on the whole" >&2
    echo "[env] chromosome and call it eic.pilot." >&2
    exit 1
}
JOINT_CHROMS=(${_AVO_RG%%$'\n'*})
EVAL_CHROMS=(${_AVO_RG##*$'\n'})
JOINT_CHROM="${JOINT_CHROMS[0]}"
unset _AVO_RG

mkdir -p "$WS/binned" "$WS/ckpt" "$WS/pred" "$WS/logs"
