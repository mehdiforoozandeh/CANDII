#!/bin/bash
# Avocado stage 4 -- score one PANEL against one TRUTH. One job, CPU-bound.
#
# RETARGETED 2026-08-31. `P1`/`P2` are RETIRED (BENCHMARK_DESIGN.md §9): eval scope is now one
# thing for every regime, so there is one scoring pass here -- the regime's eval_chroms -- and the
# genome-wide pass is gone. §4 blanks Avocado's `genome-wide` cell and rules that a blanked cell is
# not computed, so the old P2 branch was computing a number nothing would ever print.
#
# THE SIGMA FIT LEFT THIS SCRIPT 2026-09-01, and that is the point of the change. It used to hold a
# refusal with a Rule-1 override env var, because the only fitter that existed fit on V_ eval-pair
# residuals -- which Rule 1 forbids. There is a real one now, `competitors.sigma_pass`,
# run by `sigma.sh` on TRAINING-track residuals, so the escape hatch is gone with the thing it was
# escaping. What is left is a CHECK: this script reads `fitted_on` out of the σ table it was given
# and refuses anything that does not start with `training-residuals:`. A void σ table and a valid
# one look identical from the outside, so the file has to say which it is.
#
#   TRUTH=store      the CANDI_STORE tracks. The default.
#   TRUTH=challenge  the 2019 blind bigwigs, converted by tools/challenge_bigwigs.py. B_ only --
#                    the challenge published no V_ answer key.
#
# This is a job rather than a login-node command for one measured reason: `harness.open_source`
# builds a StoreDataset over the whole corpus and took several minutes on the Fir login node during
# pre-flight, and `stream_truth` then reads every bin of every scored chromosome. That is compute,
# and compute belongs on a compute node.
#
# The MIG slice is requested for the AGENTS.md 3.13 reason only -- nothing here touches the GPU.
#
# WALL IS A MEASURED THING, AND THE OLD ONE WAS FOR A DIFFERENT SCOPE: the retired P1 (chr21, 45
# tracks) took 6 min, and the retired genome-wide pass reached pair 22 of 26 in 7.8 h before an 8 h
# wall killed it. The pass this script now runs is chr20+21+22 -- 6,478,903 bins, 3.47x chr21 --
# so ~21 min of scorer, not 8 h. The 24 h below is left as headroom; size it down if the queue is
# the constraint.
#
#   mkdir -p slurm-logs && sbatch competitors/avocado/slurm/score.sh
#   mkdir -p slurm-logs && sbatch --export=ALL,PANEL=B_,TRUTH=challenge \
#       competitors/avocado/slurm/score.sh
#
#SBATCH --account=def-maxwl
#SBATCH --job-name=t81_avo_score
#SBATCH --output=slurm-logs/t81_avo_score_%j.out
#SBATCH --error=slurm-logs/t81_avo_score_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

source "${SLURM_SUBMIT_DIR:-$PWD}/competitors/avocado/slurm/_env.sh"

TRUTH="${TRUTH:-store}"
case "$TRUTH" in
  store) ;;
  challenge)
    [ "$PANEL" = "B_" ] || { echo "[avo_score] TRUTH=challenge is B_ only: the challenge published no V_ answer key" >&2; exit 2; }
    [ -s "$TRUTH_CHALLENGE/manifest.json" ] || { echo "[avo_score] no challenge truth root at $TRUTH_CHALLENGE" >&2; exit 2; } ;;
  *) echo "[avo_score] TRUTH must be store or challenge, got '$TRUTH'" >&2; exit 2 ;;
esac

PRED="${PRED:-$PRED_PANEL}"
SIGMA="${SIGMA:-$SIGMA_JSON}"
OUT="${OUT:-$SCORES_DIR/${TRUTH}.${PANEL}.json}"
mkdir -p "$(dirname "$OUT")"
cd "$REPO"

# 1. THE RULE 1 CHECK. Not a fit -- sigma.sh does the fitting, hours earlier and against a
#    different prediction root. All this asks is whether the table in front of it is the
#    training-residual kind. Exit 3 is the code the old refusal used, so anything watching for it
#    still sees it.
if [ ! -s "$SIGMA" ]; then
    echo "[avo_score] no σ table at $SIGMA. Run competitors/avocado/slurm/sigma.sh for" >&2
    echo "[avo_score]   REGIME=$REGIME first -- CRPS has no meaning without one." >&2
    exit 3
fi
python - "$SIGMA" "$SIGMA_FITTED_ON_PREFIX" <<'PYEOF' || exit 3
import json, sys
path, prefix = sys.argv[1], sys.argv[2]
try:
    got = str(json.load(open(path))["fitted_on"])
except Exception as exc:
    sys.exit(f"[avo_score] cannot read `fitted_on` from {path}: {exc}")
if not got.startswith(prefix):
    sys.exit(f"[avo_score] REFUSING {path}: fitted_on = {got!r}, which does not start with "
             f"{prefix!r}. BENCHMARK_DESIGN.md §7 fits σ on training-set residuals only -- never "
             f"on V_, never on B_ -- and §12.2 declares every σ fit any other way VOID. Refit it "
             f"with competitors/avocado/slurm/sigma.sh; there is no override.")
print(f"[avo_score] σ OK: {path} fitted_on {got!r}")
PYEOF

# 2. The panel regime. The prediction root holds this panel's tracks and no others, so scoring it
#    against the shipped 38-pair regime would report twelve missing tracks and take a macro over
#    the wrong denominator. Same tool and same file name as predict.sh, so re-deriving is free.
derive_panel_regime || exit 1

# A protocol whose result file is already COMPLETE is skipped, so a re-submission after a wall-clock
# kill resumes the chain instead of restarting it. "Complete" is verified, not assumed: the file has
# to parse, cover every declared track, and carry no missing_tracks. A truncated json from a job
# killed mid-write fails all three and is recomputed.
complete() {
    [ -s "$1" ] || return 1
    python - "$1" <<'PYEOF'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
    p = r["provenance"]
    ok = (len(r["tracks"]) == p["declared_tracks"]) and not p.get("missing_tracks")
except Exception:
    ok = False
print("complete" if ok else "incomplete")
sys.exit(0 if ok else 1)
PYEOF
}

# 3. The one scoring pass -- this panel's pairs, the regime's own eval_chroms (§4 `held-out`).
#    `--chroms` is passed explicitly rather than left to the scorer's default: §4 blanks Avocado's
#    genome-wide cell, so held-out is the only scope this method is ever scored on, and a launcher
#    that says which scope it means cannot inherit a different one.
CH=(--chroms "$(IFS=,; echo "${EVAL_CHROMS[*]}")")
TR=(); [ "$TRUTH" = "challenge" ] && TR=(--truth-root "$TRUTH_CHALLENGE")

if complete "$OUT" >/dev/null 2>&1; then
    echo "[avo_score] $OUT is already complete, skipping"
    exit 0
fi
[ -e "$OUT" ] && echo "[avo_score] $OUT exists but is incomplete, recomputing"
echo "[avo_score] regime=$REGIME_NAME panel=$PANEL truth=$TRUTH pred=$PRED -> $OUT"
python -m candi.bench.external --store "$PANEL_REGIME" --pred "$PRED" \
    --sigma-table "$SIGMA" --out "$OUT" "${CH[@]}" "${TR[@]}" || exit 1
echo "[avo_score] done; $OUT"
exit 0
