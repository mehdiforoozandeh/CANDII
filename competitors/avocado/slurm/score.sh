#!/bin/bash
# Avocado stage 4 -- the sigma-table, then scoring. One job, CPU-bound.
#
# RETARGETED 2026-08-31. `P1`/`P2` are RETIRED (BENCHMARK_DESIGN.md §9): eval scope is now one
# thing for every regime, so there is one scoring pass here -- the regime's eval_chroms -- and the
# genome-wide pass is gone. §4 blanks Avocado's `genome-wide` cell and rules that a blanked cell is
# not computed, so the old P2 branch was computing a number nothing would ever print.
#
# This is a job rather than a login-node command for one measured reason: `harness.open_source`
# builds a StoreDataset over the whole corpus and took several minutes on the Fir login node during
# pre-flight. Three stages each pay that cost, and `stream_truth` then reads every bin of every
# scored chromosome for 26 pairs. That is compute, and compute belongs on a compute node.
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

PRED="${PRED:-$WS/pred}"
OUT="${OUT:-$WS/scores}"
SIGMA="${SIGMA:-$OUT/sigma_avocado_${REGIME_NAME}.json}"
mkdir -p "$OUT"
cd "$REPO"

# 1. the sigma-table.
#
#    STOP. `fit_sigma.py` FITS ON V_ EVAL-PAIR RESIDUALS, and BENCHMARK_DESIGN.md §7 rules that
#    "sigma is fit on training-set residuals only -- never on V_, never on B_". §12.2 says every
#    existing sigma is VOID under Rule 1 and must refit on TRAINING residuals. Fitting one here
#    would read the scored track, which Rule 1 forbids at every stage including a sigma-fit.
#
#    This is not a flag to flip: the prediction root holds only the declared eval tracks, so there
#    are no training-track predictions to take a residual against. Producing them is new work.
#    Refuse rather than write a void table that looks like a valid one.
if [ ! -s "$SIGMA" ]; then
    if [ "${SIGMA_RULE1_OVERRIDE:-0}" != "1" ]; then
        echo "[avo_score] REFUSING to fit $SIGMA: fit_sigma.py fits on V_ eval-pair residuals," >&2
        echo "[avo_score]   which BENCHMARK_DESIGN.md Rule 1 forbids and §7 rules out explicitly." >&2
        echo "[avo_score]   A training-residual sigma needs predictions on TRAINING tracks, which" >&2
        echo "[avo_score]   this prediction root does not contain. Raise it; do not override." >&2
        exit 3
    fi
    python "$AVO/fit_sigma.py" --regime "$REGIME" --pred "$PRED" --out "$SIGMA" || exit 1
else
    echo "[avo_score] $SIGMA exists, keeping it (a refit would change what CRPS means)"
fi

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

score_scope() {   # name, out path, extra args...
    local name="$1" out="$2"; shift 2
    if complete "$out" >/dev/null 2>&1; then
        echo "[avo_score] $name: $out is already complete, skipping"
        return 0
    fi
    [ -e "$out" ] && echo "[avo_score] $name: $out exists but is incomplete, recomputing"
    python -m candi.bench.external --store "$REGIME" --pred "$PRED" \
        --sigma-table "$SIGMA" --out "$out" "$@" || return 1
}

# 2. the one scoring pass -- declared pairs, the regime's own eval_chroms (§4 `held-out`).
score_scope held-out "$OUT/scores_avocado_${REGIME_NAME}_heldout.json" || exit 1
echo "[avo_score] done; results in $OUT"
exit 0
