#!/bin/bash
# t50 stage 4 -- the sigma-table, then P1 and P2 scoring. One job, three steps, CPU-bound.
#
# This is a job rather than a login-node command for one measured reason: `harness.open_source`
# builds a StoreDataset over the whole corpus and took several minutes on the Fir login node during
# pre-flight. Three stages each pay that cost, and `stream_truth` then reads every bin of every
# scored chromosome for 26 pairs. That is compute, and compute belongs on a compute node.
#
# The MIG slice is requested for the AGENTS.md 3.13 reason only -- nothing here touches the GPU.
#
# WALL IS A MEASURED THING: P1 (chr21, 45 tracks) took 6 min; P2 (genome-wide, the same 45 tracks
# over 121 M bins) reached pair 22 of 26 in 7.8 h before an 8 h wall killed it. Hence 24 h, and
# hence the completeness guard below -- a re-submission must not spend the wall redoing P1.
#
#   mkdir -p slurm-logs && sbatch competitors/avocado/slurm/score.sh
#   P2=0 sbatch competitors/avocado/slurm/score.sh     # P1 only
#   P1=0 sbatch competitors/avocado/slurm/score.sh     # P2 only (P1 already on disk)
#
#SBATCH --account=def-maxwl
#SBATCH --job-name=t50_score
#SBATCH --output=slurm-logs/t50_score_%j.out
#SBATCH --error=slurm-logs/t50_score_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

source /project/def-maxwl/mforooz/CANDII_t50/competitors/avocado/slurm/_env.sh

PRED="${PRED:-$WS/pred/eic_val}"
OUT="${OUT:-$WS/scores}"
SIGMA="${SIGMA:-$OUT/sigma_avocado_v.json}"
mkdir -p "$OUT"
cd "$REPO"

# 1. the 6.1 sigma-table, fitted on V-pair residuals over the regime's P1 chromosomes.
#    Fitted ONCE. A B-pair run must reuse this file unchanged -- that is what makes its CRPS
#    leak-free, and `fitted_on` inside the json is the only thing that records which panel it came
#    from. Do not regenerate it against regime.eic_test.json.
if [ ! -s "$SIGMA" ]; then
    python "$AVO/fit_sigma.py" --regime "$REGIME" --pred "$PRED" --out "$SIGMA" || exit 1
else
    echo "[t50_score] $SIGMA exists, keeping it (a refit would change what CRPS means)"
fi

# A protocol whose result file is already COMPLETE is skipped, so a re-submission after a wall-clock
# kill resumes the chain instead of restarting it. "Complete" is verified, not assumed: the file has
# to parse, cover every declared track, and carry no missing_tracks. A truncated json from a job
# killed mid-write fails all three and is recomputed. This matters because P2 is the expensive pass
# (~10 h genome-wide) and P1 is not -- redoing P1 to reach P2 wastes most of an 8 h wall, which is
# how the first attempt (56847589) ran out of time in the first place.
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

run_protocol() {   # name, out path, extra args...
    local name="$1" out="$2"; shift 2
    if complete "$out" >/dev/null 2>&1; then
        echo "[t50_score] $name: $out is already complete, skipping"
        return 0
    fi
    [ -e "$out" ] && echo "[t50_score] $name: $out exists but is incomplete, recomputing"
    python -m candi.bench.external --store "$REGIME" --pred "$PRED" \
        --sigma-table "$SIGMA" --out "$out" "$@" || return 1
}

# 2. P1 -- declared pairs, the regime's own eval_chroms.
[ "${P1:-1}" != "0" ] && { run_protocol P1 "$OUT/scores_avocado_P1.json" || exit 1; }

# 3. P2 -- the same root, every chromosome the store carries.
[ "${P2:-1}" != "0" ] && {
    run_protocol P2 "$OUT/scores_avocado_P2.json" --chroms "$(paste -sd, "$AVO/chroms.txt")" || exit 1
}
exit 0

echo "[t50_score] done; results in $OUT"
