#!/bin/bash
# The naive baseline suite — generate + score on the regime's eval scope (§4 `held-out`).
#
#   mkdir -p slurm-logs && sbatch slurm/t49_baselines_p1.sh
#   REGIME=configs/regime.eic_19.json sbatch slurm/t49_baselines_p1.sh
#
# RETARGETED 2026-08-31 for plan/BENCHMARK_DESIGN.md's two live regimes, and READ THE NEXT
# PARAGRAPH BEFORE LAUNCHING — the §12.2 collapse does NOT hold for all five methods.
#
# §12.2 rules that the five naive baselines run ONCE, not once per regime, "because their fit is
# regime-independent, so is their output: the two regimes would produce byte-identical predictions",
# and asks for an ASSERTION rather than an argument. THE ASSERTION DOES NOT EXIST — there is no test
# in tests/ or in competitors/ that predicts under both regimes and compares — AND THE CODE SAYS THE
# CLAIM IS FALSE FOR THREE OF THE FIVE:
#
#   avg, avg-arcsinh   regime-independent. The contributor set is `biosamples.train` minus the
#                      target's cell type, and the prediction is a per-bin function of the
#                      contributors AT THE PREDICTED POSITION. No training locus enters. Collapse
#                      to one run: correct.
#   knn1, knn5         `generate.similarity_table` correlates over `panel.train_chroms`
#                      (generate.py:207-241). Different train_chroms -> a different similarity
#                      ranking -> different predictions. NOT regime-independent.
#   marginal           `generate.fit_marginal` pools over `panel.train_chroms`
#                      (generate.py:262-300). NOT regime-independent.
#
# So it is 2 runs collapsed and 3 runs x 2 regimes = 8 method-regime units, not 5 — unless the PI
# rules otherwise. And `generate.py` reads `train_chroms` RAW: it has no `regions` support at all,
# so under eic.pilot knn/marginal would fit over 18 WHOLE chromosomes (~2.7 Gbp) instead of the
# 25,588,197 bp the regime declares. That is a Rule 2 break, and it is why REGIME defaults to
# eic_19 here and eic_pilot is refused below.
#
# Generate five prediction roots, score each through `python -m candi.bench.external`, and fold the
# score files into one leaderboard json. Pure numpy + h5py; no GPU compute.
#
# DO_SCORE=0 STOPS AFTER GENERATION, and on a real panel that is usually what you want. Measured on
# Fir, chr21, 45 declared tracks: generating all five roots takes ~4.5 minutes and scoring takes
# ~110 minutes PER METHOD. Five methods at two floors inside one job is eighteen hours of serial
# work for something that is ten independent runs, so generate here and hand the scoring to
# `slurm/t49_baselines_score.sh --array=0-4`, one task per method. Scoring in-line is kept for a
# small panel, where the whole thing is minutes and one job is simpler than three.
#

# WHY IT RUNS TWICE, AT TWO POISSON FLOORS. §5.1 pre-registers `n = 1e6` for the NB's Poisson floor
# and `candi.metrics.nb_crps` returns NaN above about `n = 2e4`, which costs the count arm its whole
# distributional tier — `crps`, `crps_oracle_scaled`, `scale_error`, `beats_marginal` — and with it
# §5.5's second sanity anchor. So both are produced: `spec/` is the pre-registered value and is what
# §5.1 says the baseline IS; `n1e4/` is the largest scoreable floor and is the only one of the two
# that can answer the anchor. Measured against the exact Poisson CRPS the two agree to 0.01 % at
# µ = 0.1 and 0.5 % at µ = 100, and where the floor actually binds they are the same number. Which
# one becomes the row of record is the PI's call, not this script's — see
# `competitors/baselines/README.md`.
#
# WHY --gres ON A CPU-ONLY JOB: see slurm/bake.sh. Without it this routes to def-maxwl_cpu
# (fairshare 0.088 vs the gpu account's 0.435) and effectively never starts. Project hard rule: the
# smallest MIG slice, and never any other gres spec.
#SBATCH --account=def-maxwl
#SBATCH --job-name=candi_t49_p1
#SBATCH --output=slurm-logs/t49_p1_%j.out
#SBATCH --error=slurm-logs/t49_p1_%j.err
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

set -uo pipefail

KIT="${KIT:-/project/def-maxwl/$USER/CANDII_t78_code}"
VENV="${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}"
REGIME="${REGIME:-configs/regime.eic_19.json}"
# The regime's eval_chroms under §4. chr21 alone was the old eval scope.
CHROMS="${CHROMS:-chr20,chr21,chr22}"
OUT="${OUT:-/project/def-maxwl/$USER/t49_baselines/p1}"
METHODS="${METHODS:-avg,avg-arcsinh,knn1,knn5,marginal}"
VARPOOL="${VARPOOL:-}"          # D7 msevar pools; without it msevar is ABSENT, never a bare 0.0
# Which Poisson floors to run. Both by default. `FLOORS=n1e4 METHODS=avg,marginal` with
# `--time=2:30:00` is the b1-bin version that answers §5.5's two sanity anchors and nothing else —
# worth having when the b2 queue is 300 jobs deep and the anchors are what gate the rest of the work.
FLOORS="${FLOORS:-spec n1e4}"
DO_SCORE="${DO_SCORE:-1}"       # 0 = generate only; score with slurm/t49_baselines_score.sh

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export MPLBACKEND=Agg
module load StdEnv/2023 python/3.10.13 >/dev/null 2>&1
source "$VENV/bin/activate" || { echo "[error] no venv at $VENV" >&2; exit 1; }
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
# `_kit_pin.sh` pins `candi` to this checkout's src and proves it. `competitors` is not installed
# anywhere, so the repo root goes on the path too — AFTER src, so the pin's guarantee still holds.
export PYTHONPATH="$KIT/src:$KIT"

# The old guard refused regime.eic_test.json, the separate B-pair regime. THE LIVE REGIMES CARRY
# THE B_ PAIRS INSIDE THEM — eic_19 and eic_pilot each declare 38 eval_pairs, 26 V_ and 12 B_ — so
# a name check protects nothing any more. §5 rules B_ is touched ONCE, at the very end. Derive a
# V_-only regime the way slurm/t81_train_candi.sh does, and refuse a regime that still has B_ in it
# unless this IS the once-only B_ run.
case "$REGIME" in *eic_test*) echo "[t49] REFUSING: $REGIME is the B-pair regime (A4)"; exit 2;; esac
if [ "${ALLOW_B_PAIRS:-0}" != "1" ]; then
  python - "$REGIME" <<'PYEOF' || exit 2
import json, sys
d = json.load(open(sys.argv[1]))
b = [p for p in d.get("eval_pairs", []) if str(p[1]).startswith("B_")]
if b:
    sys.exit(f"[t49] REFUSING: {sys.argv[1]} declares {len(b)} B_ eval pair(s). BENCHMARK_DESIGN "
             f"\u00a75 touches B_ ONCE, from the selected checkpoint. Derive a V_-only regime "
             f"(see slurm/t81_train_candi.sh), or set ALLOW_B_PAIRS=1 if this IS the final B_ run.")
PYEOF
fi
if python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('regions') else 1)" "$REGIME"; then
  echo "[t49] REFUSING: $REGIME declares a \`regions\` BED. competitors/baselines/generate.py reads" >&2
  echo "      train_chroms raw and has no regions support, so knn1/knn5/marginal would fit over" >&2
  echo "      WHOLE chromosomes instead of the Pilot Regions. Rule 2 break. Raise it." >&2
  exit 3
fi
echo "[t49] host=$(hostname) commit=$(git rev-parse --short HEAD) regime=$REGIME chroms=$CHROMS"

rc=0
for FLOOR in $FLOORS; do
    case "$FLOOR" in
        spec) N=1e6 ;;
        n1e4) N=1e4 ;;
    esac
    PRED="$OUT/$FLOOR/preds"
    SCORES="$OUT/$FLOOR/scores"
    mkdir -p "$PRED" "$SCORES"
    echo "=== [t49] floor=$FLOOR poisson_n=$N ==============================================="

    python -m competitors.baselines.generate \
        --store "$REGIME" --out "$PRED" --chroms "$CHROMS" --methods "$METHODS" \
        --poisson-n "$N" || { rc=1; continue; }

    if [ "$DO_SCORE" != "1" ]; then
        echo "[t49] DO_SCORE=0 — generated only; score with slurm/t49_baselines_score.sh"
        continue
    fi

    ARGS=()
    for M in ${METHODS//,/ }; do
        python -m candi.bench.external \
            --store "$REGIME" --pred "$PRED/$M" --out "$SCORES/$M.json" \
            --chroms "$CHROMS" ${VARPOOL:+--varpool "$VARPOOL"} || { rc=1; continue; }
        ARGS+=("$M=$SCORES/$M.json")
    done

    python -m competitors.baselines.leaderboard --protocol P1 \
        --scores "${ARGS[@]}" --out "$OUT/$FLOOR/leaderboard.json" \
        --notes "P1, $REGIME, chroms=$CHROMS, poisson_n=$N ($FLOOR)"
    # NOT --check-anchors: a failed anchor must be REPORTED, and a non-zero exit here would look
    # like a crashed job. §5.5 says stop and report, and the verdicts are in the leaderboard json.
done

echo "[t49] exit=$rc"
find "$OUT" -name '*.json' -maxdepth 3 -exec ls -la {} \; 2>/dev/null | head -30
exit $rc
