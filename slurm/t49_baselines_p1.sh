#!/bin/bash
# t49 / RIVALS_PLAN.md §5 — the naive baseline suite, protocol P1 (declared pairs, eval chromosome).
#
#   mkdir -p slurm-logs && sbatch slurm/t49_baselines_p1.sh
#   REGIME=configs/regime.eic_val.json CHROMS=chr21 sbatch slurm/t49_baselines_p1.sh
#
# Generate five prediction roots, score each through `python -m candi.bench.external`, and fold the
# score files into one leaderboard json. Pure numpy + h5py; no GPU compute.
#
# NEVER POINT THIS AT regime.eic_test.json. B pairs are run ONCE per method at the very end (A4),
# and this script is the development loop.
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

KIT="${KIT:-/project/def-maxwl/$USER/CANDII_t49}"
VENV="${VENV:-/project/def-maxwl/$USER/candi_venv}"
REGIME="${REGIME:-configs/regime.eic_val.json}"
CHROMS="${CHROMS:-chr21}"
OUT="${OUT:-/project/def-maxwl/$USER/t49_baselines/p1}"
METHODS="${METHODS:-avg,avg-arcsinh,knn1,knn5,marginal}"
VARPOOL="${VARPOOL:-}"          # D7 msevar pools; without it msevar is ABSENT, never a bare 0.0
# Which Poisson floors to run. Both by default. `FLOORS=n1e4 METHODS=avg,marginal` with
# `--time=2:30:00` is the b1-bin version that answers §5.5's two sanity anchors and nothing else —
# worth having when the b2 queue is 300 jobs deep and the anchors are what gate the rest of the work.
FLOORS="${FLOORS:-spec n1e4}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export MPLBACKEND=Agg
module load StdEnv/2023 python/3.10.13 >/dev/null 2>&1
source "$VENV/bin/activate" || { echo "[error] no venv at $VENV" >&2; exit 1; }
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
# `_kit_pin.sh` pins `candi` to this checkout's src and proves it. `competitors` is not installed
# anywhere, so the repo root goes on the path too — AFTER src, so the pin's guarantee still holds.
export PYTHONPATH="$KIT/src:$KIT"

case "$REGIME" in *eic_test*) echo "[t49] REFUSING: $REGIME is the B-pair regime (A4)"; exit 2;; esac
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
