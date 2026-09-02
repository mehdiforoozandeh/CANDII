#!/bin/bash
# Score one PANEL against one TRUTH. CPU: this is store reads and numpy, no model.
#
#   sbatch competitors/lavawizard/slurm/score.sh
#   sbatch --export=ALL,PANEL=B_,TRUTH=challenge competitors/lavawizard/slurm/score.sh
#
# RETARGETED 2026-08-31. `P1`/`P2` are RETIRED (BENCHMARK_DESIGN.md §9) and the genome-wide pass is
# gone: §4 blanks Lavawizard's `genome-wide` cell and rules that a blanked cell is not computed, so
# there is one pass — the regime's eval_chroms. The MaxRSS on the anchor's chr21 scorer came in at
# 16.3 GB against a 16 GB ask; chr20+21+22 is 3.47x chr21, so 32 G is the right ask and the wall is
# ~20 minutes of scorer, not hours.
#
# THE SIGMA FIT LEFT THIS SCRIPT 2026-09-01. It used to hold a refusal with a Rule-1 override env
# var, because the only fitter that existed — the old per-method one — fits on V_ eval-pair
# residuals, which §7 forbids and §12.2 declares VOID. There is a real fitter now,
# `competitors.sigma_pass`, run by `sigma.sh` on TRAINING-track residuals, so the hatch is gone
# with the thing it was escaping. What is left is a CHECK: read `fitted_on` out of the table and
# refuse anything not starting with `training-residuals:`. A void σ table and a valid one look
# identical from the outside, so the file has to say which it is.
#
#   TRUTH=store      the CANDI_STORE tracks. The default.
#   TRUTH=challenge  the 2019 blind bigwigs, converted by tools/challenge_bigwigs.py. B_ only —
#                    the challenge published no V_ answer key.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t53_score
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# fc30560 gave repeated Lustre OSError 108 on files its neighbours read fine (anchor run).
#SBATCH --exclude=fc30560
# `$0` is sbatch's spool copy, not this file, so the path comes from the submit directory.
ENV="${SLURM_SUBMIT_DIR:-$PWD}/competitors/lavawizard/slurm/_env.sh"
[ -f "$ENV" ] || { echo "[error] no $ENV -- submit from the repo root" >&2; exit 2; }
source "$ENV"

TRUTH="${TRUTH:-store}"
case "$TRUTH" in
  store) ;;
  challenge)
    [ "$PANEL" = "B_" ] || { echo "[score] TRUTH=challenge is B_ only: the challenge published no V_ answer key" >&2; exit 2; }
    [ -s "$TRUTH_CHALLENGE/manifest.json" ] || { echo "[score] no challenge truth root at $TRUTH_CHALLENGE" >&2; exit 2; } ;;
  *) echo "[score] TRUTH must be store or challenge, got '$TRUTH'" >&2; exit 2 ;;
esac

PRED="${PRED:-$PRED_PANEL}"
SIGMA="${SIGMA:-$SIGMA_JSON}"
OUT="${OUT:-$SCORES_DIR/${TRUTH}.${PANEL}.json}"
mkdir -p "$(dirname "$OUT")"

# `--chroms` is passed explicitly rather than left to the scorer's default: §4 blanks Lavawizard's
# genome-wide cell, so held-out is the only scope this method is ever scored on, and a launcher that
# says which scope it means cannot inherit a different one.
CH=(--chroms "$(IFS=,; echo "${CHROMS[*]}")")
TR=(); [ "$TRUTH" = "challenge" ] && TR=(--truth-root "$TRUTH_CHALLENGE")

# THE RULE 1 CHECK. Not a fit — sigma.sh does the fitting, against a different prediction root. All
# this asks is whether the table in front of it is the training-residual kind. Exit 3 is the code
# the old refusal used, so anything watching for it still sees it.
if [ ! -s "$SIGMA" ]; then
  echo "[score] no σ table at $SIGMA. Run competitors/lavawizard/slurm/sigma.sh for" >&2
  echo "[score]   REGIME=$REGIME first — CRPS has no meaning without one." >&2
  exit 3
fi
python - "$SIGMA" "$SIGMA_FITTED_ON_PREFIX" <<'PYEOF' || exit 3
import json, sys
path, prefix = sys.argv[1], sys.argv[2]
try:
    got = str(json.load(open(path))["fitted_on"])
except Exception as exc:
    sys.exit(f"[score] cannot read `fitted_on` from {path}: {exc}")
if not got.startswith(prefix):
    sys.exit(f"[score] REFUSING {path}: fitted_on = {got!r}, which does not start with {prefix!r}. "
             f"BENCHMARK_DESIGN.md §7 fits σ on training-set residuals only — never on V_, never "
             f"on B_ — and §12.2 declares every σ fit any other way VOID. Refit it with "
             f"competitors/lavawizard/slurm/sigma.sh; there is no override.")
print(f"[score] σ OK: {path} fitted_on {got!r}")
PYEOF

# The panel regime. The prediction root holds this panel's tracks and no others, so scoring it
# against the shipped 38-pair regime would report twelve missing tracks and take a macro over the
# wrong denominator. Same tool and same file name as predict.sh, so re-deriving is free.
derive_panel_regime || exit 1

# The PI ruled sampled CRPS at k=100 for every method. It changes the COUNT arm only
# (`candi.metrics.nb_crps_sampled`), and Lavawizard has no count arm — B1b forbids inventing a read
# depth — so for this method the flag is a no-op that exists to keep one command across the rivals.
# It is passed only when the installed entry point accepts it; `--crps-approx` landed on t56 and is
# not on `main` yet, and a hard-coded flag would make this script fail on a checkout without it.
CRPS=()
if python -m candi.bench.external --help 2>&1 | grep -q -- "--crps-approx"; then
  CRPS=(--crps-approx "${CRPS_K:-100}" --crps-seed "${CRPS_SEED:-0}")
  echo "[score] sampled CRPS ${CRPS[*]} (count arm only; this method is pval-only)"
else
  echo "[score] this candi.bench.external has no --crps-approx; pval-only root, nothing to change"
fi

echo "[score] regime=$REGIME_NAME panel=$PANEL truth=$TRUTH pred=$PRED -> $OUT"
srun python -u -m candi.bench.external --store "$PANEL_REGIME" --pred "$PRED" \
     --sigma-table "$SIGMA" --out "$OUT" "${CH[@]}" "${TR[@]}" "${CRPS[@]}"
