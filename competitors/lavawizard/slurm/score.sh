#!/bin/bash
# sigma-fit then score. CPU: this is store reads and numpy, no model.
#
#   sbatch competitors/lavawizard/slurm/score.sh
#
# RETARGETED 2026-08-31. `P1`/`P2` are RETIRED (BENCHMARK_DESIGN.md §9) and the genome-wide pass is
# gone: §4 blanks Lavawizard's `genome-wide` cell and rules that a blanked cell is not computed, so
# there is one pass — the regime's eval_chroms. The MaxRSS on the anchor's chr21 scorer came in at
# 16.3 GB against a 16 GB ask; chr20+21+22 is 3.47x chr21, so 32 G is the right ask and the wall is
# ~20 minutes of scorer, not hours.
#
# THE SIGMA FIT IS REFUSED, AND THAT IS THE RULING, NOT A BUG. `fit_sigma.py` fits on V_ eval-pair
# residuals. §7 rules "sigma is fit on training-set residuals only — never on V_, never on B_", and
# §12.2 declares every existing sigma VOID under Rule 1. A training-residual sigma needs predictions
# on TRAINING tracks, which this prediction root does not contain, so it is new work rather than a
# flag. Refuse rather than write a void table that looks like a valid one.
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
PRED="${PRED:-$RUNS/pred}"
SIGMA="${SIGMA:-$RUNS/sigma.json}"
CH=(--chroms "$(IFS=,; echo "${CHROMS[*]}")")
if [ ! -f "$SIGMA" ]; then
  if [ "${SIGMA_RULE1_OVERRIDE:-0}" != "1" ]; then
    echo "[sigma] REFUSING to fit $SIGMA: fit_sigma.py fits on V_ eval-pair residuals, which" >&2
    echo "[sigma]   BENCHMARK_DESIGN.md Rule 1 forbids and §7 rules out explicitly. A training-" >&2
    echo "[sigma]   residual sigma needs predictions on TRAINING tracks, which this root does not" >&2
    echo "[sigma]   contain. Raise it; do not override." >&2
    exit 3
  fi
  srun python -u competitors/lavawizard/fit_sigma.py --regime "$REGIME" --pred "$PRED" \
       --out "$SIGMA"
fi
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

srun python -u -m candi.bench.external --store "$REGIME" --pred "$PRED" \
     --sigma-table "$SIGMA" --out "$RUNS/scores_${REGIME_NAME}_heldout.json" "${CH[@]}" "${CRPS[@]}"
