#!/bin/bash
# sigma-fit then score. CPU: this is store reads and numpy, no model.
#
#   PROTOCOL=p1 sbatch competitors/lavawizard/slurm/score.sh
#   PROTOCOL=p2 sbatch --time=08:00:00 --mem=64G competitors/lavawizard/slurm/score.sh
#
# The sigma-table is fitted ONCE, on the P1 (V-pair) residuals, and reused unchanged by P2 and by
# any later B-pair run (§6.1). Refitting it genome-wide would silently change what the CRPS column
# means between two rows of one table. The MaxRSS on the anchor's scorer came in at 16.3 GB against
# a 16 GB ask, so P1 asks 32 and P2 asks 64.
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
PROTOCOL="${PROTOCOL:-p1}"
PRED="${PRED:-$RUNS/pred}"
SIGMA="${SIGMA:-$RUNS/sigma.json}"
case "$PROTOCOL" in
  p1) CH=() ;;
  p2) CH=(--chroms all) ;;
  *)  echo "[error] PROTOCOL must be p1 or p2, got '$PROTOCOL'" >&2; exit 2 ;;
esac
if [ ! -f "$SIGMA" ]; then
  echo "[sigma] fitting on the P1 panel (eval_chroms), whatever protocol is being scored"
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
     --sigma-table "$SIGMA" --out "$RUNS/scores_${PROTOCOL}.json" "${CH[@]}" "${CRPS[@]}"
