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
srun python -u -m candi.bench.external --store "$REGIME" --pred "$PRED" \
     --sigma-table "$SIGMA" --out "$RUNS/scores_${PROTOCOL}.json" "${CH[@]}"
