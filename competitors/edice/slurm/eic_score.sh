#!/bin/bash
# predict -> σ-fit -> score, in one job. Runs after eic_train.sh for the same N_TARGETS.
#
#   N_TARGETS=31 PROTOCOL=p1 sbatch --time=03:00:00 competitors/edice/slurm/eic_score.sh
#   N_TARGETS=31 PROTOCOL=p2 sbatch --time=12:00:00 --mem=96G competitors/edice/slurm/eic_score.sh
#
# P1 = the regime's eval chroms (chr21, 1,868,399 bins). P2 = every chromosome the store carries,
# which is ~66x the bins, ~66x the store reads, and ~22 GB of npz -- hence the bigger ask.
#
# The σ-table is fitted ONCE, on the P1 (V-pair) residuals, and is reused unchanged by any later
# B-pair run (§6.1). P2 therefore reuses the P1 table rather than fitting its own: refitting σ on a
# genome-wide pass would silently change what the CRPS column means between two rows of the same
# table.
#
# --gres is the AGENTS.md hard-rule MIG slice and is never any other spec.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t52_edice_score
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

set -uo pipefail

if [ -z "${N_TARGETS:-}" ]; then
  echo "[error] N_TARGETS is required -- it names which trained model to score." >&2; exit 2
fi
PROTOCOL="${PROTOCOL:-p1}"

REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_t52}"
VENV="${VENV:-/project/def-maxwl/mforooz/candi_venv}"
REGIME="${REGIME:-$REPO/configs/regime.eic_val.json}"
RUNS="${RUNS:-/project/def-maxwl/mforooz/rivals_src/edice_runs}"
MODEL="${MODEL:-$RUNS/eic_nt${N_TARGETS}/model.pt}"
PRED="${PRED:-$RUNS/eic_nt${N_TARGETS}/preds_${PROTOCOL}}"
# The σ-table lives with the P1 run whatever protocol is being scored -- see the header.
SIGMA="${SIGMA:-$RUNS/eic_nt${N_TARGETS}/preds_p1/sigma.json}"
SCORES="${SCORES:-$RUNS/eic_nt${N_TARGETS}/scores_${PROTOCOL}.json}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
source "$VENV/bin/activate"

cd "$REPO/competitors/edice"
export PYTHONPATH="$PWD:$REPO/src"

case "$PROTOCOL" in
  p1) CHROMS=() ;;                       # default: the regime's eval_chroms
  p2) CHROMS=(--chroms all) ;;
  *)  echo "[error] PROTOCOL must be p1 or p2, got '$PROTOCOL'" >&2; exit 2 ;;
esac

if [ ! -f "$MODEL" ]; then
  echo "[error] no model at $MODEL -- run eic_train.sh for N_TARGETS=$N_TARGETS first" >&2; exit 2
fi

echo "[edice] EIC $PROTOCOL  n_targets=$N_TARGETS  model=$MODEL  pred=$PRED"
echo "[edice] host=$(hostname)"; nvidia-smi -L || true

python run_eic.py predict \
  --regime "$REGIME" --model "$MODEL" --out "$PRED" "${CHROMS[@]}" || exit $?

if [ "$PROTOCOL" = "p1" ]; then
  python fit_sigma.py --regime "$REGIME" --pred "$PRED" --out "$SIGMA" || exit $?
elif [ ! -f "$SIGMA" ]; then
  echo "[error] P2 reuses the P1 σ-table and $SIGMA does not exist. Score P1 first." >&2; exit 2
fi

mkdir -p "$(dirname "$SCORES")"
# `candi.bench.external` takes --chroms as ONE comma-separated string; run_eic.py takes a list. Read
# the chromosomes back out of the manifest the predict pass just wrote rather than re-deriving them,
# so the set scored is exactly the set emitted.
BENCH_CHROMS=$(python -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['chroms']))" \
                 "$PRED/manifest.json") || exit $?
python -m candi.bench.external \
  --store "$REGIME" --pred "$PRED" --out "$SCORES" --sigma-table "$SIGMA" --chroms "$BENCH_CHROMS"
rc=$?

echo "[edice] DONE eic-score $PROTOCOL n_targets=$N_TARGETS rc=$rc  scores=$SCORES"
exit $rc
