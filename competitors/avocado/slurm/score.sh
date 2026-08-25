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
#   mkdir -p slurm-logs && sbatch competitors/avocado/slurm/score.sh
#   P2=0 sbatch competitors/avocado/slurm/score.sh     # P1 only, when P2 is not wanted yet
#
#SBATCH --account=def-maxwl
#SBATCH --job-name=t50_score
#SBATCH --output=slurm-logs/t50_score_%j.out
#SBATCH --error=slurm-logs/t50_score_%j.err
#SBATCH --time=08:00:00
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

# 2. P1 -- declared pairs, the regime's own eval_chroms.
python -m candi.bench.external --store "$REGIME" --pred "$PRED" \
    --sigma-table "$SIGMA" --out "$OUT/scores_avocado_P1.json" || exit 1

# 3. P2 -- the same root, every chromosome the store carries.
if [ "${P2:-1}" != "0" ]; then
    python -m candi.bench.external --store "$REGIME" --pred "$PRED" \
        --sigma-table "$SIGMA" --chroms "$(paste -sd, "$AVO/chroms.txt")" \
        --out "$OUT/scores_avocado_P2.json" || exit 1
fi

echo "[t50_score] done; results in $OUT"
