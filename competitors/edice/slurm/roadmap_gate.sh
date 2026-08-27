#!/bin/bash
# eDICE validation gate (RIVALS_PLAN §7.3): our PyTorch reimplementation against the published
# Roadmap numbers -- Supplementary Table 2, 203 chr21 test tracks, GW Corr 0.735 +- 0.018 and
# MSE Global 0.091 +- 0.005.
#
# Two modes, chosen by MODE:
#   MODE=sample  the repo's packaged 10k-bin chr21 sample, 20 epochs. A SMOKE run: it proves the
#                loop trains and predicts end to end, and nothing is published for it to hit.
#   MODE=full    the Edmond deposit doi:10.17617/3.VKEFB6, all of chr21, train+val as supports,
#                50 epochs. THIS is the gate.
#
#   mkdir -p slurm-logs
#   MODE=sample sbatch competitors/edice/slurm/roadmap_gate.sh
#   MODE=full   sbatch --time=24:00:00 --mem=64G competitors/edice/slurm/roadmap_gate.sh
#
# --gres is the AGENTS.md §hard-rule MIG slice and is never any other spec. Wall is a measurement:
# --time below is the ASK. Read the real cost out of `seff` or the per-epoch seconds in gate.json.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t52_edice_gate
#SBATCH --output=slurm-logs/t52_edice_%x_%j.out
#SBATCH --error=slurm-logs/t52_edice_%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -uo pipefail

MODE="${MODE:-sample}"
REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_t52}"
VENV="${VENV:-/project/def-maxwl/mforooz/candi_venv}"
EDICE_SRC="${EDICE_SRC:-/project/def-maxwl/mforooz/rivals_src/eDICE}"
DATA="${DATA:-/project/def-maxwl/mforooz/rivals_src/edice_data}"
OUT="${OUT:-/project/def-maxwl/mforooz/rivals_src/edice_runs/${MODE}}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
source "$VENV/bin/activate"

cd "$REPO/competitors/edice"
# competitors/ is NOT importable from candi and does not want candi on the path -- only itself.
export PYTHONPATH="$PWD"
mkdir -p "$OUT"

# idmap and the PredictD splits ship with the reference repo; they are the same files for both modes.
IDMAP="$EDICE_SRC/sample_data/roadmap/idmap.json"
SPLITS="$EDICE_SRC/sample_data/roadmap/predictd_splits.json"

case "$MODE" in
  sample)
    H5="$EDICE_SRC/sample_data/roadmap/SAMPLE_chr21_roadmap_train.h5"
    ARGS=(--epochs 20 --train-splits train)
    ;;
  full)
    H5="$DATA/roadmap_tracks_shuffled.h5"
    ARGS=(--epochs 50 --train-splits train val)
    ;;
  *) echo "[error] MODE must be sample or full, got '$MODE'" >&2; exit 2 ;;
esac

echo "[edice] mode=$MODE h5=$H5 out=$OUT host=$(hostname)"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python run_roadmap.py \
  --h5 "$H5" --idmap "$IDMAP" --splits "$SPLITS" \
  --out "$OUT" "${ARGS[@]}"
rc=$?

echo "[edice] DONE mode=$MODE rc=$rc"
exit $rc
