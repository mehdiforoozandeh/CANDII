#!/bin/bash
# XE: score the published EIC entries in the onion signal space (tools/onion_eval/external_signal.py).
# One array task per entry; the task id indexes eic_score.TEAMS (0..24). CPU only, no GPU.
# Skip-if-exists on <out_root>/<team id>/blind/SCORED.
#
#   sbatch --array=0-24%12 --output=<logs>/%x_%A_%a.out $KIT/slurm/onion_eval/external_signal.sh \
#       <kit_dir> <evalset_blind.json> <out_root> [extra external_signal.py score args]
#   -> <out_root>/<team id>/blind/{scores.json,scores_full.json,suffstats.npz,SCORED}
# Entries are read from $SUBMISSIONS/<team id>/C##M##.bigwig (default: eic_score's
# /project/def-maxwl/mforooz/ONION/eic/submissions). Blind scoring (gate G4 only): score.py logs
# every pass in BLIND_PASSES.tsv beside the eval set and refuses a repeat without
# --repeat-blind-reason "<text>". Then one table:
#   python3 tools/onion_eval/external_signal.py table --scores <out_root>/*/blind \
#       --model <run dirs> --out <tsv>
# Time and memory are not measured: each target keeps its regions in memory (256 regions x 16384
# bins x 4 B = 17 MB per target) and each bigWig chromosome with a region is decompressed once per
# target. Read seff of the first tasks before the next submission.
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_xe
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000M
#SBATCH --gres=none

set -euo pipefail
USAGE="usage: onion_eval/external_signal.sh <kit_dir> <evalset_blind.json> <out_root> [score args]"
KIT="${1:?$USAGE}"
EVALSET="${2:?$USAGE}"
OUT_ROOT="${3:?$USAGE}"
shift 3
SUBMISSIONS="${SUBMISSIONS:-/project/def-maxwl/mforooz/ONION/eic/submissions}"
TASK="${SLURM_ARRAY_TASK_ID:?run as an array: --array=0-24%12}"
[ -f "$EVALSET" ] || { echo "no evalset $EVALSET" >&2; exit 2; }
echo "=== onion XE job ${SLURM_JOB_ID:-local} task $TASK host=$(hostname) $(date -u)"
set +u; module load python/3.10.13; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$KIT/requirements-fir.txt"
pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
TEAM=$(python3 -c "import sys; sys.path.insert(0, 'tools'); from onion_eval.eic_score import TEAMS; print(TEAMS[int(sys.argv[1])][0])" "$TASK")
OUT="$OUT_ROOT/$TEAM/blind"
if [ -f "$OUT/SCORED" ]; then echo "skip: $OUT/SCORED exists"; exit 0; fi
python3 "$KIT/tools/onion_eval/external_signal.py" score --entry-dir "$SUBMISSIONS/$TEAM" \
    --entry "$TEAM" --evalset "$EVALSET" --allow-blind --out "$OUT" "$@"
echo "=== done $(date -u)"
