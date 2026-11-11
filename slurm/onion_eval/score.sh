#!/bin/bash
# E6: score one miniCANDI run or one training-free baseline on the onion eval set
# (tools/onion_eval/score.py). One CPU job, no array, no GPU. Skip-if-exists on <out>/SCORED
# (written by score.py after scores.json; never training's DONE, which a run dir always holds).
#
#   sbatch --output=<logs>/%x_%j.out $KIT/slurm/onion_eval/score.sh \
#       <kit_dir> <evalset.json> <out_dir> run <run_dir> [extra score.py args]
#   sbatch ... score.sh <kit_dir> <evalset.json> <out_dir> baseline <name> [extra score.py args]
#   -> <out_dir>/{scores.json,suffstats.npz,SCORED}
# A run is scored into its own seed dir (<out_dir> = <run_dir>); a baseline into
# /project/def-maxwl/mforooz/ONION/eval/<tables>/baselines/<name>/seed0. Blind scoring (gate G4
# only) needs "--split blind --allow-blind" in the extra args, the blind eval set
# eval/eic_v2/evalset_blind.json and <out_dir> = <seed dir>/blind; score.py logs every blind pass in
# BLIND_PASSES.tsv beside the eval set and refuses a repeat without --repeat-blind-reason "<text>".
# 16 cores x 4000 MB (Nibi/Fir CPU rule); --time set explicitly (Nibi default is 15 min).
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_score
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --gres=none

set -euo pipefail
USAGE="usage: onion_eval/score.sh <kit_dir> <evalset.json> <out_dir> run|baseline <run_dir|name> [score.py args]"
KIT="${1:?$USAGE}"
EVALSET="${2:?$USAGE}"
OUT="${3:?$USAGE}"
WHAT="${4:?$USAGE}"
WHO="${5:?$USAGE}"
shift 5
case "$WHAT" in
  run) SRC=(--run "$WHO") ;;
  baseline)
    case "$WHO" in
      marginal_nb|position_average|depth_only_scaling|pseudoreplicate_floor) ;;
      *) echo "unknown baseline $WHO" >&2; exit 2 ;;
    esac
    SRC=(--baseline "$WHO") ;;
  *) echo "$USAGE" >&2; exit 2 ;;
esac
[ -f "$EVALSET" ] || { echo "no evalset $EVALSET" >&2; exit 2; }
CMD=(python3 "$KIT/tools/onion_eval/score.py" "${SRC[@]}" --evalset "$EVALSET" --out "$OUT" "$@")
if [ "${DRY_RUN:-0}" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
if [ -f "$OUT/SCORED" ]; then echo "skip: $OUT/SCORED exists"; exit 0; fi
echo "=== onion score job ${SLURM_JOB_ID:-local} host=$(hostname) $(date -u)"
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
"${CMD[@]}"
echo "=== done $(date -u)"
