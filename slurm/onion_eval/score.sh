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
# CPU form: 2 cores x 4000 MB (F3, measured on Fir, pilot_rung2 seed 0, 64 regions: 2 CPUs 25.0 min
# = 0.83 CPU-h; 4 CPUs 20.9 min = 1.39; 8 CPUs 21.6 min = 2.88; the old 16 CPUs 39 min = 10.4; the
# scorer is mostly single-threaded). Rung 1 (3.3e10 FLOPs per window) on the GPU form: 39.8 min,
# 0.66 CPU-h + 0.66 GPU-h, against 38.3 CPU-h on 16 CPUs; rung 5 (7.5e9) not measured. GPU scores
# differ from CPU scores by up to 1.1e-5: score every seed of one comparison on one device.
# GPU form (a 1g.10gb slice is free; rung 1's forward is 44x rung 2's): add
#   --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=1 --mem=24000M (and on Fir
#   --exclude=fc10713,fc10715) to the sbatch line; score.py's --device auto then runs on the GPU.
# Torch uses OMP_NUM_THREADS = the allocated CPUs (never the node's cores; F3).
# SM speed (outputs identical either way): ONION_STAGE=1 (default) copies the h5 files the scoring
#   reads to $SLURM_TMPDIR/stage first (--stage-dir; Nibi NFS opens cost 0.21 s); ONION_STAGE=0 reads
#   in place. ONION_SCORE_WORKERS=<n> (default: the allocated CPUs) forks n worker processes over
#   the imputation groups and the pairs (--workers; CPU device only: a GPU scoring stays serial).
#   Memory grows with the workers (each holds one target group): see cruxvault/results/scorer_memory.
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_score
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
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
SPEED=(--workers "${ONION_SCORE_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}")
if [ "${ONION_STAGE:-1}" = 1 ] && [ -n "${SLURM_TMPDIR:-}" ]; then
  SPEED+=(--stage-dir "$SLURM_TMPDIR/stage")
fi
CMD=(python3 "$KIT/tools/onion_eval/score.py" "${SRC[@]}" --evalset "$EVALSET" --out "$OUT" "${SPEED[@]}" "$@")
if [ "${DRY_RUN:-0}" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
if [ -f "$OUT/SCORED" ]; then echo "skip: $OUT/SCORED exists"; exit 0; fi
echo "=== onion score job ${SLURM_JOB_ID:-local} host=$(hostname) $(date -u)"
[ -z "${SLURM_TMPDIR:-}" ] || df -h "$SLURM_TMPDIR" | tail -1
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
# SM: numpy's BLAS pool is pinned (default 2 threads, the CPU count of the E7a baseline and rung-1
# scorings): r_spec's Level fit (inv, gemv) sums in an order that depends on it (2.3e-7 between 2
# and 8 threads on a 32-region subset), so a fixed pool makes scores independent of the job's CPUs.
export OPENBLAS_NUM_THREADS="${ONION_BLAS_THREADS:-2}" FLEXIBLAS_NUM_THREADS="${ONION_BLAS_THREADS:-2}" \
  MKL_NUM_THREADS="${ONION_BLAS_THREADS:-2}"
"${CMD[@]}"
echo "=== done $(date -u)"
