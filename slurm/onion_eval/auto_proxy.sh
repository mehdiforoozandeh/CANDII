#!/bin/bash
# EVt: one seed of one proxy attempt of the evolution step (tools/onion_eval/auto_proxy.py).
# Submitted by auto_proxy.py through `hpc run <cluster> 'sbatch ...'`, one job per seed:
#
#   sbatch --output=<attempt>/seed<s>.log $KIT/slurm/onion_eval/auto_proxy.sh \
#       <kit_dir> <attempt_dir> <seed> <windows> <tables_dir> <evalset.json> <recipe_rel> <tally.tsv>
#   -> <attempt_dir>/seed<s>/{proxy_config.json,ckpt.pt,run.json,scores.json,SCORED,PROXY.json}
#
# The job builds the kit's venv, copies the kit's code (tools/, src/, slurm/; about 9 MB) into
# $SLURM_TMPDIR/tree, puts the candidate's config.json and model.py (shipped to
# <attempt_dir>/recipe/) at <recipe_rel> inside that tree, so a `model_file` relative to the
# repository root resolves to the candidate, and runs `auto_proxy.py job`: input checks (no blind
# row, validation targets only, eic_v1 tables), training of <windows> windows, the target-
# independence check of the trained model, and `score.py` on split val (never --allow-blind).
# Every other file comes from the frozen kit.
# Tally: on exit, in any state, the job appends one row (time, attempt, seed, job id, cpus, wall
# seconds, CPU-h = wall x allocated CPUs / 3600, exit code) to <tally.tsv> under flock.
# Resources (foreman budget, Log 2026-11-11): 12 CPUs, 4000 MB per CPU, 8 loader workers, CPU only.
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_evo
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --gres=none

set -euo pipefail
USAGE="usage: onion_eval/auto_proxy.sh <kit_dir> <attempt_dir> <seed> <windows> <tables_dir> <evalset.json> <recipe_rel> <tally.tsv>"
[ $# -eq 8 ] || { echo "$USAGE" >&2; exit 2; }
KIT="$1" ATT="$2" SEED="$3" WINDOWS="$4" TABLES="$5" EVALSET="$6" RECIPE_REL="$7" TALLY="$8"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be an integer" >&2; exit 2; }
[[ "$WINDOWS" =~ ^[1-9][0-9]*$ ]] || { echo "windows must be a positive integer" >&2; exit 2; }
[[ "$RECIPE_REL" =~ ^tools/minicandi/[A-Za-z0-9_./-]+$ ]] && [[ "$RECIPE_REL" != *..* ]] \
  || { echo "recipe_rel must be a path under tools/minicandi/ without '..'" >&2; exit 2; }
case "$EVALSET" in *blind*) echo "refused: blind eval set $EVALSET" >&2; exit 2 ;; esac
ATTEMPT="$(basename "$ATT")"
OUT="$ATT/seed$SEED"
WORKERS=8
CMD=(python3 tools/onion_eval/auto_proxy.py job --recipe "$RECIPE_REL" --tables "$TABLES"
     --evalset "$EVALSET" --seed "$SEED" --windows "$WINDOWS" --out "$OUT" --workers "$WORKERS")
if [ "${DRY_RUN:-0}" = 1 ]; then echo "${CMD[*]}"; echo "tally $TALLY attempt $ATTEMPT"; exit 0; fi

START=$(date +%s)
tally() {
  local rc=$? end wall cpus cpuh
  end=$(date +%s); wall=$(( end - START )); cpus="${SLURM_CPUS_PER_TASK:-1}"
  cpuh=$(awk -v w="$wall" -v c="$cpus" 'BEGIN { printf "%.4f", w * c / 3600 }')
  mkdir -p "$(dirname "$TALLY")"
  (
    flock 9
    [ -s "$TALLY" ] || printf 'time\tattempt\tseed\tjob_id\tcpus\twall_seconds\tcpu_hours\texit\n' >> "$TALLY"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ATTEMPT" "$SEED" \
      "${SLURM_JOB_ID:-local}" "$cpus" "$wall" "$cpuh" "$rc" >> "$TALLY"
  ) 9>>"$TALLY.lock"
  echo "=== tally: $cpuh CPU-h (wall $wall s x $cpus CPUs), exit $rc"
}
trap tally EXIT
trap 'exit 143' TERM

[ -d "$ATT/recipe" ] || { echo "no recipe dir $ATT/recipe" >&2; exit 2; }
[ -f "$EVALSET" ] || { echo "no evalset $EVALSET" >&2; exit 2; }
mkdir -p "$OUT"
echo "=== onion evo proxy job ${SLURM_JOB_ID:-local} host=$(hostname) attempt=$ATTEMPT seed=$SEED $(date -u)"
set +u; module load python/3.10.13; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$KIT/requirements-fir.txt"
pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
TREE="$SLURM_TMPDIR/tree"
mkdir -p "$TREE"
cp -a "$KIT/tools" "$KIT/src" "$KIT/slurm" "$TREE/"
mkdir -p "$TREE/$RECIPE_REL"
install -m 0644 "$ATT/recipe/config.json" "$TREE/$RECIPE_REL/config.json"
install -m 0644 "$ATT/recipe/model.py" "$TREE/$RECIPE_REL/model.py"
cd "$TREE"
KIT="$TREE"
source "$TREE/slurm/_kit_pin.sh"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
"${CMD[@]}"
echo "=== done $(date -u)"
