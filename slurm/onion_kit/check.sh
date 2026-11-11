#!/bin/bash
# Short Python check inside an ONION code kit (t152, chunk K0). Submitted by
# tools/onion_kit/remote_check.sh; not meant to be submitted by hand.
#
#   sbatch --output=<log> $KIT/slurm/onion_kit/check.sh <kit dir> <python args...>
#
# Builds a venv in $SLURM_TMPDIR from requirements-fir.txt and the kit's wheels/, pins `candi`
# to the kit (slurm/_kit_pin.sh), cds into the kit and runs `python <python args>`.
# Setup output goes to <log>.setup; the log itself holds only the check's own stdout and stderr,
# then one last line `[onion-check] exit=<rc>` that remote_check.sh reads.
# remote_check.sh may override --time, --cpus-per-task and --mem on the sbatch line.
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_check
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --gres=none

set -uo pipefail
KIT="${1:?usage: check.sh <kit dir> <python args...>}"
shift
[ $# -gt 0 ] || { echo "check.sh: no python args" >&2; echo "[onion-check] exit=2"; exit 2; }
SETUP_LOG="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *StdOut=//p').setup"
[ "$SETUP_LOG" != .setup ] || SETUP_LOG="$SLURM_TMPDIR/setup.log"

setup() {
  echo "=== onion check job $SLURM_JOB_ID host=$(hostname) kit=$KIT $(date -u)"
  cat "$KIT/.kit_stamp" 2>/dev/null || true
  set +u; module load python/3.10.13; set -u
  virtualenv --no-download "$SLURM_TMPDIR/venv" &&
  source "$SLURM_TMPDIR/venv/bin/activate" &&
  pip install --no-index --upgrade pip &&
  pip install --no-index -r "$KIT/requirements-fir.txt" &&
  pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
}
if ! setup > "$SETUP_LOG" 2>&1; then
  echo "check.sh: venv setup failed; last lines of $SETUP_LOG:"
  tail -20 "$SETUP_LOG"
  echo "[onion-check] exit=3"
  exit 3
fi
source "$SLURM_TMPDIR/venv/bin/activate"
cd "$KIT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
source "$KIT/slurm/_kit_pin.sh" >> "$SETUP_LOG" 2>&1 || { echo "check.sh: kit pin failed"; tail -5 "$SETUP_LOG"; echo "[onion-check] exit=4"; exit 4; }
python "$@"
rc=$?
echo "[onion-check] exit=$rc"
exit $rc
