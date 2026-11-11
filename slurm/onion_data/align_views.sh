#!/bin/bash
# t167 (chunk D5): run tools/onion_data/align_views.py inside an ONION code kit.
#
#   sbatch --output=<log> [--time ... --mem ... --cpus-per-task ...] \
#       <kit>/slurm/onion_data/align_views.sh <kit dir> build <build args...>
#   sbatch ... align_views.sh <kit dir> verify --measurements <dir>
#
# Environment: the kit venv of slurm/onion_kit/run.sh (requirements-fir.txt and the kit wheels, in
# $SLURM_TMPDIR), plus pyBigWig from the Alliance wheelhouse (--no-index) for the signal binning,
# as slurm/t117/pseudoreps.sh did, and apptainer 1.3.5 for the pipeline SIFs. Nothing is installed
# anywhere else. `build` gets `--tmp $SLURM_TMPDIR`; every intermediate stays there.
# The kit root is an argument (SLURM runs a spool copy of this file); nothing goes via --export.
# Header values are the plan's D5 request (16 cores, 64000M); re-measure with `seff <job id>`.
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_align_views
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --gres=none

set -uo pipefail
KIT="${1:?usage: align_views.sh <kit dir> build|verify <args...>}"
MODE="${2:?usage: align_views.sh <kit dir> build|verify <args...>}"
shift 2
KIT=$(cd "$KIT" && pwd)
[ -f "$KIT/tools/onion_data/align_views.py" ] || { echo "no align_views.py in $KIT" >&2; exit 2; }
echo "=== onion align_views $MODE job ${SLURM_JOB_ID:-none} host=$(hostname) kit=$KIT $(date -u)"
cat "$KIT/.kit_stamp" 2>/dev/null || true
df -h "$SLURM_TMPDIR" | tail -1

set +u; module load apptainer/1.3.5 python/3.10.13; set -u
{ virtualenv --no-download "$SLURM_TMPDIR/venv" &&
  source "$SLURM_TMPDIR/venv/bin/activate" &&
  pip install -q --no-index --upgrade pip &&
  pip install -q --no-index -r "$KIT/requirements-fir.txt" &&
  pip install -q --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt" &&
  pip install -q --no-index pyBigWig; } ||
  { echo "align_views.sh: venv setup failed"; echo "[onion-run] exit=3"; exit 3; }
source "$SLURM_TMPDIR/venv/bin/activate"
cd "$KIT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
source "$KIT/slurm/_kit_pin.sh"

case "$MODE" in
  build) python tools/onion_data/align_views.py build "$@" --tmp "$SLURM_TMPDIR" ;;
  verify) python tools/onion_data/align_views.py verify "$@" ;;
  *) echo "unknown mode $MODE" >&2; echo "[onion-run] exit=2"; exit 2 ;;
esac
rc=$?
echo "=== done $(date -u)"
echo "[onion-run] exit=$rc"
exit $rc
