#!/bin/bash
# XRd (Amendment 3): run tools/onion_data/rna_inputs.py inside an ONION code kit.
#
#   sbatch --output=<log> [--time ... --mem ... --cpus-per-task ...] \
#       <kit>/slurm/onion_data/rna_inputs.sh <kit dir> list|fetch|tracks|vector|check <args...>
#
# Environment: the kit venv of slurm/onion_kit/run.sh (requirements-fir.txt and the kit wheels, in
# $SLURM_TMPDIR), plus pyBigWig from the Alliance wheelhouse (--no-index) for the bigWig binning,
# as slurm/onion_data/align_views.sh does. `tracks` gets `--tmp $SLURM_TMPDIR`, so the bedGraph
# intermediates stay node-local. Compute nodes have internet, so `list` and `fetch` run here too.
# The kit root is an argument (SLURM runs a spool copy of this file); nothing goes via --export.
# Header values are an estimate (4 cores, 16000M, the plan's XRf request); re-measure with seff.
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_rna_inputs
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --gres=none

set -uo pipefail
KIT="${1:?usage: rna_inputs.sh <kit dir> list|fetch|tracks|vector|check <args...>}"
MODE="${2:?usage: rna_inputs.sh <kit dir> list|fetch|tracks|vector|check <args...>}"
shift 2
KIT=$(cd "$KIT" && pwd)
[ -f "$KIT/tools/onion_data/rna_inputs.py" ] || { echo "no rna_inputs.py in $KIT" >&2; exit 2; }
echo "=== onion rna_inputs $MODE job ${SLURM_JOB_ID:-none} host=$(hostname) kit=$KIT $(date -u)"
cat "$KIT/.kit_stamp" 2>/dev/null || true

set +u; module load python/3.10.13; set -u
{ virtualenv --no-download "$SLURM_TMPDIR/venv" &&
  source "$SLURM_TMPDIR/venv/bin/activate" &&
  pip install -q --no-index --upgrade pip &&
  pip install -q --no-index -r "$KIT/requirements-fir.txt" &&
  pip install -q --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt" &&
  pip install -q --no-index pyBigWig; } ||
  { echo "rna_inputs.sh: venv setup failed"; echo "[onion-run] exit=3"; exit 3; }
source "$SLURM_TMPDIR/venv/bin/activate"
cd "$KIT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
source "$KIT/slurm/_kit_pin.sh"

case "$MODE" in
  tracks) python tools/onion_data/rna_inputs.py tracks "$@" --tmp "$SLURM_TMPDIR/bdg" ;;
  list|fetch|vector|check) python tools/onion_data/rna_inputs.py "$MODE" "$@" ;;
  *) echo "unknown mode $MODE" >&2; echo "[onion-run] exit=2"; exit 2 ;;
esac
rc=$?
echo "=== done $(date -u)"
echo "[onion-run] exit=$rc"
exit $rc
