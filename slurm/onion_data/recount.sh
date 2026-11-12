#!/bin/bash
# t152 Amendment 3 (chunk RCt): run tools/onion_data/recount.py inside an ONION code kit.
#
#   sbatch --output=<log> [--time ... --mem ... --cpus-per-task ...] \
#       <kit>/slurm/onion_data/recount.sh <kit dir> regions --evalset <json> --mask <h5> --out <tsv>
#   sbatch ... recount.sh <kit dir> run --work <dir> --experiment <acc> --regions <tsv> \
#       (--frag-len <int|0> | --frag-len-source xcor ...) --halves 0|1 --out <store dir>
#   sbatch ... recount.sh <kit dir> verify --out <measurement dir> --counts <eic_v1 counts.h5>
#   sbatch ... recount.sh <kit dir> verify --all --tables <eic_res_v1> [--workers N]
#   sbatch ... recount.sh <kit dir> tables --eic-v1 <dir> --store <dir> --out <eic_res_v1>
#
# Environment: the kit venv of slurm/onion_kit/run.sh (requirements-fir.txt and the kit wheels, in
# $SLURM_TMPDIR). Nothing is installed anywhere else. The kit root is an argument (SLURM runs a
# spool copy of this file); nothing goes via --export.
# Header values are a first guess (a run holds two uint32 1 bp arrays of 768 x 409,600 positions,
# 1.26 GB each); chunk RC0 measures the cost; re-measure with `seff <job id>`.
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_recount
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000M
#SBATCH --gres=none

set -uo pipefail
KIT="${1:?usage: recount.sh <kit dir> regions|run|verify|tables <args...>}"
MODE="${2:?usage: recount.sh <kit dir> regions|run|verify|tables <args...>}"
shift 2
KIT=$(cd "$KIT" && pwd)
[ -f "$KIT/tools/onion_data/recount.py" ] || { echo "no recount.py in $KIT" >&2; exit 2; }
echo "=== onion recount $MODE job ${SLURM_JOB_ID:-none} host=$(hostname) kit=$KIT $(date -u)"
cat "$KIT/.kit_stamp" 2>/dev/null || true

set +u; module load python/3.10.13; set -u
{ virtualenv --no-download "$SLURM_TMPDIR/venv" &&
  source "$SLURM_TMPDIR/venv/bin/activate" &&
  pip install -q --no-index --upgrade pip &&
  pip install -q --no-index -r "$KIT/requirements-fir.txt" &&
  pip install -q --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"; } ||
  { echo "recount.sh: venv setup failed"; echo "[onion-run] exit=3"; exit 3; }
source "$SLURM_TMPDIR/venv/bin/activate"
cd "$KIT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
source "$KIT/slurm/_kit_pin.sh"

case "$MODE" in
  regions|run|verify|tables) python tools/onion_data/recount.py "$MODE" "$@" ;;
  *) echo "unknown mode $MODE" >&2; echo "[onion-run] exit=2"; exit 2 ;;
esac
rc=$?
echo "=== done $(date -u)"
echo "[onion-run] exit=$rc"
exit $rc
