#!/bin/bash
# t392 C11a: the read-start average file, `rs_dataset.build_average_h5`, inside an ONION code kit.
#
#   sbatch --job-name=rs392_avg --output=<logs>/%x_%j.out [--dependency=afterok:...] \
#       <kit>/slurm/t392/rs_avg.sh <kit dir> <reference average h5> <store root> <tables> <out h5>
#
# Environment: the kit venv of slurm/t392/rs_recount.sh.
#SBATCH --account=def-maxwl
#SBATCH --job-name=rs392_avg
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --gres=none

set -uo pipefail
USAGE="usage: rs_avg.sh <kit dir> <reference average h5> <store root> <tables> <out h5>"
KIT="${1:?$USAGE}"
REF="${2:?$USAGE}"
STORE="${3:?$USAGE}"
TABLES="${4:?$USAGE}"
OUT="${5:?$USAGE}"
KIT=$(cd "$KIT" && pwd)
for p in "$REF" "$TABLES/measurements.tsv" "$KIT/tools/minicandi/rs_dataset.py"; do
  [ -f "$p" ] || { echo "missing $p" >&2; exit 2; }
done
[ -d "$STORE" ] || { echo "no store $STORE" >&2; exit 2; }
echo "=== rs392 avg job ${SLURM_JOB_ID:-none} host=$(hostname) kit=$KIT $(date -u)"
cat "$KIT/.kit_stamp" 2>/dev/null || true

set +u; module load python/3.10.13; set -u
{ virtualenv --no-download "$SLURM_TMPDIR/venv" &&
  source "$SLURM_TMPDIR/venv/bin/activate" &&
  pip install -q --no-index --upgrade pip &&
  pip install -q --no-index -r "$KIT/requirements-fir.txt" &&
  pip install -q --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"; } ||
  { echo "rs_avg.sh: venv setup failed"; echo "[onion-run] exit=3"; exit 3; }
source "$SLURM_TMPDIR/venv/bin/activate"
cd "$KIT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
source "$KIT/slurm/_kit_pin.sh"

python - "$REF" "$STORE" "$TABLES" "$OUT" <<'PY'
import sys, time
sys.path.insert(0, "tools")
from minicandi.rs_dataset import build_average_h5
t = time.time()
out = build_average_h5(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
print(f"[rs392] wrote {out} in {time.time() - t:.0f} s")
PY
r=$?
echo "=== done $(date -u)"
echo "[onion-run] exit=$r"
exit $r
