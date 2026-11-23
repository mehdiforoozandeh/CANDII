#!/bin/bash
# t392 C3: stranded read-start recount of the plan rows of one array task, inside an ONION code kit.
#
#   sbatch --job-name=rs392_<name> --array=1-<tasks> --output=<logs>/%x_%A_%a.out \
#       [--time ... --mem ... --cpus-per-task ...] \
#       <kit>/slurm/t392/rs_recount.sh <kit dir> <plan.tsv> \
#       --work-root <prefix> --out <store root> [--measurements <eic_v1_2 dirs>] [--reach 8]
#
# <plan.tsv> has a header with the column `task` and `measurement_id` (tools/t392/build_plan.py).
# The rows whose `task` equals $SLURM_ARRAY_TASK_ID run, one `read_starts.py run` each. A
# measurement whose provenance.json records a passed gate on all three files is skipped. One line
# per measurement: id, wall seconds, peak RSS (kB), n_differing (starts, pr1, pr2), conversion
# floor (starts, pr1, pr2). Exit 1 when any measurement fails.
#
# Environment: the kit venv of slurm/onion_data/recount.sh. The kit root is an argument (SLURM
# runs a spool copy of this file); nothing goes via --export.
#SBATCH --account=def-maxwl
#SBATCH --job-name=rs392_recount
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000M
#SBATCH --gres=none

set -uo pipefail
USAGE="usage: rs_recount.sh <kit dir> <plan.tsv> --work-root <prefix> --out <store root> [--measurements <dir>] [--reach N]"
KIT="${1:?$USAGE}"
PLAN="${2:?$USAGE}"
shift 2
MEAS=/scratch/mforooz/onion/measurements
WORK_ROOT=""
OUT=""
REACH=8
while [ $# -gt 0 ]; do
  case "$1" in
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --measurements) MEAS="$2"; shift 2 ;;
    --reach) REACH="$2"; shift 2 ;;
    *) echo "unknown argument $1; $USAGE" >&2; exit 2 ;;
  esac
done
[ -n "$WORK_ROOT" ] && [ -n "$OUT" ] || { echo "$USAGE" >&2; exit 2; }
TASK="${SLURM_ARRAY_TASK_ID:?rs_recount.sh runs as an array task}"
KIT=$(cd "$KIT" && pwd)
PLAN=$(cd "$(dirname "$PLAN")" && pwd)/$(basename "$PLAN")
[ -f "$KIT/tools/onion_data/read_starts.py" ] || { echo "no read_starts.py in $KIT" >&2; exit 2; }
[ -f "$PLAN" ] || { echo "no plan $PLAN" >&2; exit 2; }
echo "=== rs392 recount task $TASK job ${SLURM_JOB_ID:-none} host=$(hostname) kit=$KIT plan=$PLAN $(date -u)"
cat "$KIT/.kit_stamp" 2>/dev/null || true

set +u; module load python/3.10.13; set -u
{ virtualenv --no-download "$SLURM_TMPDIR/venv" &&
  source "$SLURM_TMPDIR/venv/bin/activate" &&
  pip install -q --no-index --upgrade pip &&
  pip install -q --no-index -r "$KIT/requirements-fir.txt" &&
  pip install -q --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"; } ||
  { echo "rs_recount.sh: venv setup failed"; echo "[onion-run] exit=3"; exit 3; }
source "$SLURM_TMPDIR/venv/bin/activate"
cd "$KIT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
source "$KIT/slurm/_kit_pin.sh"

# the measurement ids of this task, from the header-named columns
mapfile -t IDS < <(python - "$PLAN" "$TASK" <<'PY'
import csv, sys
with open(sys.argv[1], newline="") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        if row["task"].strip() == sys.argv[2]:
            print(row["measurement_id"].strip())
PY
)
[ "${#IDS[@]}" -gt 0 ] || { echo "no plan rows for task $TASK"; echo "[onion-run] exit=2"; exit 2; }

# status line of one store folder: "<passed 0|1> <id> <wall_s> <peak_rss_kb> <n_diff,..> <floor,..>"
summary() {
  python - "$1" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
try:
    p = json.loads((d / "provenance.json").read_text())
    o = p["outputs"]
    names = ("starts.h5", "starts_pr1.h5", "starts_pr2.h5")
    ok = all(o[n]["gate"]["n_differing"] == 0 and o[n]["gate"]["result"] == "pass" for n in names)
    nd = ",".join(str(o[n]["gate"]["n_differing"]) for n in names)
    fl = ",".join(f'{o[n]["conversion_floor_nmse"]:.6g}' for n in names)
    print(int(ok), d.name, p["wall_s"], p["peak_rss_kb"], nd, fl)
except Exception as e:  # no or partial record
    print(0, d.name, "NA", "NA", "NA", f"no-record:{type(e).__name__}")
PY
}

rc=0
for MID in "${IDS[@]}"; do
  read -r PASSED LINE <<<"$(summary "$OUT/$MID")"
  if [ "$PASSED" = 1 ]; then
    echo "[rs392] skip $LINE"
    continue
  fi
  python tools/onion_data/read_starts.py run --measurement "$MID" --measurements "$MEAS" \
      --work-root "$WORK_ROOT" --out "$OUT" --reach "$REACH"
  r=$?
  read -r PASSED LINE <<<"$(summary "$OUT/$MID")"
  echo "[rs392] $LINE exit=$r"
  [ "$r" -eq 0 ] && [ "$PASSED" = 1 ] || rc=1
done
echo "=== done $(date -u)"
echo "[onion-run] exit=$rc"
exit $rc
