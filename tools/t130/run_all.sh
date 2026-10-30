#!/bin/bash
# t130 blind check-set readings: bring the extract and the t137 floor down, then the table and
# the second method, from the laptop. It does NOT submit the extract job (slurm/t130/extract.sh).
#
#   bash tools/t130/run_all.sh            # rsync from Nibi, then analyze + verify
#   bash tools/t130/run_all.sh --local    # analyze + verify on the files already in $T130_RESULTS
#
# Environment:
#   T130_ROOT     Nibi extract dir   (default /project/def-maxwl/mforooz/t130)
#   T137_ROOT     Nibi floor dir     (default /project/def-maxwl/mforooz/t137/floor)
#   NESTED_ROOT   Nibi nested-read oracle dir (default $T130_ROOT/nested; slurm/t130/nested_oracle.sh)
#   T130_RESULTS  local results dir  (default $PWD/cruxvault/results/t130)
#   PANDAS_PY     python with pandas (default the candii env)
# Exit 69/255 from ssh = the tunnel is down: run `hpc up nibi` in Terminal, then re-run.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
T130_ROOT=${T130_ROOT:-/project/def-maxwl/mforooz/t130}
T137_ROOT=${T137_ROOT:-/project/def-maxwl/mforooz/t137/floor}
NESTED_ROOT=${NESTED_ROOT:-$T130_ROOT/nested}
RES=${T130_RESULTS:-$PWD/cruxvault/results/t130}
PY=${PANDAS_PY:-/Users/mforooz/miniforge3/envs/candii/bin/python}
H=nibi
LOCAL=0
[ "${1:-}" = --local ] && LOCAL=1
mkdir -p "$RES"

if [ "$LOCAL" = 0 ]; then
  rc=0
  ssh -o BatchMode=yes "$H" "test -f $T130_ROOT/EXTRACT_DONE" || rc=$?
  if [ "$rc" = 69 ] || [ "$rc" = 255 ]; then
    echo "ssh $H exited $rc: the tunnel is down. Run 'hpc up $H' in Terminal, then re-run." >&2
    exit "$rc"
  fi
  [ "$rc" = 0 ] || { echo "no $H:$T130_ROOT/EXTRACT_DONE: the extract has not finished" >&2; exit 1; }
  for f in records.tsv.gz baselines.tsv.gz runs.tsv EXTRACT_DONE; do
    rsync -av --no-g --no-p -e 'ssh -o BatchMode=yes' "$H:$T130_ROOT/$f" "$RES/"
  done
  for f in floor_pairs.tsv floor_curve.tsv floor_p15M.tsv; do
    rsync -av --no-g --no-p -e 'ssh -o BatchMode=yes' "$H:$T137_ROOT/$f" "$RES/"
  done
  ssh -o BatchMode=yes "$H" "test -f $NESTED_ROOT/NESTED_DONE" \
    || { echo "no $H:$NESTED_ROOT/NESTED_DONE: the nested-read oracle has not finished" >&2; exit 1; }
  for f in nested_oracle.tsv nested_dispersion.tsv NESTED_DONE; do
    rsync -av --no-g --no-p -e 'ssh -o BatchMode=yes' "$H:$NESTED_ROOT/$f" "$RES/"
  done
fi

"$PY" "$HERE/analyze.py" "$RES"
"$PY" "$HERE/verify_raw.py" "$RES/records.tsv.gz" "$RES/baselines.tsv.gz" \
  "$RES/floor_pairs.tsv" "$RES/floor_curve.tsv" "$RES/nested_oracle.tsv" > /dev/null
"$PY" "$HERE/verify_compare.py" "$RES"

cat > "$RES/NIBI_PATH.txt" <<EOT
t130 blind check-set readings (h1 depth, h2 run type). Exploratory readings; no verdicts.
  extract (records.tsv.gz, baselines.tsv.gz, runs.tsv)   nibi:$T130_ROOT/
  floor (floor_pairs, floor_curve, floor_p15M; t137)     nibi:$T137_ROOT/   (context column)
  nested-read oracle (nested_oracle, nested_dispersion)  nibi:$NESTED_ROOT/   (the depth-pair oracle)
  row-1 ladder runs (read only)                          nibi:/project/def-maxwl/mforooz/t118/ladder/runs/
  baselines rungs_v2 (read only)                         nibi:/project/def-maxwl/mforooz/t118/rungs_v2/
  extract summary: $(cat "$RES/EXTRACT_DONE" 2>/dev/null || echo "(EXTRACT_DONE not here)")
Rebuild: bash tools/t130/run_all.sh   (tables only: add --local)
EOT
echo "DISAGREE lines: $(grep -c DISAGREE "$RES/verify.txt" || true)"
