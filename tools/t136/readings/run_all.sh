#!/bin/bash
# t136 readings (h28, the arm-to-arm grid): the whole pipeline in one command, from the laptop.
# Idempotent; re-run it when more Darm2 runs land.
#
#   bash tools/t136/readings/run_all.sh            # clusters + tables + second method + mirror
#   bash tools/t136/readings/run_all.sh --local    # tables + second method from the files in $RES
#
# 1. Nibi (the grid, read-only): extract_records.py grid + verify_raw.py collect ... grid, on the
#    login node (33-42 runs, about 10 s of stdlib json reading; no job is submitted);
# 2. Fir (the Dref / Dviabase references, read-only, never rerun): extract_records.py ref +
#    verify_raw.py collect ... ref, on the login node (42 runs, about 10 s);
# 3. brings back only the small tables; 4. analyze.py (pandas), verify_raw.py reduce (stdlib),
#    verify_compare.py -> verify.txt; 5. NIBI_PATH.txt, FIR_PATH.txt; 6. mirrors $RES -> $MIRROR.
# Exit 69/255 from ssh = the tunnel is down: run `hpc up <cluster>` in Terminal, then re-run.
#
# Environment: T136_ROOT (Nibi grid, default /project/def-maxwl/mforooz/t136), T128P2_ROOT (Fir
# references, default /project/def-maxwl/mforooz/t128p2), T136_RESULTS, T136_MIRROR, PANDAS_PY.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
T=${T136_ROOT:-/project/def-maxwl/mforooz/t136}
P2=${T128P2_ROOT:-/project/def-maxwl/mforooz/t128p2}
RN=$T/readings                                 # on Nibi
RF=/project/def-maxwl/mforooz/t136/readings    # on Fir
CANDII=/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII
RES=${T136_RESULTS:-$CANDII/.claude/worktrees/counterfactual-f/cruxvault/results/t136}
MIRROR=${T136_MIRROR:-$CANDII/cruxvault/results/t136}
PANDAS_PY=${PANDAS_PY:-/Users/mforooz/miniforge3/envs/candii/bin/python}
LOCAL=0
[ "${1:-}" = --local ] && LOCAL=1
mkdir -p "$RES"

remote() {  # remote <host> <command>; stops the pipeline on a dead tunnel
  local rc=0
  ssh -o BatchMode=yes "$1" "$2" || rc=$?
  if [ "$rc" = 69 ] || [ "$rc" = 255 ]; then
    echo "ssh $1 exited $rc: the tunnel is down. Run 'hpc up $1' in Terminal, then re-run." >&2
    exit "$rc"
  fi
  return "$rc"
}

if [ "$LOCAL" = 0 ]; then
  for spec in "nibi $RN $T grid nibi" "fir $RF $P2 ref fir"; do
    set -- $spec
    remote "$1" "mkdir -p $2 && test -d $3"
    rsync -a -e 'ssh -o BatchMode=yes' "$HERE/extract_records.py" "$HERE/verify_raw.py" "$1:$2/" \
      || { echo "rsync to $1 failed" >&2; exit 1; }
    remote "$1" "cd $2 && python3 extract_records.py $4 $2 $3 && python3 verify_raw.py collect $2/verify_collect_$5.json $3 $4"
  done
  rsync -t -e 'ssh -o BatchMode=yes' "nibi:$RN/records_nibi.tsv.gz" "nibi:$RN/runs_nibi.tsv" \
    "nibi:$RN/verify_collect_nibi.json" "$RES/" || { echo "rsync from nibi failed" >&2; exit 1; }
  rsync -t -e 'ssh -o BatchMode=yes' "fir:$RF/ref_records_fir.tsv.gz" "fir:$RF/ref_runs_fir.tsv" \
    "fir:$RF/verify_collect_fir.json" "$RES/" || { echo "rsync from fir failed" >&2; exit 1; }
  QUEUE=$(remote nibi "squeue -u \$USER -h -o '%i %j %T %r' | grep t136 || true")
  JOBS=$(remote nibi "cat $T/jobs.tsv 2>/dev/null || true")
  STAMP=$(date -u +%FT%TZ)
  cat > "$RES/NIBI_PATH.txt" <<EOF
t136 grid (h28), read $STAMP. Tree read: nibi:$T
  runs (42: Darm1 0-20, Darm2 21-41; <arm>_<g>_counts_real_s<seed>)   nibi:$T/runs/
  readings (extraction + second-method collection)                  nibi:$RN/
  kit K1 (git 1f634e1)                                              nibi:$T/code/K1/
Job ledger (nibi:$T/jobs.tsv, copy below):
$JOBS
t136 jobs in the Nibi queue at read time (squeue, read-only):
${QUEUE:-  (none)}
Rebuild everything: bash tools/t136/readings/run_all.sh   (tables only: add --local)
EOF
  cat > "$RES/FIR_PATH.txt" <<EOF
t136 references (h28), read $STAMP. Read-only; never rerun.
  Dref     (phase-2 old D direct, per-track g; 21 runs)   fir:$P2/runs/D_<g>_counts_real_s{0,1,2}
  Dviabase (Dref's maps through the base; 21 runs)        fir:$P2/viabase/D_<g>_counts_real_s{0,1,2}
  readings (extraction + second-method collection)        fir:$RF/
EOF
fi

"$PANDAS_PY" "$HERE/analyze.py" "$RES" | tee "$RES/analyze_out.txt"
python3 "$HERE/verify_raw.py" reduce "$RES/verify_raw.json" "$RES/verify_collect_nibi.json" \
  "$RES/verify_collect_fir.json"
"$PANDAS_PY" "$HERE/verify_compare.py" "$RES"
if [ "$(cd "$RES" && pwd -P)" != "$(mkdir -p "$MIRROR" && cd "$MIRROR" && pwd -P)" ]; then
  rsync -a "$RES/" "$MIRROR/"
  echo "mirrored $RES -> $MIRROR"
fi
echo "DISAGREE lines: $(grep -c DISAGREE "$RES/verify.txt" || true)"
