#!/bin/bash
# t128 phase 2 readings: the whole pipeline in one command, from the laptop. Idempotent; re-run it
# when more runs land (the Fir step re-reads only runs whose DONE marks changed).
#
#   bash tools/t128p2/readings/run_all.sh            # Fir + tables + second method
#   bash tools/t128p2/readings/run_all.sh --local    # tables + second method from the files here
#
# Environment:
#   T128P2_ROOT     the Fir tree (default /project/def-maxwl/mforooz/t128p2; the smoke is
#                   $T128P2_ROOT=/project/def-maxwl/mforooz/t128p2/smoke). Readings go to
#                   $T128P2_ROOT/readings/.
#   T128P2_RESULTS  the local results dir (default the main checkout's cruxvault/results/t128p2).
#   PANDAS_PY       python with pandas (default the candi-local env).
#
# 1. copies extract_records.py, verify_raw.py, extract.sh, verify.sh to $T128P2_ROOT/readings/;
# 2. submits extract.sh and verify.sh on Fir (CPU, --gres=none, --exclude=fc10713,fc10715), waits;
# 3. brings back ONLY the small tables (records/latent/cclean/runs/viabase/verify_collect);
# 4. copies the phase-1 tables knob/depth/fit as phase1_*.tsv (side column, Decision (vii));
# 5. runs analyze.py (pandas), verify_raw.py reduce (stdlib) and verify_compare.py -> verify.txt;
# 6. for the default (full) tree only: mirrors the results dir into this checkout's
#    cruxvault/results/t128p2/ when it differs.
# Exit 69/255 from ssh = the tunnel is down.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
DEFAULT_ROOT=/project/def-maxwl/mforooz/t128p2
T=${T128P2_ROOT:-$DEFAULT_ROOT}
R=$T/readings
RES=${T128P2_RESULTS:-/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/cruxvault/results/t128p2}
MIRROR=$REPO/cruxvault/results/t128p2
PHASE1=/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/cruxvault/results/t128
PANDAS_PY=${PANDAS_PY:-/Users/mforooz/miniforge3/envs/candi-local/bin/python}
H=fir
SCRIPTS=(extract_records.py verify_raw.py extract.sh verify.sh analyze.py verify_compare.py run_all.sh)
LOCAL=0
[ "${1:-}" = --local ] && LOCAL=1
mkdir -p "$RES"

remote() {  # remote <command>; stops the pipeline on a dead tunnel
  local rc=0
  ssh -o BatchMode=yes "$H" "$1" || rc=$?
  if [ "$rc" = 69 ] || [ "$rc" = 255 ]; then
    echo "ssh $H exited $rc: the tunnel is down. Run 'hpc up $H' in Terminal, then re-run." >&2
    exit "$rc"
  fi
  return "$rc"
}

if [ "$LOCAL" = 0 ]; then
  remote "mkdir -p $R && test -d $T/runs"
  rsync -a -e 'ssh -o BatchMode=yes' "$HERE/extract_records.py" "$HERE/verify_raw.py" \
    "$HERE/extract.sh" "$HERE/verify.sh" "$H:$R/" || { echo "rsync to $H failed" >&2; exit 1; }
  JOBS=()                      # "kind id"
  for k in extract verify; do
    id=$(remote "cd $R && sbatch --parsable --output=$R/${k}_%j.log $k.sh $T") || exit $?
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch $k.sh gave '$id'" >&2; exit 1; }
    JOBS+=("$k $id")
    printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$H" "$id" "$k.sh $T" >> "$RES/jobs.tsv"
    echo "$H: $k.sh job $id"
  done
  while :; do                  # wait for both jobs (sacct every 30 s)
    pending=0
    for j in "${JOBS[@]}"; do
      set -- $j
      s=$(remote "sacct -j $2 -X -n -P -o State") || exit $?
      s=$(echo "$s" | head -1 | tr -d ' ')
      case "$s" in
        COMPLETED) ;;
        PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING|"") pending=1 ;;
        *) echo "job $2 ($1) ended $s; log $R/${1}_$2.log" >&2; exit 1 ;;
      esac
    done
    [ "$pending" = 0 ] && break
    sleep 30
  done
  for j in "${JOBS[@]}"; do
    set -- $j
    rsync -rt -e 'ssh -o BatchMode=yes' "$H:$R/${1}_$2.log" "$RES/" || { echo "rsync from $H failed" >&2; exit 1; }
    grep -h -e 'assigned runs' -e 'via-base' -e 'collect:' "$RES/${1}_$2.log" || true
  done
  rsync -rt -e 'ssh -o BatchMode=yes' --include="*_$H.tsv.gz" --include="*_$H.tsv" \
    --include="verify_collect_$H.json" --exclude='*' "$H:$R/" "$RES/" \
    || { echo "rsync from $H failed" >&2; exit 1; }
fi

for t in knob_table depth_table fit_table; do
  if [ -f "$PHASE1/$t.tsv" ]; then cp "$PHASE1/$t.tsv" "$RES/phase1_$t.tsv"; fi
done
cp "${SCRIPTS[@]/#/$HERE/}" "$RES/"
"$PANDAS_PY" "$RES/analyze.py" | tee "$RES/analyze_out.txt"
"$PANDAS_PY" "$RES/verify_raw.py" reduce "$RES/verify_raw.json" "$RES/verify_collect_$H.json"
"$PANDAS_PY" "$RES/verify_compare.py"

cat > "$RES/FIR_PATH.txt" <<EOF
t128 phase 2 readings. Tree read: fir:$T   (T128P2_ROOT; the default full tree is $DEFAULT_ROOT)
  runs (96: DSfree, DSfixed, DSanchor, D x 8 g x 3 seeds)   fir:$DEFAULT_ROOT/runs/
  smoke (indices 9, 33, 57, 81; 300 steps)                fir:$DEFAULT_ROOT/smoke/
  via the base (24, from the new D maps)                  fir:$DEFAULT_ROOT/viabase/
  readings (extraction + second-method collection)        fir:$R/
  kit K1 (sha 524f4d7)                                    fir:$DEFAULT_ROOT/code/K1/
Training/law jobs (plan_t128p2.md Log): verify 453753; smoke train 453754, law 453755,
  via-base 453756; full train 470429 (0-95%24), law 470431 (aftercorr), via-base 470432.
Readings jobs (time, cluster, id, script), every run of run_all.sh into this dir:
$(cat "$RES/jobs.tsv" 2>/dev/null)
Rebuild everything: bash tools/t128p2/readings/run_all.sh   (tables only: add --local)
EOF

if [ "$T" = "$DEFAULT_ROOT" ] && \
   [ "$(cd "$RES" && pwd -P)" != "$(mkdir -p "$MIRROR" && cd "$MIRROR" && pwd -P)" ]; then
  rsync -a "$RES/" "$MIRROR/"
  echo "mirrored $RES -> $MIRROR"
fi
grep -c DISAGREE "$RES/verify.txt" | sed 's/^/DISAGREE lines: /' || true
