#!/bin/bash
# t128 phase 3 readings: the whole pipeline in one command, from the laptop. Idempotent; re-run it
# when more runs land (the Fir step re-reads only runs whose DONE marks changed).
#
#   bash tools/t128p3/readings/run_all.sh            # Fir + tables + second method
#   bash tools/t128p3/readings/run_all.sh --local    # tables + second method from the files here
#
# Environment:
#   T128P3_ROOT     the Fir tree (default /project/def-maxwl/mforooz/t128p3; the smoke is
#                   T128P3_ROOT=/project/def-maxwl/mforooz/t128p3/smoke). Readings go to
#                   $T128P3_ROOT/readings/.
#   T128P3_RESULTS  the local results dir (default the main checkout's cruxvault/results/t128p3).
#                   A non-default T128P3_ROOT needs a non-default T128P3_RESULTS.
#   PANDAS_PY       python with pandas (default the candi-local env).
#
# 1. copies extract_records.py, verify_raw.py, extract.sh, verify.sh to $T128P3_ROOT/readings/;
# 2. submits extract.sh and verify.sh on Fir (CPU, --gres=none, --exclude=fc10713,fc10715), waits;
# 3. brings back ONLY the small tables (records/runs/viabase/verify_collect) and jobs.tsv;
# 4. copies the phase-2 reference files (records, viabase, runs, m1, m9; never recomputed on Fir)
#    from the main checkout's cruxvault/results/t128p2/ as ref_*;
# 5. runs analyze.py (pandas; T128P3_FULL_TREE=1 on the full tree: every run max_steps 15,000 or
#    stop), verify_raw.py reduce (stdlib) and verify_compare.py -> verify.txt;
# 6. for the default (full) tree only: mirrors the results dir into this checkout's
#    cruxvault/results/t128p3/ when it differs.
# Exit 69/255 from ssh = the tunnel is down.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
DEFAULT_ROOT=/project/def-maxwl/mforooz/t128p3
T=${T128P3_ROOT:-$DEFAULT_ROOT}
R=$T/readings
DEFAULT_RES=/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/cruxvault/results/t128p3
RES=${T128P3_RESULTS:-$DEFAULT_RES}
case "$T $RES" in *t128p2*) echo "refusing a t128p2 path (T128P3_ROOT=$T, T128P3_RESULTS=$RES)" >&2; exit 2 ;; esac
REF=/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/cruxvault/results/t128p2
REF_FILES=(records_fir.tsv.gz viabase_fir.tsv.gz runs_fir.tsv m1_never_table.tsv m9_steps_table.tsv)
# The job ledger: $R/jobs.tsv on Fir (purpose, job_id, submitted_utc). Whoever submits a t128p3
# job appends a row; this script appends its extract/verify jobs and FIR_PATH.txt lists the file.
LEDGER_SEED="purpose\tjob_id\tsubmitted_utc"
MIRROR=$REPO/cruxvault/results/t128p3
PANDAS_PY=${PANDAS_PY:-/Users/mforooz/miniforge3/envs/candi-local/bin/python}
H=fir
SCRIPTS=(extract_records.py verify_raw.py extract.sh verify.sh analyze.py verify_compare.py run_all.sh)
LOCAL=0
[ "${1:-}" = --local ] && LOCAL=1
realp() { if [ -d "$1" ]; then (cd "$1" && pwd -P); else echo "${1%/}"; fi; }
if [ "$T" != "$DEFAULT_ROOT" ]; then   # a non-full tree must never write the full tree's results
  if [ -z "${T128P3_RESULTS:-}" ] || [ "$(realp "$T128P3_RESULTS")" = "$(realp "$DEFAULT_RES")" ]; then
    echo "T128P3_ROOT=$T is not the full tree: set T128P3_RESULTS to a dir other than $DEFAULT_RES" >&2
    exit 2
  fi
  export T128P3_FULL_TREE=0
else
  export T128P3_FULL_TREE=1          # analyze.py then requires max_steps == 15000 on every run
fi
for f in "${REF_FILES[@]}"; do
  [ -s "$REF/$f" ] || { echo "reference file $REF/$f missing" >&2; exit 1; }
done
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
  remote "test -s $R/jobs.tsv || printf '$LEDGER_SEED\n' > $R/jobs.tsv"
  rsync -a -e 'ssh -o BatchMode=yes' "$HERE/extract_records.py" "$HERE/verify_raw.py" \
    "$HERE/extract.sh" "$HERE/verify.sh" "$H:$R/" || { echo "rsync to $H failed" >&2; exit 1; }
  JOBS=()                      # "kind id"
  for k in extract verify; do
    id=$(remote "cd $R && sbatch --parsable --output=$R/${k}_%j.log $k.sh $T") || exit $?
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch $k.sh gave '$id'" >&2; exit 1; }
    JOBS+=("$k $id")
    remote "printf 'readings $k ($T)\t$id\t$(date -u +%FT%TZ)\n' >> $R/jobs.tsv"
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
    --include="verify_collect_$H.json" --include=jobs.tsv --exclude='*' "$H:$R/" "$RES/" \
    || { echo "rsync from $H failed" >&2; exit 1; }
fi

for f in "${REF_FILES[@]}"; do cp "$REF/$f" "$RES/ref_$f"; done
cp "${SCRIPTS[@]/#/$HERE/}" "$RES/"
"$PANDAS_PY" "$RES/analyze.py" | tee "$RES/analyze_out.txt"
"$PANDAS_PY" "$RES/verify_raw.py" reduce "$RES/verify_raw.json" "$RES/verify_collect_$H.json"
"$PANDAS_PY" "$RES/verify_compare.py"

cat > "$RES/FIR_PATH.txt" <<EOF
t128 phase 3 readings. Tree read: fir:$T   (T128P3_ROOT; the default full tree is $DEFAULT_ROOT)
  runs (48: Dbigg, Dbigf x 8 g x 3 seeds; 15,000-step cap)  fir:$DEFAULT_ROOT/runs/
  smoke (indices 9, 33)                                     fir:$DEFAULT_ROOT/smoke/
  via the base (one per run, from the chained law job)      fir:$DEFAULT_ROOT/viabase/
  readings (extraction + second-method collection)          fir:$R/
  kit                                                       fir:$DEFAULT_ROOT/code/
  reference (phase 2 arm D + its via-base, not rerun)       fir:/project/def-maxwl/mforooz/t128p2/{runs/D_*,viabase/}
                                                            local copy: $REF (ref_* here)
Every job (fir:$R/jobs.tsv, appended by whoever submits; copy below):
$(cat "$RES/jobs.tsv" 2>/dev/null || echo "  (jobs.tsv not brought down yet)")
Rebuild everything: bash tools/t128p3/readings/run_all.sh   (tables only: add --local)
EOF

if [ "$T" = "$DEFAULT_ROOT" ] && \
   [ "$(cd "$RES" && pwd -P)" != "$(mkdir -p "$MIRROR" && cd "$MIRROR" && pwd -P)" ]; then
  rsync -a "$RES/" "$MIRROR/"
  echo "mirrored $RES -> $MIRROR"
fi
grep -c DISAGREE "$RES/verify.txt" | sed 's/^/DISAGREE lines: /' || true
grep -c '^FAIL' "$RES/checks.txt" | sed 's/^/FAIL checks: /' || true
