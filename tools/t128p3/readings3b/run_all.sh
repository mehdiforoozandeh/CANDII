#!/bin/bash
# t128 phase 3b readings: the whole pipeline in one command, from the laptop. Idempotent.
#
#   bash tools/t128p3/readings3b/run_all.sh            # Fir + tables
#   bash tools/t128p3/readings3b/run_all.sh --local    # tables from the files already here
#
# Environment:
#   T128P3B_ROOT     the Fir tree (default /project/def-maxwl/mforooz/t128p3/seeds3b); must be a
#                    seeds3b tree. Readings go to $T128P3B_ROOT/readings/.
#   T128P3B_RESULTS  the local results dir (default the main checkout's cruxvault/results/t128p3b).
#   T128P3B_RESCORE  control (b): the Fir dir of the rescore of one phase-2 D run through the base
#                    (holds viabase.json + law.json). Unset: control (b) reads "not run".
#   PANDAS_PY        python with pandas + scipy (default the candi-local env).
#
# 1. copies phase 3's extract_records.py and this extract.sh to $T/readings/, submits extract.sh on
#    Fir (CPU, --gres=none, --exclude=fc10713,fc10715), waits;
# 2. brings back ONLY the small tables (records/runs/viabase), the log, jobs.tsv, and the rescore's
#    viabase.json + law.json into rescore/;
# 3. copies seeds 0-2: Dbigf from the phase-3 mirror (p3_*), D from the phase-2 mirror (ref_*);
# 4. runs analyze.py (+ phase 3's analyze.py as analyze_p3.py) -> tables, checks.txt.
# Never writes into the t128p2 or t128p3 result mirrors. Exit 69/255 from ssh = the tunnel is down.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
P3=$HERE/../readings
DEFAULT_ROOT=/project/def-maxwl/mforooz/t128p3/seeds3b
T=${T128P3B_ROOT:-$DEFAULT_ROOT}
R=$T/readings
RESULTS=/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/cruxvault/results
RES=${T128P3B_RESULTS:-$RESULTS/t128p3b}
P3RES=$RESULTS/t128p3
P2RES=$RESULTS/t128p2
RESCORE=${T128P3B_RESCORE:-}
PANDAS_PY=${PANDAS_PY:-/Users/mforooz/miniforge3/envs/candi-local/bin/python}
H=fir
case "$T" in */t128p3/seeds3b|*/t128p3/seeds3b/*) ;; *) echo "T128P3B_ROOT=$T is not a seeds3b tree" >&2; exit 2 ;; esac
case "${RES%/}" in
  */t128p2|*/t128p2/*|*/t128p3|*/t128p3/*) echo "refusing to write into a t128p2/t128p3 mirror ($RES)" >&2; exit 2 ;;
esac
mkdir -p "$RES"
RESP=$(cd "$RES" && pwd -P)
for m in "$P3RES" "$P2RES"; do
  [ -d "$m" ] || continue
  MP=$(cd "$m" && pwd -P)
  case "$RESP" in "$MP"|"$MP"/*) echo "refusing: $RES resolves into $m" >&2; exit 2 ;; esac
done
P3_FILES=(records_fir.tsv.gz viabase_fir.tsv.gz runs_fir.tsv m1_never_table.tsv)
P2_FILES=(records_fir.tsv.gz viabase_fir.tsv.gz runs_fir.tsv)
for f in "${P3_FILES[@]}"; do [ -s "$P3RES/$f" ] || { echo "phase-3 file $P3RES/$f missing" >&2; exit 1; }; done
for f in "${P2_FILES[@]}"; do [ -s "$P2RES/$f" ] || { echo "phase-2 file $P2RES/$f missing" >&2; exit 1; }; done
LOCAL=0
[ "${1:-}" = --local ] && LOCAL=1

remote() {  # remote <command>; stops the pipeline on a dead tunnel
  local rc=0
  ssh -o BatchMode=yes "$H" "$1" || rc=$?
  if [ "$rc" = 69 ] || [ "$rc" = 255 ]; then
    echo "ssh $H exited $rc: the tunnel is down. Run 'hpc up $H' in Terminal, then re-run." >&2
    exit "$rc"
  fi
  return "$rc"
}
RS=(rsync -e 'ssh -o BatchMode=yes')

if [ "$LOCAL" = 0 ]; then
  remote "mkdir -p $R && test -d $T/runs"
  remote "test -s $R/jobs.tsv || printf 'purpose\tjob_id\tsubmitted_utc\n' > $R/jobs.tsv"
  "${RS[@]}" -a "$P3/extract_records.py" "$HERE/extract.sh" "$H:$R/" \
    || { echo "rsync to $H failed" >&2; exit 1; }
  id=$(remote "cd $R && sbatch --parsable --gres=none --exclude=fc10713,fc10715 --output=$R/extract_%j.log extract.sh $T") || exit $?
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch extract.sh gave '$id'" >&2; exit 1; }
  remote "printf 'readings3b extract ($T)\t$id\t$(date -u +%FT%TZ)\n' >> $R/jobs.tsv"
  echo "$H: extract.sh job $id"
  while :; do
    s=$(remote "sacct -j $id -X -n -P -o State") || exit $?
    s=$(echo "$s" | head -1 | tr -d ' ')
    case "$s" in
      COMPLETED) break ;;
      PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING|"") sleep 30 ;;
      *) echo "job $id ended $s; log $R/extract_$id.log" >&2; exit 1 ;;
    esac
  done
  "${RS[@]}" -rt "$H:$R/extract_$id.log" "$RES/" || { echo "rsync from $H failed" >&2; exit 1; }
  grep -h -e 'assigned runs' -e 'via-base' "$RES/extract_$id.log" || true
  "${RS[@]}" -rt --include="*_$H.tsv.gz" --include="*_$H.tsv" --include=jobs.tsv --exclude='*' \
    "$H:$R/" "$RES/" || { echo "rsync from $H failed" >&2; exit 1; }
  if [ -n "$RESCORE" ]; then
    mkdir -p "$RES/rescore"
    "${RS[@]}" -t "$H:$RESCORE/viabase.json" "$H:$RESCORE/law.json" "$RES/rescore/" \
      || { echo "rsync of the rescore $RESCORE failed" >&2; exit 1; }
  fi
fi

for f in "${P3_FILES[@]}"; do cp "$P3RES/$f" "$RES/p3_$f"; done
for f in "${P2_FILES[@]}"; do cp "$P2RES/$f" "$RES/ref_$f"; done
cp "$HERE/analyze.py" "$HERE/extract.sh" "$HERE/run_all.sh" "$P3/extract_records.py" "$RES/"
cp "$P3/analyze.py" "$RES/analyze_p3.py"
"$PANDAS_PY" "$RES/analyze.py" | tee "$RES/analyze_out.txt"

cat > "$RES/FIR_PATH.txt" <<EOT
t128 phase 3b readings. Tree read: fir:$T   (T128P3B_ROOT; default $DEFAULT_ROOT)
  runs (42: Dbigf, D x 7 per-track g x seeds 3-5)   fir:$T/runs/
  via the base (one per run)                         fir:$T/viabase/
  readings (extraction)                              fir:$R/
  seeds 0-2: Dbigf from fir:/project/def-maxwl/mforooz/t128p3 (local p3_* from $P3RES),
             D from fir:/project/def-maxwl/mforooz/t128p2 (local ref_* from $P2RES)
  control (b) rescore: ${RESCORE:-not given (control (b) not run)}
Every job (fir:$R/jobs.tsv; copy below):
$(cat "$RES/jobs.tsv" 2>/dev/null || echo "  (jobs.tsv not brought down yet)")
Rebuild: bash tools/t128p3/readings3b/run_all.sh   (tables only: add --local)
EOT
grep -c '^FAIL' "$RES/checks.txt" | sed 's/^/FAIL checks: /' || true
