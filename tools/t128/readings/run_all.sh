#!/bin/bash
# t128 readings: the whole pipeline in one command, from the laptop. Idempotent; re-run it when
# more runs land (the cluster step re-reads only runs whose DONE marks changed).
#
#   bash tools/t128/readings/run_all.sh            # clusters + tables + second method
#   bash tools/t128/readings/run_all.sh --local    # tables + second method from the files already here
#
# 1. copies extract_records.py, verify_raw.py, extract.sh, verify.sh to t128/readings/ on Fir and Nibi;
# 2. submits extract.sh and verify.sh on both (CPU, --gres=none; Fir gets --exclude=) and waits;
# 3. brings back ONLY the small tables (records/latent/cclean/runs/viabase/verify_collect);
# 4. runs analyze.py (pandas), verify_raw.py reduce (stdlib) and verify_compare.py -> verify.txt;
# 5. mirrors the results dir into this checkout's cruxvault/results/t128/ when it differs.
# Run outputs: Fir runs/ holds indices 0-196 and 225 (fir_idx.txt), Nibi runs/ the rest
# (nibi_idx.txt); each cluster reads only its own list. Exit 69/255 from ssh = a tunnel is down.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
RES=${T128_RESULTS:-/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/cruxvault/results/t128}
MIRROR=$REPO/cruxvault/results/t128
T126_D=/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/cruxvault/results/t126/records_D.tsv.gz
PANDAS_PY=${PANDAS_PY:-/Users/mforooz/miniforge3/envs/candi-local/bin/python}
T=/project/def-maxwl/mforooz/t128
R=$T/readings
SCRIPTS=(extract_records.py verify_raw.py extract.sh verify.sh analyze.py verify_compare.py run_all.sh)
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
  JOBS=()                      # "host kind id" (macOS bash 3.2: no associative arrays)
  for H in fir nibi; do
    remote "$H" "mkdir -p $R && test -s $T/${H}_idx.txt"
    rsync -a -e 'ssh -o BatchMode=yes' "$HERE/extract_records.py" "$HERE/verify_raw.py" \
      "$HERE/extract.sh" "$HERE/verify.sh" "$H:$R/" || { echo "rsync to $H failed" >&2; exit 1; }
    EX=""
    [ "$H" = fir ] && EX="--exclude="
    for k in extract verify; do
      id=$(remote "$H" "cd $R && sbatch --parsable $EX $k.sh $H") || exit $?
      [[ "$id" =~ ^[0-9]+$ ]] || { echo "$H: sbatch $k.sh gave '$id'" >&2; exit 1; }
      JOBS+=("$H $k $id")
      printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$H" "$id" "$k.sh $H" >> "$RES/jobs.tsv"
      echo "$H: $k.sh job $id"
    done
  done
  while :; do                  # wait for the four jobs (sacct every 30 s)
    pending=0
    for j in "${JOBS[@]}"; do
      set -- $j
      s=$(remote "$1" "sacct -j $3 -X -n -P -o State") || exit $?
      s=$(echo "$s" | head -1 | tr -d ' ')
      case "$s" in
        COMPLETED) ;;
        PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING|"") pending=1 ;;
        *) echo "$1 job $3 ($2) ended $s; log $R/${2}_$3.log" >&2; exit 1 ;;
      esac
    done
    [ "$pending" = 0 ] && break
    sleep 30
  done
  for j in "${JOBS[@]}"; do
    set -- $j
    rsync -a -e 'ssh -o BatchMode=yes' "$1:$R/${2}_$3.log" "$RES/" || { echo "rsync from $1 failed" >&2; exit 1; }
    grep -h -e 'assigned runs' -e 'via-base' -e 'collect:' "$RES/${2}_$3.log" || true
  done
  for H in fir nibi; do
    rsync -a -e 'ssh -o BatchMode=yes' --include="*_$H.tsv.gz" --include="*_$H.tsv" \
      --include="verify_collect_$H.json" --exclude='*' "$H:$R/" "$RES/" \
      || { echo "rsync from $H failed" >&2; exit 1; }
  done
fi

[ -f "$RES/records_D_t126.tsv.gz" ] || cp "$T126_D" "$RES/records_D_t126.tsv.gz"
cp "${SCRIPTS[@]/#/$HERE/}" "$RES/"
"$PANDAS_PY" "$RES/analyze.py" | tee "$RES/analyze_out.txt"
"$PANDAS_PY" "$RES/verify_raw.py" reduce "$RES/verify_raw.json" "$RES"/verify_collect_*.json
"$PANDAS_PY" "$RES/verify_compare.py"

cat > "$RES/FIR_PATH.txt" <<EOF
t128 readings. The runs live on TWO clusters, same path on both: $T
  Fir  runs/  indices 0-196 and 225 (fir_idx.txt; 198 runs)   fir:$T/runs/
  Nibi runs/  indices 197-224 and 226-239 (nibi_idx.txt; 42)   nibi:$T/runs/
  Copies of a run on the other cluster (old smoke attempts) are ignored.
  Via the base (24 runs, Nibi):                                 nibi:$T/viabase/
  Old D reference (t126, read only):                            nibi:/project/def-maxwl/mforooz/t126/D/runs/
                                                                (local copy: records_D_t126.tsv.gz)
  Readings (extraction + second-method collection), both:       $R/
Training/law jobs (plan_t128.md Log): Nibi smoke 24757111-14, 24772322/3, 24775599/600, 24797365/6;
  Fir smoke 61556, 61593, 61594; Fir full 70473/70474, DSbox 93258/93259, reruns 127388/9,
  129233/4, 133214/5, 214375/6/7; Nibi full 24963690/1; Nibi via-base 24896767.
Readings jobs (cluster, id, script), every run of run_all.sh:
$(cat "$RES/jobs.tsv" 2>/dev/null)
Rebuild everything: bash tools/t128/readings/run_all.sh   (tables only: add --local)
EOF

if [ "$(cd "$RES" && pwd -P)" != "$(mkdir -p "$MIRROR" && cd "$MIRROR" && pwd -P)" ]; then
  rsync -a "$RES/" "$MIRROR/"
  echo "mirrored $RES -> $MIRROR"
fi
grep -c DISAGREE "$RES/verify.txt" | sed 's/^/DISAGREE lines: /' || true
