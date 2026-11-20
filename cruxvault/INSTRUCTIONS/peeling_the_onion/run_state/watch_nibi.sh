#!/bin/bash
# Foreman watch of the Nibi EIC build. Emits a line only on an event. Args: baseline failed-like count of the original arrays.
base="${1:-47}"; seenf=""; seeno=""; seens=0
t1=0; t2=0; smz=0; qz=0
while true; do
  now=$(date -u +%Y%m%d%H%M)
  if [ $t1 -eq 0 ] && [ "$now" -ge 202611131230 ] && [ ! -e "$(dirname "$0")/.d6b_launched" ]; then echo "TIME 12:30 UTC: launch the collect of complete rows (D6b)"; t1=1; fi
  if [ $t2 -eq 0 ] && [ "$now" -ge 202611131350 ] && [ ! -e "$(dirname "$0")/.d6f_launched" ]; then echo "TIME 13:50 UTC: launch the freeze builder (D6f; build at 14:30 UTC)"; t2=1; fi
  out=$(hpc run nibi 'squeue -u mforooz -h -r -o "%i %j %T"; echo ---; sacct -j 27087748,27087756,27087789,27081348,27081352,27081354,27081357,27079130,27105143,27112838,27117064,27123484,27124280,27124284,27143106,27150100,27161195,27165180,27165188,27165190,27165192,27165194,27167692,27167694 -X -n -o JobID%20,State%14 | grep -vE "COMPLETED|RUNNING|PENDING" | sort; echo ---; sacct -j 26853261,26853270,26855580,26855582,26857540 -X -n -o State | grep -c -E "FAILED|TIMEOUT|OUT_OF"' 2>/dev/null); rc=$?
  if [ $rc -eq 69 ]; then echo "TUNNEL DOWN (hpc exit 69)"; exit 0; fi
  if [ $rc -eq 0 ] && echo "$out" | grep -q "onion_"; then
    q=$(echo "$out" | awk '/^---$/{n++; next} n==0'); f=$(echo "$out" | awk '/^---$/{n++; next} n==1' | tr -s ' ' | tr '\n' ';'); o=$(echo "$out" | awk '/^---$/{n++; next} n==2')
    b=$(echo "$q" | grep -c onion_build); s=$(echo "$q" | grep -c onion_signal); sm=$(echo "$q" | grep -c -E "^(27081348|27081352|27081354|27081357)_")
    if [ -n "$f" ] && [ "$f" != "$seenf" ]; then echo "FAILED TASKS (resubmitted, smoke, row 396): $f"; seenf="$f"; fi
    if [ -n "$o" ] && [ "$o" != "$base" ] && [ "$o" != "$seeno" ]; then echo "ORIGINAL ARRAYS: failed-like task count is now $o (baseline $base)"; seeno="$o"; fi
    if [ "$sm" -eq 0 ]; then smz=$((smz+1)); else smz=0; fi
    if [ "$smz" -ge 2 ] && [ $seens -eq 0 ]; then echo "SMOKES ENDED (renote can run from kit D6y)"; seens=1; fi
    if [ "$b" -le 6 ]; then qz=$((qz+1)); else qz=0; fi
    if [ "$qz" -ge 2 ]; then echo "QUEUE EVENT (two readings): onion_build=$b onion_signal=$s"; exit 0; fi
  fi
  sleep 300
done
