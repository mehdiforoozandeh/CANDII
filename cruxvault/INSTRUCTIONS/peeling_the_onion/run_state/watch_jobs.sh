#!/bin/bash
# Emits one line when every job id given has left the queue (two consecutive readings), with their sacct states.
ids="$1"; label="$2"; z=0
while true; do
  out=$(hpc run nibi "squeue -u mforooz -h -o '%i %T'; echo ---OK" 2>/dev/null); rc=$?
  if [ $rc -eq 69 ]; then echo "TUNNEL DOWN (hpc exit 69)"; exit 0; fi
  if [ $rc -eq 0 ] && echo "$out" | grep -q -- "---OK"; then
    live=0; for i in $(echo "$ids" | tr ',' ' '); do echo "$out" | grep -q -E "^${i}(_| )" && live=$((live+1)); done
    if [ $live -eq 0 ]; then z=$((z+1)); else z=0; fi
    if [ $z -ge 2 ]; then st=$(hpc run nibi "sacct -j $ids -X -n -o JobID%14,State%12 | tr -s ' ' | tr '\n' ';'" 2>/dev/null); echo "JOBS ENDED ($label): $st"; exit 0; fi
  fi
  sleep 240
done
