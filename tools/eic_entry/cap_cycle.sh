#!/bin/bash
# One pass of the depth-capped validation board (crux task t390, fix b), run on Fir every ~10 min.
# For each track whose raw prediction is complete:
#   ChIP-seq with depth_reads > 20 M: build its capped counts (depth_cap.py, one job), then score them.
#   other tracks (unchanged by the cap): copy the raw board's score once it exists.
# After copies alone, rebuild the board. Markers in $V/capwatch/{built,scored}/<track>.
#   cap_cycle.sh <track> ...
set -uo pipefail
V=/scratch/mforooz/eicx/val
PRED=/project/def-maxwl/mforooz/ONION/runs_rcp/eic_val/rcp_combo_w2x_unet_s0
RAW=$V/rcp_combo_w2x_unet_s0_raw
CAP=$V/rcp_s0_depthcap
E=/scratch/mforooz/eicx/entry/code/tools/eic_entry
PY=$V/slope_cal/venv/bin/python
W=$V/capwatch
mkdir -p $W/built $W/scored $W/logs
LIMIT=${CPUH_LIMIT:-55}   # stop submitting once this run's eicx_ jobs since 2026-11-24 19:30 local reach the limit
used=$(sacct -u $USER -S 2026-11-24T19:30 -X -n -P -o JobName,CPUTimeRAW | awk -F'|' '$1 ~ /^eicx_/ {s += $2} END {printf "%.1f", s / 3600}')
awk -v u=$used -v l=$LIMIT 'BEGIN {exit !(u >= l)}' && { echo "limit: $used of $LIMIT CPU-h used, nothing submitted"; exit 0; }
copied=0
for t in "$@"; do
  [ -e $W/scored/$t ] && continue
  f=$PRED/counts/$t.h5
  [ -f $f ] && [ $(stat -c %s $f) -eq 485110096 ] && [ $(( $(date +%s) - $(stat -c %Y $f) )) -gt 600 ] || continue
  read route depth < <(awk -F'\t' -v t=$t '$1 == t {print $2, $3}' $V/cov.tsv)
  if [ "$route" = chip ] && [ "${depth%.*}" -gt 20000000 ]; then
    P=$V/depth_cap2; [ -f $P/counts/$t.h5 ] || P=$W/tracks/$t
    if [ -f $P/counts/$t.h5 ] && [ -f $P/cap.tsv ]; then   # cap.tsv is written last
      (cd $V && BOARD=validation bash $E/run_entry.sh submit $P $P/cov.tsv $CAP $t) | grep -E "^$t|^collect"
      touch $W/scored/$t
    elif [ ! -e $W/built/$t ]; then
      mkdir -p $W/tracks/$t
      j=$(sbatch --parsable --account=def-maxwl --gres=none --job-name=eicx_entry_capbuild_$t --cpus-per-task=2 \
        --mem=8G --time=0:30:00 --output=$W/logs/%x_%j.out \
        --wrap="$PY $E/depth_cap.py --pred $PRED --cov $V/cov.tsv --out $W/tracks/$t --tracks $t")
      echo "build $t $j"; touch $W/built/$t
    fi
  elif [ -f $RAW/scores/validation_$t.tsv ]; then
    cp $RAW/scores/validation_$t.tsv $CAP/scores/; mkdir -p $CAP/raw/$t; cp $RAW/raw/$t/$t.npy.json $CAP/raw/$t/
    echo "reused raw score $t" | tee -a $CAP/JOBS.txt; touch $W/scored/$t; copied=1
  fi
done
[ $copied = 1 ] && (cd $V && BOARD=validation bash $E/run_entry.sh collect $CAP) | grep collect
exit 0
