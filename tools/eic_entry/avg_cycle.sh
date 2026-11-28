#!/bin/bash
# One pass of the averaged board (final + base model mean, then the 20 M depth cap; crux task t390), run on
# Fir every ~10 min after the depth-capped board. Second board only; the depth-capped board is primary.
#   AVG="<tracks with base counts at the same covariate row>" ONLY="<final-model-only tracks>" avg_cycle.sh
# AVG track: once the depth-capped board has submitted it, build its averaged counts (one job), then score.
# ONLY track: copy its depth-capped score (same computation) once it exists.
# Lower priority than the depth-capped board: nothing is submitted while this run's spend plus 1.1 CPU-h per
# priority track not yet scored on both boards (PRIO) reaches CPUH_LIMIT (default 80).
set -uo pipefail
V=/scratch/mforooz/eicx/val
FINAL=/project/def-maxwl/mforooz/ONION/runs_rcp/eic_val/rcp_combo_w2x_unet_s0
BASE=/project/def-maxwl/mforooz/ONION/runs_rcp/eic_val/eic_v1_2_rung4_base_heldout_s0
CAP=$V/rcp_s0_depthcap
OUT=$V/rcp_s0_avgbase_cap
E=/scratch/mforooz/eicx/entry/code/tools/eic_entry
PY=$V/slope_cal/venv/bin/python
W=$V/avgwatch
mkdir -p $W/built $W/scored $W/logs $OUT/scores
LIMIT=${CPUH_LIMIT:-80}
used=$(sacct -u $USER -S 2026-11-24T19:30 -X -n -P -o JobName,CPUTimeRAW | awk -F'|' '$1 ~ /^eicx_/ {s += $2} END {printf "%.1f", s / 3600}')
owe=0
for t in ${PRIO:-}; do
  [ -f $V/rcp_combo_w2x_unet_s0_raw/scores/validation_$t.tsv ] || owe=$((owe + 1))
  [ -f $CAP/scores/validation_$t.tsv ] || owe=$((owe + 1))
done
if awk -v u=$used -v o=$owe -v l=$LIMIT 'BEGIN {exit !(u + 1.1 * o >= l)}'; then
  echo "limit: $used CPU-h used + $owe priority scores owed (1.1 each) >= $LIMIT; no averaged-board submission"
  ALLOW=0
else
  ALLOW=1
fi
copied=0
for t in ${ONLY:-}; do
  [ -e $W/scored/$t ] && continue
  if [ -f $CAP/scores/validation_$t.tsv ]; then
    cp $CAP/scores/validation_$t.tsv $OUT/scores/; mkdir -p $OUT/raw/$t; cp $CAP/raw/$t/$t.npy.json $OUT/raw/$t/
    echo "final-model-only $t (depth-capped score copied)" | tee -a $OUT/JOBS.txt; touch $W/scored/$t; copied=1
  fi
done
for t in ${AVG:-}; do
  [ -e $W/scored/$t ] && continue
  [ -e $V/capwatch/scored/$t ] || [ -f $CAP/scores/validation_$t.tsv ] || continue   # after the primary board
  P=$W/tracks/$t
  if [ -f $P/cap.tsv ] && [ -f $P/counts/$t.h5 ]; then
    [ $ALLOW = 1 ] || continue
    (cd $V && BOARD=validation bash $E/run_entry.sh submit $P $P/cov.tsv $OUT $t) | grep -E "^$t|^collect"
    touch $W/scored/$t
  elif [ ! -e $W/built/$t ] && [ $ALLOW = 1 ]; then
    mkdir -p $P
    j=$(sbatch --parsable --account=def-maxwl --gres=none --job-name=eicx_entry_avgbuild_$t --cpus-per-task=2 \
      --mem=8G --time=0:30:00 --output=$W/logs/%x_%j.out \
      --wrap="$PY $E/depth_cap.py --pred $FINAL --pred2 $BASE --cov $V/cov.tsv --out $P --tracks $t")
    echo "build $t $j"; touch $W/built/$t
  fi
done
[ $copied = 1 ] && (cd $V && BOARD=validation bash $E/run_entry.sh collect $OUT) | grep collect
exit 0
