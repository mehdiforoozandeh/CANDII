#!/bin/bash
# Close-out D4 (peeling the onion): build every remaining priority-4 row of the EIC build list
# (721 of 722 rows; row 1074 is built) and rerun the 10 build tasks that failed with NODE_FAIL in
# the Nibi outage of 2026-11-15. Dry run is the default. Runs on a Nibi login node, e.g.
#
#   ssh -o BatchMode=yes nibi 'bash -s -- 1'           < slurm/onion_data/closeout_d4_submit.sh
#   ssh -o BatchMode=yes nibi 'DRY_RUN=0 bash -s -- 1' < slurm/onion_data/closeout_d4_submit.sh
#
# Argument: the batches to handle, "1", "2", "3" or "2 3". Order (the blind tracks finish first):
#   batch 1  51 blind base rows + the 11 blind control experiments they need (62 rows), and the
#            outage reruns (7 EIC training DNase-seq rows, 3 second-experiment DNase-seq rows)
#   batch 2  the 147 FASTQ-level arms of the blind experiments (crop36, crop50, se)
#   batch 3  the 512 per-library and per-lane rows (training and validation experiments)
# Batches 2 and 3 are refused while any of the 62 blind rows (496-557) is queued or running;
# the outage reruns do not hold them (FORCE=1 overrides).
# Batch 3 stages nothing new (its FASTQs are on /scratch); batch 2 reads the FASTQs batch 1 staged.
#
# Every call is `build_row.sh submit` of kit KF4 (KX4 for the second experiments): it skips rows that
# are built, NOT_APPLICABLE, queued or running, so a second run of this script submits nothing twice.
# Row 56 is not rerun here: its pending signal task 27124285_56 (7 days, 192000M) rebuilds it.
# Costs, per-row limits and their sources: cruxvault/results/closeout_D4/ (rows_plan.tsv, BUILD.md).
#
# Env: DRY_RUN (default 1: print the pending check and the sbatch lines, submit nothing),
#      CONC (build tasks per array, default 50), SCONC (stage tasks per array, default 2), FORCE.
set -uo pipefail
DRY_RUN="${DRY_RUN:-1}"; CONC="${CONC:-50}"; SCONC="${SCONC:-2}"; FORCE="${FORCE:-0}"
BATCHES="${*:-}"
[ -n "$BATCHES" ] || { echo "usage: closeout_d4_submit.sh <batch...>  (1, 2, 3 or '2 3')"; exit 2; }

O=/project/def-maxwl/mforooz/ONION
# KF4H/KX4H = KF4/KX4 plus one line: an honest USER_AGENT (repo 8e9c4af2). ENCODE answered 405 to the
# browser-style header on 2026-11-23 and every batch-1 stage task failed; PI approved the change 2026-11-24.
KF4=/project/def-maxwl/mforooz/onion_kit/KF4H; KX4=/project/def-maxwl/mforooz/onion_kit/KX4H
B=$O/eic/build_list.tsv; BX=$O/eic/other_experiment_build_list.tsv
M=/scratch/mforooz/onion/measurements; W=/scratch/mforooz/onion/work
B_SHA=33c077f464138c57cbe31827857061b1524a22ed571d00a7ab8b7de81ddc0b83
BX_SHA=267443cdf86c4f6aea619da6bbe06f77d810381c5eec532b5326f8bca7ea87bc
SCRATCH_CAP_KB=$((16 * 1024 * 1024 * 1024))   # refuse above 16 TiB (hard limit 20 TiB)

# --- rows (from rows_plan.tsv; one line per submit call: batch|kit|list|priority|time|mem|rows|label)
CALLS=(
"1|KF4|B|4|3-00:00:00||496-500,502-520,522-555,557|blind ChIP-seq and ATAC-seq base rows + their 11 controls (one call: the signal pass waits for the controls)"
"1|KF4|B|4|7-00:00:00||501,521,556|blind DNase-seq base rows (paired-end)"
"1|KF4|B|2|7-00:00:00||113,211,216,232|outage rerun: EIC training DNase-seq (rule B.5: 7 days)"
"1|KF4|B|2|7-00:00:00|192000M|384,385,387|outage rerun: EIC training DNase-seq (as their last run: 192000M)"
"1|KX4|BX|5|7-00:00:00||2049,2075|outage rerun: second-experiment DNase-seq"
"1|KX4|BX|5|7-00:00:00|192000M|2079|outage rerun: second-experiment DNase-seq, 1,115 M reads"
"2|KF4|B|4|1-00:00:00||558-704|blind arms crop36, crop50, se"
"3|KF4|B|4|7-00:00:00||726,728,732,734,739-740,743,745,747,749,752,754,756,758,799-801,809-814,983-984,1067-1068,1077-1080,1090-1091,1094-1095,1130-1131,1134-1145,1166-1172,1184-1193,1209-1210,1216-1217|paired-end DNase-seq library and lane rows (slow bwa sampe)"
"3|KF4|B|4|1-00:00:00||705-725,727,729-731,733,735-738,741-742,744,746,748,750-751,753,755,757,759-798,802-808,815-982,985-1066,1069-1073,1075-1076,1081-1089,1092-1093,1096-1129,1132-1133,1146-1165,1173-1183,1194-1208,1211-1215|other library and lane rows"
)
BATCH1_B="496-557"   # the gate waits for the blind rows only, not for the 7-day outage reruns

die() { echo "REFUSED: $*"; exit 3; }
case "$(hostname)" in *nibi*) ;; *) die "run this on a Nibi login node";; esac
grep -q "git=0d578dc" "$KF4/.kit_stamp" || die "kit KF4 is not git 0d578dc"
grep -q "git=2f79ec4" "$KX4/.kit_stamp" || die "kit KX4 is not git 2f79ec4"
for k in "$KF4" "$KX4"; do grep -q "patch=USER_AGENT" "$k/.kit_stamp" || die "kit $k lacks the USER_AGENT patch"; done
[ "$(sha256sum "$B" | cut -d' ' -f1)" = "$B_SHA" ] || die "build_list.tsv changed"
[ "$(sha256sum "$BX" | cut -d' ' -f1)" = "$BX_SHA" ] || die "other_experiment_build_list.tsv changed"

ACTIVE=$(squeue -u "$USER" -h -r -n onion_build,onion_signal,onion_stage -o '%K' 2>/dev/null \
         | grep -E '^[0-9]+$' | sort -un | paste -sd, -)
echo "=== closeout D4 build $(date -u +%FT%TZ) batches: $BATCHES  DRY_RUN=$DRY_RUN CONC=$CONC SCONC=$SCONC"
echo "queued or running onion_* array tasks (all build lists): ${ACTIVE:-none}"
USED_KB=$(quota -w 2>/dev/null | awk '$1 ~ /scratch/ {gsub(/\*/, "", $2); print $2}')
echo "/scratch used: ${USED_KB:-?} KB; fastq dir: $(du -sh /scratch/mforooz/onion/fastq 2>/dev/null | cut -f1)"
[ -n "$USED_KB" ] || [ "$FORCE" = 1 ] || die "cannot read the /scratch quota (FORCE=1 to override)"
[ -z "$USED_KB" ] || [ "$USED_KB" -lt "$SCRATCH_CAP_KB" ] || [ "$FORCE" = 1 ] \
  || die "/scratch above 16 TiB; batch 1 stages 1,227 GB and keeps about 1.4 TB of blind BAMs"

if [ "$BATCHES" != 1 ] && [ "$FORCE" != 1 ]; then   # batch 1 must have left the queue
  for pair in "$B:$BATCH1_B"; do
    L="${pair%%:*}"; S="${pair#*:}"
    if [ -n "$ACTIVE" ]; then
      left=$(python3 - "$S" "$ACTIVE" <<'EOF'
import sys
def p(s):
    out = set()
    for t in s.split(","):
        a, _, b = t.partition("-"); out.update(range(int(a), int(b or a) + 1))
    return out
print(",".join(map(str, sorted(p(sys.argv[1]) & p(sys.argv[2])))))
EOF
)
      [ -z "$left" ] || die "batch 1 rows still queued or running in $(basename "$L"): $left"
    fi
  done
fi

LOG=/dev/null
if [ "$DRY_RUN" = 0 ]; then
  LOG=$O/logs/closeout_d4_submit_$(date -u +%Y%m%dT%H%M%SZ).log
  echo "submit record: $LOG"
fi
for c in "${CALLS[@]}"; do
  IFS='|' read -r BATCH KIT LIST PRI TIME MEM ROWS LABEL <<< "$c"
  case " $BATCHES " in *" $BATCH "*) ;; *) continue;; esac
  K=$KF4; [ "$KIT" = KX4 ] && K=$KX4
  L=$B; [ "$LIST" = BX ] && L=$BX
  echo "--- batch $BATCH: $LABEL"
  echo "    kit $KIT, list $(basename "$L"), priority $PRI, time $TIME, mem ${MEM:-64000M (header)}, rows $ROWS"
  if [ "$DRY_RUN" != 0 ]; then
    python3 "$K/tools/onion_data/build_list.py" pending --build-list "$L" --priority "$PRI" --rows "$ROWS" \
      --out "$M" --work "$W" --files "$O/catalog/encode/files.tsv" ${ACTIVE:+--active "$ACTIVE"} 2>&1 >/dev/null \
      | awk '{n[$1]++} /^skip/{print "    " $0} END{printf "    pending check: %d submit, %d skip\n", n["submit"], n["skip"]}'
    env ONION_SBATCH_TEST=1 ONION_TIME="$TIME" ${MEM:+ONION_MEM=$MEM} \
      bash "$K/slurm/onion_data/build_row.sh" submit "$K" "$L" "$PRI" "$CONC" "$SCONC" "$ROWS" 2>/dev/null \
      | sed 's/^/    /'
  else
    { echo "--- batch $BATCH: $LABEL ($(date -u +%FT%TZ))"
      env ONION_TIME="$TIME" ${MEM:+ONION_MEM=$MEM} \
        bash "$K/slurm/onion_data/build_row.sh" submit "$K" "$L" "$PRI" "$CONC" "$SCONC" "$ROWS" 2>&1; } \
      | tee -a "$LOG" | grep -v '^skip\|^submit row'
  fi
done
echo "=== done. Dry run: $([ "$DRY_RUN" = 0 ] && echo no || echo 'yes (nothing submitted)')"
