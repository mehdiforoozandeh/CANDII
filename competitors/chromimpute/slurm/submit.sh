#!/bin/bash
# Lay out a run directory, build the work-item lists, and submit the six stages as one dependency
# chain of SLURM arrays.
#
#   bash submit.sh [RUNDIR] [FROM_STAGE] [TARGETS]
#
#   pilot  bash submit.sh $S/pilot                             # 20 targets, the eval scope
#   full   bash submit.sh $S/eic_19 prepare targets.tsv        # every declared target
#
# RETARGETED 2026-08-31. `P1`/`P2` are RETIRED (BENCHMARK_DESIGN.md §9), and the genome-wide run is
# GONE: §4 blanks ChromImpute's `genome-wide` cell and rules that a blanked cell is not computed, so
# the only scope is the regime's eval_chroms — chr20+21+22. `CI_REGIME` selects which live regime
# (eic_19 or eic_pilot); `CI_CHROMS` defaults to that regime's eval_chroms.
#
# RULE 2, SETTLED 2026-09-01 (PI ruling). `dist` and `gtd` run on the regime's TRAINING loci, in
# every regime and not only under a `regions` BED: `prepare.py` always writes `chrominfo.train.txt`
# and `stage.sh` hands that file, and only that file, to ComputeGlobalDist and GenerateTrainData.
# `Apply` keeps `chrominfo.txt` — its neighbour features are per-position inference, which §2 Rule 2
# names as legitimate on the eval chromosomes. This DEPARTS from the published recipe, which samples
# inside the predicted chromosomes; the departure is recorded in `../README.md`.
#
# ONE THING THIS SCRIPT DOES NOT DO, AND IT IS A BLOCKER, NOT AN OVERSIGHT:
#
#   `targets.tsv` is EVERY declared eval pair, and the live regimes declare 38 — 26 V_ AND 12 B_.
#   §5 rules B_ is touched ONCE, at the very end. Filter the target list to V_ before running
#   anything but the final B_ pass.
#
# Nothing here decides science. The compendium is every training track the regime declares (§6.2,
# enforced in prepare.training_tracks); the pilot targets come from prepare.pilot_subset; every
# ChromImpute flag is a paper default except the ones that only choose what to parallelize over.
set -euo pipefail

RUN=${1:-$HOME/scratch/t51_chromimpute/pilot_chr21}
FROM=${2:-prepare}
TARGETS=${3:-targets_pilot.tsv}
CHROMS=${CI_CHROMS:-chr20,chr21,chr22}
REPO=${CI_REPO:-/project/def-maxwl/mforooz/CANDII_t78_code}
STORE=${CI_STORE:-/project/def-maxwl/mforooz/CANDI_STORE/eic}
PY=${CI_PY:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv/bin/python}
REGIME=${CI_REGIME:-$REPO/configs/regime.eic_19.json}
JAR=${CI_JAR:-$HOME/scratch/t51_chromimpute/tool/ChromImpute.jar}
ACCT=${CI_ACCT:-def-maxwl}
HERE=$REPO/competitors/chromimpute
STAGE=$HERE/slurm/stage.sh

mkdir -p "$RUN"/{lists,logs,timing}

# --- the four text files, and the target lists -------------------------------------------------
# Deleted first, not overwritten: a training grid left behind by an earlier run of a different
# regime would retarget `dist` and `gtd`, and it would do it silently. `prepare.py` writes a fresh
# one below, so the only way this file survives the delete is a prepare that did not run.
rm -f "$RUN/input/chrominfo.train.txt"
PYTHONPATH=$REPO/src $PY "$HERE/prepare.py" --store "$STORE" \
    --regime "$REGIME" --out "$RUN/input" --chroms "$CHROMS" \
    --pilot 20 --no-signal

NTRACK=$(wc -l < "$RUN/input/inputinfofile.txt")
# 12 and no higher. `prepare.py` reaches the store through `candi.store.reader`, which imports
# `candi`, which imports torch; forty-eight processes loading torch off /project at once get a
# half-read module back as "cannot import name 'graph' ... circular import", and retries do not
# clear it. Four at a time is clean.
NSHARD=${CI_NSHARD:-12}
seq 0 $((NSHARD - 1)) | sed "s|\$|/$NSHARD|" > "$RUN/lists/prepare.txt"
cut -f2 "$RUN/input/inputinfofile.txt" | sort -u            > "$RUN/lists/convert.txt"
# ComputeGlobalDist runs for EVERY mark in the compendium, not just the target marks.
# GenerateTrainData's loadDistInfo opens DISTANCEDIR/<sample>_<mark>.txt for every (sample, mark)
# pair in inputinfofile before it looks at the target, so one missing mark fails every target.
cp "$RUN/lists/convert.txt"                                   "$RUN/lists/dist.txt"
# GenerateTrainData parallelizes over chromosomes as well as marks — one mark over a wide grid is
# hours of scanning in a single task. The chromosomes it may be split over are the TRAINING grid's,
# never `chrominfo.txt`: `-c` picks a chromosome out of the chrominfo the command is handed, and
# the command is handed `chrominfo.train.txt`. Naming an eval chromosome here is what used to put
# the sample on the scored chromosomes.
#
# A D32 region grid is the exception and stays unsplit: the whole of it is 1,023,489 bins, a
# fortieth of one real chromosome, so one task per mark is minutes and 40 array tasks per mark buy
# nothing.
NTRAIN=$(wc -l < "$RUN/input/chrominfo.train.txt")
if $PY -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('regions') else 1)" \
       "$REGIME"; then
  REGIONS=1
else
  REGIONS=0
fi
if [ "$REGIONS" = 1 ] || [ "$NTRAIN" -le 1 ]; then
  cut -f3 "$RUN/input/$TARGETS" | sort -u                          > "$RUN/lists/gtd.txt"
else
  while read -r M; do
    while read -r C; do printf '%s\t%s\n' "$M" "$C"
    done < <(cut -f1 "$RUN/input/chrominfo.train.txt")
  done < <(cut -f3 "$RUN/input/$TARGETS" | sort -u)                > "$RUN/lists/gtd.txt"
fi
echo "training grid: $NTRAIN declared chromosome(s)/region(s), \
$(awk -F'\t' '{s += $2 / 25} END {printf "%d", s}' "$RUN/input/chrominfo.train.txt") bins \
(regions=$REGIONS) | apply grid: $CHROMS"
cut -f1,3 "$RUN/input/$TARGETS"                               > "$RUN/lists/train.txt"
cp "$RUN/lists/train.txt"                                     "$RUN/lists/apply.txt"

echo "compendium: $NTRACK tracks | convert marks: $(wc -l < "$RUN/lists/convert.txt") | \
targets: $(wc -l < "$RUN/lists/train.txt")"

# --- the chain ----------------------------------------------------------------------------------
ENV_COMMON="ALL,CI_RUN=$RUN,CI_REPO=$REPO,CI_STORE=$STORE,CI_PY=$PY,CI_JAR=$JAR,CI_CHROMS=$CHROMS,CI_REGIME=$REGIME"

# `CI_THROTTLE` caps how many array tasks of one stage run at once. GenerateTrainData and Apply
# each hold ONE OPEN gzip reader per compendium track — 267 of them — so an unthrottled array puts
# thousands of concurrent handles on Lustre and draws transient
# "Cannot send after transport endpoint shutdown" opens that look like missing files.
THROTTLE=${CI_THROTTLE:-10}

sub() {  # sub <stage> <time> <mem> <mx> [afterok-jobid]
  local stage=$1 time=$2 mem=$3 mx=$4 dep=${5:-}
  local n; n=$(wc -l < "$RUN/lists/$stage.txt")
  local cap=""; case $stage in gtd|apply|train) cap="%$THROTTLE" ;; esac
  local depflag=(); [ -n "$dep" ] && depflag=(--dependency="afterok:$dep")
  sbatch --parsable --account="$ACCT" --nodes=1 --ntasks=1 --cpus-per-task=1 \
         --job-name="ci_$stage" --array="0-$((n - 1))$cap" --time="$time" --mem="$mem" \
         --output="$RUN/logs/%x_%A_%a.out" --error="$RUN/logs/%x_%A_%a.err" \
         "${depflag[@]}" --export="$ENV_COMMON,CI_STAGE=$stage,CI_MX=$mx" "$STAGE"
}

# `FROM` resumes the chain at a stage whose inputs already exist on disk — a stage that failed on
# its own list rather than on its data does not make the stages before it wrong.
STAGES="prepare convert dist gtd train apply"
case " $STAGES " in *" $FROM "*) ;; *) echo "unknown start stage $FROM"; exit 2;; esac
SKIP=1; DEP=""
for s in $STAGES; do
  [ "$s" = "$FROM" ] && SKIP=0
  [ "$SKIP" = 1 ] && continue
  case $s in
    prepare) t=3:00:00; m=8000M;  x=4000M ;;
    convert) t=6:00:00; m=8000M;  x=6000M ;;
    dist)    t=6:00:00; m=16000M; x=12000M ;;
    gtd)     t=12:00:00; m=24000M; x=20000M ;;
    train)   t=6:00:00; m=24000M; x=20000M ;;
    apply)   t=12:00:00; m=16000M; x=12000M ;;
  esac
  DEP=$(sub "$s" "$t" "$m" "$x" "$DEP")
  printf '%-8s %s\n' "$s" "$DEP"
done

cat > "$RUN/collect.sh" <<EOF
#!/bin/bash
# after ci_apply: Apply's wigs -> the RIVALS_PLAN.md §4.1 prediction root
set -euo pipefail
PYTHONPATH=$REPO/src $PY $HERE/collect.py --store $STORE \\
    --targets $RUN/input/$TARGETS --impute-dir $RUN/OUTPUTIMPUTEDIR \\
    --pred-root $RUN/pred --chroms \$(cut -f1 $RUN/input/chrominfo.txt | paste -sd, -) \\
    --jar $JAR --notes "$TARGETS on $CHROMS, $(basename $REGIME), ChromImpute paper defaults"
EOF
chmod +x "$RUN/collect.sh"
echo "collect with: $RUN/collect.sh"
