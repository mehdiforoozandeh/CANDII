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
# TWO THINGS THIS SCRIPT DOES NOT DO, AND THEY ARE BLOCKERS, NOT OVERSIGHTS:
#
#  1. UNDER A REGIME WITH NO `regions` — eic_19 — `gtd` (GenerateTrainData) SAMPLES ITS 100,000
#     TRAINING LOCATIONS INSIDE $CI_CHROMS, i.e. inside the eval chromosomes. The predictors
#     trained on them are TRANSFERABLE parameters, so under Rule 2 (§2) they belong on the
#     regime's train_chroms instead. Nothing here reads the scored track, so Rule 1 is intact; it
#     is the scope NAME that would be wrong. Moving it is a PI decision, not ours, and it is left
#     alone deliberately. Raise it before launching eic_19.
#     A `regions` regime is a different case and IS handled: `prepare.py` writes a separate D32
#     training grid and `stage.sh` points `dist` and `gtd` at it, so under eic_pilot the
#     transferable parameters are fit inside the Pilot Regions and nowhere near chr20/21/22.
#  2. `targets.tsv` is EVERY declared eval pair, and the live regimes declare 38 — 26 V_ AND 12 B_.
#     §5 rules B_ is touched ONCE, at the very end. Filter the target list to V_ before running
#     anything but the final B_ pass.
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

# --- the three text files, and the target lists -------------------------------------------------
# Deleted first, not overwritten: `prepare.py` only writes it for a `regions` regime, and a file
# left behind by an earlier run of a different regime would silently retarget `dist` and `gtd`.
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
# GenerateTrainData parallelizes over chromosomes as well as marks whenever there is more than one
# chromosome — one mark genome-wide is hours of scanning in a single task.
#
# A D32 training grid is the exception: one mark over the whole of it is minutes, and splitting it
# would raise a question the manual does not answer — whether `-c` divides the 100,000 locations or
# repeats them per declared chromosome. One task per mark makes the count exactly 100,000 whichever
# it is, which is the number §11 prices the method at.
if [ -s "$RUN/input/chrominfo.train.txt" ]; then
  cut -f3 "$RUN/input/$TARGETS" | sort -u                          > "$RUN/lists/gtd.txt"
  echo "D32 training grid: $(wc -l < "$RUN/input/chrominfo.train.txt") region(s), \
$(awk -F'\t' '{s += $2 / 25} END {printf "%d", s}' "$RUN/input/chrominfo.train.txt") bins"
elif [ "$(wc -l < "$RUN/input/chrominfo.txt")" -gt 1 ]; then
  while read -r M; do
    while read -r C; do printf '%s\t%s\n' "$M" "$C"; done < <(cut -f1 "$RUN/input/chrominfo.txt")
  done < <(cut -f3 "$RUN/input/$TARGETS" | sort -u)                > "$RUN/lists/gtd.txt"
else
  cut -f3 "$RUN/input/$TARGETS" | sort -u                          > "$RUN/lists/gtd.txt"
fi
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
