#!/bin/bash
# Lay out a run directory, build the work-item lists, and submit the six stages as one dependency
# chain of SLURM arrays. Drives all three runs — the §7.2 pilot, the full P1 grid, and P2.
#
#   bash submit.sh [RUNDIR] [FROM_STAGE] [TARGETS]
#
#   pilot   bash submit.sh $S/pilot_chr21                                  # 20 targets, chr21
#   P1      bash submit.sh $S/p1_chr21    prepare targets.tsv              # 45 targets, chr21
#   P2      CI_CHROMS=all bash submit.sh $S/p2_genome prepare targets.tsv  # 45 targets, 23 chroms
#
# Nothing here decides science. The compendium is every training track the regime declares (§6.2,
# enforced in prepare.training_tracks); the pilot targets come from prepare.pilot_subset; every
# ChromImpute flag is a paper default except the ones that only choose what to parallelize over.
set -euo pipefail

RUN=${1:-$HOME/scratch/t51_chromimpute/pilot_chr21}
FROM=${2:-prepare}
TARGETS=${3:-targets_pilot.tsv}
CHROMS=${CI_CHROMS:-chr21}
REPO=${CI_REPO:-/project/def-maxwl/mforooz/CANDII_t51}
STORE=${CI_STORE:-/project/def-maxwl/mforooz/CANDI_STORE/eic}
PY=${CI_PY:-/project/def-maxwl/mforooz/candi_venv/bin/python}
JAR=${CI_JAR:-$HOME/scratch/t51_chromimpute/tool/ChromImpute.jar}
ACCT=${CI_ACCT:-def-maxwl}
HERE=$REPO/competitors/chromimpute
STAGE=$HERE/slurm/stage.sh

mkdir -p "$RUN"/{lists,logs,timing}

# --- the three text files, and the target lists -------------------------------------------------
PYTHONPATH=$REPO/src $PY "$HERE/prepare.py" --store "$STORE" \
    --regime "$REPO/configs/regime.eic_val.json" --out "$RUN/input" --chroms "$CHROMS" \
    --pilot 20 --no-signal

NTRACK=$(wc -l < "$RUN/input/inputinfofile.txt")
NSHARD=${CI_NSHARD:-16}
seq 0 $((NSHARD - 1)) | sed "s|\$|/$NSHARD|" > "$RUN/lists/prepare.txt"
cut -f2 "$RUN/input/inputinfofile.txt" | sort -u            > "$RUN/lists/convert.txt"
# ComputeGlobalDist runs for EVERY mark in the compendium, not just the target marks.
# GenerateTrainData's loadDistInfo opens DISTANCEDIR/<sample>_<mark>.txt for every (sample, mark)
# pair in inputinfofile before it looks at the target, so one missing mark fails every target.
cp "$RUN/lists/convert.txt"                                   "$RUN/lists/dist.txt"
cut -f3 "$RUN/input/$TARGETS" | sort -u                       > "$RUN/lists/gtd.txt"
cut -f1,3 "$RUN/input/$TARGETS"                               > "$RUN/lists/train.txt"
cp "$RUN/lists/train.txt"                                     "$RUN/lists/apply.txt"

echo "compendium: $NTRACK tracks | convert marks: $(wc -l < "$RUN/lists/convert.txt") | \
targets: $(wc -l < "$RUN/lists/train.txt")"

# --- the chain ----------------------------------------------------------------------------------
ENV_COMMON="ALL,CI_RUN=$RUN,CI_REPO=$REPO,CI_STORE=$STORE,CI_PY=$PY,CI_JAR=$JAR,CI_CHROMS=$CHROMS"

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
    prepare) t=1:00:00; m=8000M;  x=4000M ;;
    convert) t=2:00:00; m=8000M;  x=6000M ;;
    dist)    t=2:00:00; m=16000M; x=12000M ;;
    gtd)     t=6:00:00; m=24000M; x=20000M ;;
    train)   t=6:00:00; m=24000M; x=20000M ;;
    apply)   t=3:00:00; m=12000M; x=8000M ;;
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
    --jar $JAR --notes "t51 $TARGETS on $CHROMS, regime.eic_val, ChromImpute paper defaults"
EOF
chmod +x "$RUN/collect.sh"
echo "collect with: $RUN/collect.sh"
