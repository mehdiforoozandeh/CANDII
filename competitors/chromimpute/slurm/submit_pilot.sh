#!/bin/bash
# The §7.2 pilot gate, chr21: lay out a run directory, build the work-item lists, and submit the
# six stages as one dependency chain of SLURM arrays. Prices the grid; does not commit to it.
#
#   bash submit_pilot.sh [RUNDIR]      # default ~/scratch/t51_chromimpute/pilot_chr21
#
# Nothing here decides science. The compendium is every training track the regime declares (§6.2,
# enforced in prepare.training_tracks); the 20 targets come from prepare.pilot_subset; every
# ChromImpute flag is a paper default except the ones that only choose what to parallelize over.
set -euo pipefail

RUN=${1:-$HOME/scratch/t51_chromimpute/pilot_chr21}
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
    --regime "$REPO/configs/regime.eic_val.json" --out "$RUN/input" --chroms chr21 \
    --pilot 20 --no-signal

NTRACK=$(wc -l < "$RUN/input/inputinfofile.txt")
NSHARD=${CI_NSHARD:-16}
seq 0 $((NSHARD - 1)) | sed "s|\$|/$NSHARD|" > "$RUN/lists/prepare.txt"
cut -f2 "$RUN/input/inputinfofile.txt" | sort -u            > "$RUN/lists/convert.txt"
cut -f3 "$RUN/input/targets_pilot.tsv" | sort -u            > "$RUN/lists/dist.txt"
cp "$RUN/lists/dist.txt"                                      "$RUN/lists/gtd.txt"
cut -f1,3 "$RUN/input/targets_pilot.tsv"                      > "$RUN/lists/train.txt"
cp "$RUN/lists/train.txt"                                     "$RUN/lists/apply.txt"

echo "compendium: $NTRACK tracks | convert marks: $(wc -l < "$RUN/lists/convert.txt") | \
targets: $(wc -l < "$RUN/lists/train.txt")"

# --- the chain ----------------------------------------------------------------------------------
ENV_COMMON="ALL,CI_RUN=$RUN,CI_REPO=$REPO,CI_STORE=$STORE,CI_PY=$PY,CI_JAR=$JAR,CI_CHROMS=chr21"

sub() {  # sub <stage> <time> <mem> <mx> [afterok-jobid]
  local stage=$1 time=$2 mem=$3 mx=$4 dep=${5:-}
  local n; n=$(wc -l < "$RUN/lists/$stage.txt")
  local depflag=(); [ -n "$dep" ] && depflag=(--dependency="afterok:$dep")
  sbatch --parsable --account="$ACCT" --nodes=1 --ntasks=1 --cpus-per-task=1 \
         --job-name="ci_$stage" --array=0-$((n - 1)) --time="$time" --mem="$mem" \
         --output="$RUN/logs/%x_%A_%a.out" --error="$RUN/logs/%x_%A_%a.err" \
         "${depflag[@]}" --export="$ENV_COMMON,CI_STAGE=$stage,CI_MX=$mx" "$STAGE"
}

J_PREP=$(sub prepare 1:00:00 8000M 4000M);            echo "prepare  $J_PREP"
J_CONV=$(sub convert 2:00:00 8000M 6000M "$J_PREP");  echo "convert  $J_CONV"
J_DIST=$(sub dist    2:00:00 16000M 12000M "$J_CONV"); echo "dist     $J_DIST"
J_GTD=$(sub  gtd     6:00:00 24000M 20000M "$J_DIST"); echo "gtd      $J_GTD"
J_TRN=$(sub  train   6:00:00 24000M 20000M "$J_GTD");  echo "train    $J_TRN"
J_APP=$(sub  apply   3:00:00 12000M 8000M "$J_TRN");   echo "apply    $J_APP"

cat > "$RUN/collect.sh" <<EOF
#!/bin/bash
# after ci_apply: Apply's wigs -> the RIVALS_PLAN.md §4.1 prediction root
set -euo pipefail
PYTHONPATH=$REPO/src $PY $HERE/collect.py --store $STORE \\
    --targets $RUN/input/targets_pilot.tsv --impute-dir $RUN/OUTPUTIMPUTEDIR \\
    --pred-root $RUN/pred --chroms chr21 --jar $JAR \\
    --notes "t51 chr21 pilot, 20 targets, regime.eic_val, ChromImpute paper defaults"
EOF
chmod +x "$RUN/collect.sh"
echo "collect with: $RUN/collect.sh"
