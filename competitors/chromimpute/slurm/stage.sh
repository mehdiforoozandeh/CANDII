#!/bin/bash
# One SLURM array task = one work item of one ChromImpute stage.
#
#   sbatch --array=0-N --job-name=ci_convert --time=... --mem=... \
#          --export=ALL,CI_STAGE=convert,CI_RUN=$RUN stage.sh
#
# The stage names are the manual's command names, lowercased. `CI_RUN` is a run directory laid out
# by `submit.sh`; every path below is derived from it, so nothing but the stage and the array
# index changes between submissions. The work-item list for a stage is `$CI_RUN/lists/<stage>.txt`,
# one item per line, read by `$SLURM_ARRAY_TASK_ID`.
#
# Timing goes to `$CI_RUN/timing/<stage>.<taskid>.tsv` — stage, item, seconds, MaxRSS is read back
# out of `sacct` afterwards. That file is the pilot memo's raw material.
set -euo pipefail

: "${CI_STAGE:?set CI_STAGE}"
: "${CI_RUN:?set CI_RUN}"
: "${CI_STORE:=/project/def-maxwl/mforooz/CANDI_STORE/eic}"
: "${CI_REPO:=/project/def-maxwl/mforooz/CANDII_t51}"
: "${CI_PY:=/project/def-maxwl/mforooz/candi_venv/bin/python}"
: "${CI_JAR:=$HOME/scratch/t51_chromimpute/tool/ChromImpute.jar}"
: "${CI_CHROMS:=chr21}"
: "${CI_MX:=8000M}"

module load java/21.0.1
unset JAVA_TOOL_OPTIONS          # the module sets -Xmx2g, which would override -mx below

IN=$CI_RUN/input
CONV=$CI_RUN/CONVERTEDDIR
DIST=$CI_RUN/DISTANCEDIR
TRAINDATA=$CI_RUN/TRAINDATADIR
PRED=$CI_RUN/PREDICTORDIR
IMP=$CI_RUN/OUTPUTIMPUTEDIR
mkdir -p "$CONV" "$DIST" "$TRAINDATA" "$PRED" "$IMP" "$CI_RUN/timing"

# `CI_CHROMS=all` is `prepare.py`'s spelling, not a chromosome. Everything downstream loops over the
# real names, and `chrominfo.txt` is the list `prepare.py` just wrote.
if [ "$CI_CHROMS" = "all" ]; then
  CHROM_LIST=$(cut -f1 "$IN/chrominfo.txt" | tr '\n' ' ')
else
  CHROM_LIST=${CI_CHROMS//,/ }
fi

LIST=$CI_RUN/lists/$CI_STAGE.txt
ITEM=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
[ -n "$ITEM" ] || { echo "no item $SLURM_ARRAY_TASK_ID in $LIST"; exit 1; }
echo "[$CI_STAGE] task $SLURM_ARRAY_TASK_ID item: $ITEM"

# Retry, because `GenerateTrainData` and `Apply` open one gzip reader per compendium track — 267 at
# once — and Lustre answers a few of those with a transient
# "FileNotFoundException ... (Cannot send after transport endpoint shutdown)" that looks exactly
# like a missing file. A command that is genuinely wrong fails in seconds, so the retries cost
# nothing; a command that hit the filesystem gets a second chance instead of failing a whole array.
CI() {
  local try
  for try in 1 2 3; do
    java -mx"$CI_MX" -jar "$CI_JAR" "$@" && return 0
    echo "[$CI_STAGE] attempt $try failed; retrying in $((try * 30))s"
    sleep $((try * 30))
  done
  return 1
}
START=$SECONDS

case "$CI_STAGE" in
  prepare)
    PYTHONPATH=$CI_REPO/src $CI_PY "$CI_REPO/competitors/chromimpute/prepare.py" \
        --store "$CI_STORE" --regime "$CI_REPO/configs/regime.eic_val.json" \
        --out "$IN" --chroms "$CI_CHROMS" --shard "$ITEM"
    ;;
  convert)   # item: a mark. Reads only inputinfofile.txt, so only training cells are converted.
    for C in $CHROM_LIST; do
      CI Convert -c "$C" -m "$ITEM" "$IN/signal" "$IN/inputinfofile.txt" "$IN/chrominfo.txt" "$CONV"
    done
    ;;
  dist)      # item: a mark
    CI ComputeGlobalDist -m "$ITEM" "$CONV" "$IN/inputinfofile.txt" "$IN/chrominfo.txt" "$DIST"
    ;;
  gtd)       # item: a target mark
    # No -c: the paper default draws its 100 000 training locations across everything chrominfo
    # declares, and one traindata file per mark is what `Train` looks for first. Passing -c would
    # give 23 chrom-prefixed files per mark for the same 100 000 instances.
    CI GenerateTrainData "$CONV" "$DIST" "$IN/inputinfofile.txt" "$IN/chrominfo.txt" \
       "$TRAINDATA" "$ITEM"
    ;;
  train)     # item: <sample> <TAB> <mark>
    S=$(echo "$ITEM" | cut -f1); M=$(echo "$ITEM" | cut -f2)
    CI Train "$TRAINDATA" "$IN/inputinfofile.txt" "$PRED" "$S" "$M"
    ;;
  apply)     # item: <sample> <TAB> <mark>
    S=$(echo "$ITEM" | cut -f1); M=$(echo "$ITEM" | cut -f2)
    for C in $CHROM_LIST; do
      CI Apply -c "$C" -o "impute.$S.$M.wig" "$CONV" "$DIST" "$PRED" "$IN/inputinfofile.txt" \
         "$IN/chrominfo.txt" "$IMP" "$S" "$M"
    done
    ;;
  *) echo "unknown CI_STAGE=$CI_STAGE"; exit 2 ;;
esac

printf '%s\t%s\t%s\t%s\n' "$CI_STAGE" "$ITEM" "$((SECONDS - START))" "$SLURM_JOB_ID" \
    > "$CI_RUN/timing/$CI_STAGE.$SLURM_ARRAY_TASK_ID.tsv"
echo "[$CI_STAGE] done in $((SECONDS - START))s"
