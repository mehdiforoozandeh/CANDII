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
: "${CI_REPO:=/project/def-maxwl/mforooz/CANDII_t78_code}"
: "${CI_PY:=/project/def-maxwl/mforooz/EpiDenoise/candi_venv/bin/python}"
# RETARGETED 2026-08-31: was a baked configs/regime.eic_val.json (train chr19, eval chr21).
# The live regimes are eic_19 and eic_pilot (BENCHMARK_DESIGN.md §3) and submit.sh passes
# this through, so the prepare stage and the scorer cannot disagree about which one ran.
: "${CI_REGIME:=$CI_REPO/configs/regime.eic_19.json}"
: "${CI_JAR:=$HOME/scratch/t51_chromimpute/tool/ChromImpute.jar}"
# The regime's eval_chroms under §4. chr21 alone was the old eval scope.
: "${CI_CHROMS:=chr20,chr21,chr22}"
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

# D32. Under a `regions` regime `prepare.py` also writes `chrominfo.train.txt`, one declared
# chromosome per Pilot Region, and THAT is the grid the transferable parameters are fit on:
# `GenerateTrainData` spreads its 100,000 locations over everything the chrominfo it is handed
# declares, so handing it the eval chromosomes as well would put most of the sample outside the
# BED. `Apply` keeps the real chromosomes. Without a `regions` regime the file is absent, both
# grids are `chrominfo.txt`, and every command below is what it was.
if [ -s "$IN/chrominfo.train.txt" ]; then
  TRAIN_INFO=$IN/chrominfo.train.txt
  TRAIN_LIST=$(cut -f1 "$TRAIN_INFO" | tr '\n' ' ')
else
  TRAIN_INFO=$IN/chrominfo.txt
  TRAIN_LIST=""
  # The fallback is silent and it is the one that costs a whole run: a `regions` regime whose
  # training grid was never written would fit its predictors on chr20+21+22 and look fine. Refuse.
  # `prepare` is exempt — it is the stage that writes the file.
  if [ "$CI_STAGE" != "prepare" ] \
     && $CI_PY -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('regions') else 1)" \
        "$CI_REGIME"; then
    echo "[$CI_STAGE] REFUSING: $CI_REGIME declares a \`regions\` BED but $IN/chrominfo.train.txt" >&2
    echo "      does not exist. Re-run the prepare stage; without that grid GenerateTrainData" >&2
    echo "      would sample outside the Pilot Regions (D32, BENCHMARK_DESIGN.md §3.1)." >&2
    exit 2
  fi
fi
GRID_LIST="$CHROM_LIST $TRAIN_LIST"

LIST=$CI_RUN/lists/$CI_STAGE.txt
ITEM=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
[ -n "$ITEM" ] || { echo "no item $SLURM_ARRAY_TASK_ID in $LIST"; exit 1; }
echo "[$CI_STAGE] task $SLURM_ARRAY_TASK_ID item: $ITEM"

# Retry, because `GenerateTrainData` and `Apply` open one gzip reader per compendium track — 267 at
# once — and Lustre answers a few of those with a transient
# "FileNotFoundException ... (Cannot send after transport endpoint shutdown)" that looks exactly
# like a missing file. A command that is genuinely wrong fails in seconds, so the retries cost
# nothing; a command that hit the filesystem gets a second chance instead of failing a whole array.
RETRY() {
  local try
  for try in 1 2 3; do
    "$@" && return 0
    echo "[$CI_STAGE] attempt $try failed; retrying in $((try * 30))s"
    sleep $((try * 30))
  done
  return 1
}
CI() { RETRY java -mx"$CI_MX" -jar "$CI_JAR" "$@"; }
START=$SECONDS

case "$CI_STAGE" in
  prepare)
    # Retried for the same reason the java stages are, but a different filesystem: `import candi`
    # pulls in torch, and forty-eight processes importing it off /project at once occasionally get
    # a half-read module back as "cannot import name 'graph' ... circular import".
    export PYTHONPATH=$CI_REPO/src
    RETRY $CI_PY "$CI_REPO/competitors/chromimpute/prepare.py" \
        --store "$CI_STORE" --regime "$CI_REGIME" \
        --out "$IN" --chroms "$CI_CHROMS" --shard "$ITEM"
    ;;
  convert)   # item: a mark. Reads only inputinfofile.txt, so only training cells are converted.
    for C in $CHROM_LIST; do
      CI Convert -c "$C" -m "$ITEM" "$IN/signal" "$IN/inputinfofile.txt" "$IN/chrominfo.txt" "$CONV"
    done
    for C in $TRAIN_LIST; do
      CI Convert -c "$C" -m "$ITEM" "$IN/signal" "$IN/inputinfofile.txt" "$TRAIN_INFO" "$CONV"
    done
    # The bedgraphs are 161 GB and ~6 000 inodes of pure intermediate genome-wide, on a scratch
    # quota shared with other tasks, so they go as soon as `Convert` has consumed them. Verify
    # FIRST: `Convert` warns and skips a missing input rather than failing, so a resumed run that
    # deleted too early would leave a hole instead of an error.
    MISSING=0
    while IFS=$'\t' read -r _S _M FNAME _REST; do
      [ "$_M" = "$ITEM" ] || continue
      for C in $GRID_LIST; do
        [ -s "$CONV/${C}_${FNAME}.wig.gz" ] || { echo "MISSING $CONV/${C}_${FNAME}.wig.gz"; MISSING=1; }
      done
    done < "$IN/inputinfofile.txt"
    [ "$MISSING" = 0 ] || { echo "[convert] converted output incomplete for $ITEM — keeping bedgraphs"; exit 1; }
    if [ "${CI_KEEP_BEDGRAPH:-0}" != "1" ]; then
      NDEL=0
      while IFS=$'\t' read -r _S _M FNAME _REST; do
        [ "$_M" = "$ITEM" ] || continue
        for C in $GRID_LIST; do rm -f "$IN/signal/${C}_${FNAME}" && NDEL=$((NDEL + 1)); done
      done < "$IN/inputinfofile.txt"
      echo "[convert] $ITEM: converted output verified, $NDEL bedgraph file(s) deleted"
    fi
    ;;
  dist)      # item: a mark
    # On the TRAINING grid. The correlation table it writes is one sample ranking reused at every
    # position `Apply` predicts, so it is a transferable parameter and Rule 2 puts it on the
    # training loci with the predictors. Without a `regions` regime this is `chrominfo.txt` and
    # the command is unchanged.
    CI ComputeGlobalDist -m "$ITEM" "$CONV" "$IN/inputinfofile.txt" "$TRAIN_INFO" "$DIST"
    ;;
  gtd)       # item: <mark>, or <mark> <TAB> <chrom>
    # The manual's own recommended parallelization for this command is over chromosomes, and it has
    # to be used genome-wide: one mark over all 23 chromosomes is ~7.6 h of scanning, past any
    # walltime bin we want. `-c` subsets the sampled locations to that chromosome and prefixes the
    # traindata file, and `Train` takes the union of the prefixed files when no unprefixed one
    # exists — so the 100 000 instances are the same either way, just spread over more files.
    # Under a `regions` regime `submit.sh` emits a bare mark instead: the whole D32 training grid
    # is 1,023,489 bins, a fortieth of one real chromosome, so one unsplit task per mark is minutes
    # — and one task is the only way the 100,000 locations are 100,000 rather than 100,000 per
    # region, which the manual does not settle for `-c`.
    M=$(echo "$ITEM" | cut -f1); C=$(echo "$ITEM" | cut -f2)
    if [ "$C" = "$M" ]; then
      CI GenerateTrainData "$CONV" "$DIST" "$IN/inputinfofile.txt" "$TRAIN_INFO" \
         "$TRAINDATA" "$M"
    else
      CI GenerateTrainData -c "$C" "$CONV" "$DIST" "$IN/inputinfofile.txt" \
         "$TRAIN_INFO" "$TRAINDATA" "$M"
    fi
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
