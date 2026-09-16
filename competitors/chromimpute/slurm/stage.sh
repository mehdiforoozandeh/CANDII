#!/bin/bash
# One SLURM array task = one work item of one ChromImpute stage.
#
#   sbatch --array=0-N --job-name=ci_convert --time=... --mem=... \
#          --export=ALL,CI_STAGE=convert,CI_RUN=$RUN stage.sh
#
# The stage names are the manual's command names, lowercased. `CI_RUN` is a run directory laid out
# by `submit.sh`; every path below is derived from it, so nothing but the stage and the array
# index changes between submissions. The work-item list for a stage is `$CI_RUN/lists/<stage>.txt`,
# one item per line, read by `$SLURM_ARRAY_TASK_ID` — except the two sigma verbs, `strain` and
# `sapply`, which share `$CI_RUN/lists/sigma_items.txt` because one writes what the other reads.
#
# Timing goes to `$CI_RUN/timing/<stage>.<taskid>.tsv` — stage, item, seconds, MaxRSS is read back
# out of `sacct` afterwards. That file is the pilot memo's raw material.
set -euo pipefail

: "${CI_STAGE:?set CI_STAGE}"
: "${CI_RUN:?set CI_RUN}"
: "${CI_STORE:=/project/def-maxwl/mforooz/CANDI_STORE/eic}"
: "${CI_REPO:=/project/def-maxwl/mforooz/CANDII_main}"
: "${CI_PY:=/project/def-maxwl/mforooz/EpiDenoise/candi_venv/bin/python}"
# RETARGETED 2026-08-31: was a baked configs/regime.eic_val.json (train chr19, eval chr21).
# The live regimes are eic_19 and eic_pilot (BENCHMARK_DESIGN.md §3) and submit.sh passes
# this through, so the prepare stage and the scorer cannot disagree about which one ran.
: "${CI_REGIME:=$CI_REPO/configs/regime.eic_19.json}"
: "${CI_JAR:=/project/def-maxwl/mforooz/tools/ChromImpute.jar}"
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
# The sigma stage's Apply output, kept apart from the board one on purpose: these are TRAINING
# self-pairs on the TRAINING grid, and a wig of theirs landing in OUTPUTIMPUTEDIR would be
# collected into a board prediction root as though it were an eval target.
SIMP=$CI_RUN/SIGMAIMPUTEDIR
# The sigma stage's own predictors. Disjoint from PREDICTORDIR by construction — a board predictor
# is (T_x, mark) for a mark T_x does NOT carry, a sigma predictor is (T_x, mark) for one it does —
# but kept apart anyway, so the checkpoint copied to /project is the fitted parameters the board
# rows were made with and nothing else.
SPRED=$CI_RUN/SIGMAPREDICTORDIR
mkdir -p "$CONV" "$DIST" "$TRAINDATA" "$PRED" "$IMP" "$SIMP" "$SPRED" "$CI_RUN/timing"

# `CI_CHROMS=all` is `prepare.py`'s spelling, not a chromosome. Everything downstream loops over the
# real names, and `chrominfo.txt` is the list `prepare.py` just wrote.
if [ "$CI_CHROMS" = "all" ]; then
  CHROM_LIST=$(cut -f1 "$IN/chrominfo.txt" | tr '\n' ' ')
else
  CHROM_LIST=${CI_CHROMS//,/ }
fi

# The two grids. `prepare.py` ALWAYS writes `chrominfo.train.txt` — the regime's `train_chroms`,
# or one declared chromosome per Pilot Region under a `regions` regime (D32) — and THAT is the grid
# the transferable parameters are fit on: `GenerateTrainData` spreads its 100,000 locations over
# everything the chrominfo it is handed declares, and `Train` turns them into predictors `Apply`
# reuses everywhere. `Apply` keeps `chrominfo.txt`, the real chromosomes being predicted.
#
# There is no fallback to `chrominfo.txt` and there must not be one: that fallback is exactly how
# the sampler used to land on the eval chromosomes, and it did it silently. A missing training grid
# is a failed `prepare`, not a default. `prepare` itself is exempt — it is the stage that writes it.
if [ ! -s "$IN/chrominfo.train.txt" ]; then
  if [ "$CI_STAGE" != "prepare" ]; then
    echo "[$CI_STAGE] REFUSING: $IN/chrominfo.train.txt does not exist. Re-run the prepare stage;" >&2
    echo "      without that grid there is nothing to fit predictors on but the chromosomes being" >&2
    echo "      predicted, which BENCHMARK_DESIGN.md Rule 2 (§2) forbids for transferable" >&2
    echo "      parameters. Falling back to chrominfo.txt is the bug, not the recovery." >&2
    exit 2
  fi
  TRAIN_INFO=$IN/chrominfo.txt
  TRAIN_LIST=""
else
  TRAIN_INFO=$IN/chrominfo.train.txt
  TRAIN_LIST=$(cut -f1 "$TRAIN_INFO" | tr '\n' ' ')
fi
# Deduplicated: a regime whose training grid names a chromosome that is also predicted needs its
# bedgraph converted once, and the delete pass below must not count it twice.
GRID_LIST=$(printf '%s\n' $CHROM_LIST $TRAIN_LIST | awk '!seen[$0]++' | tr '\n' ' ')

# ONE LIST FOR BOTH SIGMA VERBS. `strain` trains a classifier set and `sapply` loads it back, so an
# item one verb has and the other does not is not a missing row — it is a `sapply` task asking the
# jar for a set nobody trained, and the jar's answer is the MARK-level
# "No previously trained classifiers for mark <mark> were found available to load!", which names
# neither the cell nor the file it looked for. Two per-stage files can drift apart; one cannot.
# `sigma.sh` writes `lists/sigma_items.txt` and sizes both arrays off it.
case "$CI_STAGE" in
  strain|sapply) LIST=$CI_RUN/lists/sigma_items.txt ;;
  *)             LIST=$CI_RUN/lists/$CI_STAGE.txt ;;
esac
ITEM=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
[ -n "$ITEM" ] || { echo "no item $SLURM_ARRAY_TASK_ID in $LIST"; exit 1; }
echo "[$CI_STAGE] task $SLURM_ARRAY_TASK_ID item: $ITEM"

# THE CLASSIFIER SET IS NAMED ONCE, HERE, for both sigma verbs — the writer and the reader cannot
# spell it differently if only one line spells it. `Train` writes
# `classifier_<sample>_<mark>_<i>_<j>.txt.gz` into the predictor directory, one per training sample
# it could use; `Apply` loads every file under that prefix.
if [ "$CI_STAGE" = strain ] || [ "$CI_STAGE" = sapply ]; then
  S=$(echo "$ITEM" | cut -f1); M=$(echo "$ITEM" | cut -f2)
  CLASSIFIERS="$SPRED/classifier_${S}_${M}_"
fi

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
      # Skipped when the apply grid already declared it: same name, same declared length, so the
      # second Convert would rewrite the identical file.
      case " $CHROM_LIST " in *" $C "*) continue ;; esac
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
    # training loci with the predictors — never on the chromosomes being scored.
    CI ComputeGlobalDist -m "$ITEM" "$CONV" "$IN/inputinfofile.txt" "$TRAIN_INFO" "$DIST"
    ;;
  gtd)       # item: <mark>, or <mark> <TAB> <chrom> — the chrom is off the TRAINING grid
    # The manual's own recommended parallelization for this command is over chromosomes, and it is
    # needed on a wide training grid: one mark over all 23 chromosomes is ~7.6 h of scanning, past
    # any walltime bin we want. `-c` divides the sampled locations between the declared chromosomes
    # and prefixes the traindata file, and `Train` takes the union of the prefixed files when no
    # unprefixed one exists — so the total is 100,000 either way, just spread over more files.
    # Measured, not assumed: on a two-chromosome grid at `-f 400`, unsplit gave 400 instances and
    # the two `-c` tasks gave 193 + 207. `submit.sh` splits only when the training grid is more
    # than one chromosome and is not a D32 region grid, where the whole scope is minutes anyway.
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
  strain)    # item: <training sample> <TAB> <mark> — a predictor for a mark the sample HAS
    # THE BOARD'S PREDICTORS ARE NO USE TO A sigma FIT, and this is why the sigma stage has a
    # Train step of its own. `lists/train.txt` is `cut -f1,3 targets.tsv`, so its every (T_x, mark)
    # is a mark T_x does NOT carry — there is no truth in the store to take a residual against.
    # §7 wants residuals on TRAINING tracks, so the sigma stage trains and applies (T_x, mark)
    # for marks T_x DOES carry. ChromImpute leaves the target sample's own track out of its fit,
    # which is what makes the residual a real one and not a copy.
    CI Train "$TRAINDATA" "$IN/inputinfofile.txt" "$SPRED" "$S" "$M"
    # `Train` EXITS 0 WHEN IT TRAINS NOTHING. Holding the target's own track out leaves a one-track
    # cell with no features, and the jar answers that by writing the `useattributes_` masks, no
    # classifier, and status 0 — so the array task reports COMPLETED and `sapply` inherits the
    # failure. Measured on Fir: strain 57807329 was 22/22 COMPLETED, and two of its items
    # (T_SK-MEL-5 DNase-seq, T_skin_of_body DNase-seq) had left SIGMAPREDICTORDIR with 33
    # `useattributes_` files and zero `classifier_` files each. `sigma.sh` now drops a one-track
    # cell before it gets here; this is the check that says so if it ever gets here anyway.
    # Outside RETRY on purpose: nothing about this repeats differently.
    NCLS=$(ls -1 "$CLASSIFIERS"* 2>/dev/null | wc -l | tr -d "[:space:]") || NCLS=0
    [ "$NCLS" -gt 0 ] || {
      echo "[strain] REFUSING: Train exited 0 but wrote no $CLASSIFIERS* — it trained nothing for" >&2
      echo "      $S / $M. The usual cause is that $S carries a single track in" >&2
      echo "      $IN/inputinfofile.txt: Train holds the target's own track out, so there is" >&2
      echo "      nothing left to build features from. Drop the cell in sigma.sh rather than" >&2
      echo "      letting sapply discover it." >&2
      exit 3
    }
    echo "[strain] $S $M: $NCLS classifier file(s)"
    ;;
  sapply)    # item: <training sample> <TAB> <mark> — the sigma stage's training self-pairs
    # Same distance tables, the sigma stage's own predictors, THE OTHER GRID. §7 fits sigma on
    # training residuals only, so this Apply runs over `chrominfo.train.txt` and writes to
    # SIGMAIMPUTEDIR — never into OUTPUTIMPUTEDIR, where `collect.py` would pick it up as a board
    # target.
    #
    # THE CLASSIFIER SET IS CHECKED BEFORE THE JAR RUNS, and outside `RETRY`. An absent set is a
    # deterministic error — `strain` never wrote it — but the jar spells it
    # "No previously trained classifiers for mark <mark> were found available to load!", which
    # names only the mark, and `RETRY` repeated that three times with a 30 s and a 60 s sleep in
    # between before the array task failed and `afterok` cancelled the fit (Fir, sapply 57807330
    # task 5, 2026-09-02). Fail in one second, and say which file was looked for.
    NCLS=$(ls -1 "$CLASSIFIERS"* 2>/dev/null | wc -l | tr -d "[:space:]") || NCLS=0
    [ "$NCLS" -gt 0 ] || {
      echo "[sapply] REFUSING: no $CLASSIFIERS* in $SPRED, so there is nothing for Apply to load" >&2
      echo "      for $S / $M. This item is in $LIST, which strain read too, so strain either did" >&2
      echo "      not run or trained nothing for it — read logs/ci_strain_*.out for this index." >&2
      echo "      Not retried: a missing classifier set does not become present on a second try." >&2
      exit 3
    }
    for C in $TRAIN_LIST; do
      CI Apply -c "$C" -o "impute.$S.$M.wig" "$CONV" "$DIST" "$SPRED" "$IN/inputinfofile.txt" \
         "$TRAIN_INFO" "$SIMP" "$S" "$M"
    done
    ;;
  *) echo "unknown CI_STAGE=$CI_STAGE"; exit 2 ;;
esac

printf '%s\t%s\t%s\t%s\n' "$CI_STAGE" "$ITEM" "$((SECONDS - START))" "$SLURM_JOB_ID" \
    > "$CI_RUN/timing/$CI_STAGE.$SLURM_ARRAY_TASK_ID.tsv"
echo "[$CI_STAGE] done in $((SECONDS - START))s"
