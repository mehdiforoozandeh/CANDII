#!/bin/bash
# Blind-set (B_) preview: ChromImpute gtd -> train -> apply -> collect, as one dependency chain.
# Run this ON THE FIR LOGIN NODE (it only submits; it computes nothing):
#
#   bash competitors/chromimpute/slurm/blind_preview_chain.sh
#
# THROWAWAY, PI-AUTHORISED (2026-09-01).
#
# BLOCKER #2 (BENCHMARK_DESIGN.md §13.3, repeated in this directory's `submit.sh`): the blind set
# is touched ONCE, at the very end. The PI has knowingly overridden that for a side-by-side
# preview. Nothing here is scored, nothing enters a leaderboard, and every array produced here is
# regenerated from retrained models later.
#
# BLOCKER #1 (train/eval locus leakage): the `full_genome` run this reuses has NO
# `input/chrominfo.train.txt`, so its GenerateTrainData sampled the 100 000 training instances
# from all 23 chromosomes — the same chromosomes it then imputes. The ATAC-seq traindata added
# below is generated the same way, on purpose, because the point is to apply THE EXISTING RECIPE
# to the missing mark, not to fix it. Every number that ever comes off these arrays carries that
# caveat; the fix belongs to the real run, not to this preview.
#
# NO re-Convert, NO re-ComputeGlobalDist. CONVERTEDDIR and DISTANCEDIR are symlinks into
# $FULL and are read-only here — Convert is 161 GB of intermediate bedgraphs that were already
# consumed and deleted, and ComputeGlobalDist depends only on the compendium, which has not
# changed. `stage.sh`'s `convert` and `dist` stages are never submitted by this script.
#
# NOT A RETRAIN. `Train` runs for the 51 blind targets and `GenerateTrainData` runs for ATAC-seq,
# both the existing t51 recipe applied to new targets. No model architecture, hyperparameter or
# input compendium changes.
#
# NEVER writes under $FULL or /project/def-maxwl/mforooz/rivals_src. TRAINDATADIR under $CIRUN is
# a REAL directory of symlinks (`cp -rs`) so ChromImpute can drop the new ATAC-seq files beside
# the 5 712 existing ones without touching the originals; PREDICTORDIR and OUTPUTIMPUTEDIR are new
# and real.
#
# java/21.0.1 and `unset JAVA_TOOL_OPTIONS` live inside `stage.sh` (verified on Fir, CANDII_t51
# fd33b71), so nothing here has to load a module or fight the module's -Xmx2g default.
#
# Env vars, all with the pinned defaults:
#   CIRUN CI_REPO CI_STORE CI_PY CI_JAR FULL OUTROOT REGIME CI_THROTTLE ACCT
set -euo pipefail

CIRUN=${CIRUN:-/home/mforooz/scratch/blind_preview_chromimpute}
OUTROOT=${OUTROOT:-/project/def-maxwl/mforooz/blind_preview_2026-09-01}
FULL=${FULL:-/home/mforooz/scratch/t51_chromimpute/full_genome}
CI_REPO=${CI_REPO:-/project/def-maxwl/mforooz/CANDII_t51}
CI_STORE=${CI_STORE:-/project/def-maxwl/mforooz/CANDI_STORE/eic}
CI_PY=${CI_PY:-/project/def-maxwl/mforooz/candi_venv/bin/python}
CI_JAR=${CI_JAR:-/home/mforooz/scratch/t51_chromimpute/tool/ChromImpute.jar}
REGIME=${REGIME:-/project/def-maxwl/mforooz/CANDII_t52/configs/regime.eic_test.json}
CI_THROTTLE=${CI_THROTTLE:-10}
ACCT=${ACCT:-def-maxwl}

STAGE_SH=$CI_REPO/competitors/chromimpute/slurm/stage.sh
COLLECT_PY=$CI_REPO/competitors/chromimpute/collect.py

# --- (1) guards ---------------------------------------------------------------------------------
# The reviewed code is one commit, and it is the commit that built $FULL. A different HEAD would
# be a different recipe applied to the same directory.
HEAD_SHA=$(git -C "$CI_REPO" rev-parse --short HEAD)
if [ "$HEAD_SHA" != "fd33b71" ]; then
  echo "[error] $CI_REPO is at $HEAD_SHA, expected fd33b71" >&2
  exit 2
fi

if [ ! -f "$CIRUN/prep/targets.tsv" ]; then
  echo "[error] no $CIRUN/prep/targets.tsv — run the ChromImpute dry-run/prepare step first" >&2
  exit 2
fi
NTARGET=$(wc -l < "$CIRUN/prep/targets.tsv")
if [ "$NTARGET" -ne 51 ]; then
  echo "[error] $CIRUN/prep/targets.tsv has $NTARGET lines, expected 51" >&2
  exit 2
fi

NFULLTD=$(ls -1 "$FULL/TRAINDATADIR" | wc -l)
if [ "$NFULLTD" -ne 5712 ]; then
  echo "[error] $FULL/TRAINDATADIR has $NFULLTD entries, expected 5712" >&2
  exit 2
fi

if [ ! -f "$STAGE_SH" ] || [ ! -f "$COLLECT_PY" ] || [ ! -f "$CI_JAR" ]; then
  echo "[error] missing one of: $STAGE_SH  $COLLECT_PY  $CI_JAR" >&2
  exit 2
fi

# Double submission is the expensive mistake here: a second chain would re-Train 51 predictors on
# top of the first chain's half-written ones. Any content under PREDICTORDIR means a chain has
# already run (or is running) in this $CIRUN.
if [ -d "$CIRUN/PREDICTORDIR" ] && [ -n "$(ls -A "$CIRUN/PREDICTORDIR" 2>/dev/null)" ]; then
  echo "[error] $CIRUN/PREDICTORDIR is not empty — a chain has already been submitted here." >&2
  echo "        Refusing to submit again. Inspect it, or point CIRUN at a fresh directory." >&2
  exit 2
fi

echo "[bp-ci] repo=$HEAD_SHA  regime=$REGIME"
echo "[bp-ci] run=$CIRUN  reusing=$FULL  out=$OUTROOT/ChromImpute"

# --- (2) layout ---------------------------------------------------------------------------------
mkdir -p "$CIRUN"/{input,lists,logs,timing,PREDICTORDIR,OUTPUTIMPUTEDIR,TRAINDATADIR}
mkdir -p "$OUTROOT/jobs"

cp "$FULL/input/inputinfofile.txt" "$FULL/input/chrominfo.txt" "$CIRUN/input/"
cp "$CIRUN/prep/targets.tsv" "$CIRUN/input/targets.tsv"

# Read-only reuse. `ln -sfn` so a re-run replaces the link rather than nesting one inside it.
ln -sfn "$FULL/CONVERTEDDIR" "$CIRUN/CONVERTEDDIR"
ln -sfn "$FULL/DISTANCEDIR"  "$CIRUN/DISTANCEDIR"

# TRAINDATADIR must be a real directory: `gtd` writes the new ATAC-seq files into it, and a
# symlinked directory would put them inside $FULL. `-n` so a partially populated directory is
# filled in rather than erroring.
NTD=$(ls -1 "$CIRUN/TRAINDATADIR" | wc -l)
if [ "$NTD" -lt 5712 ]; then
  cp -rsn "$FULL/TRAINDATADIR/." "$CIRUN/TRAINDATADIR/"
  NTD=$(ls -1 "$CIRUN/TRAINDATADIR" | wc -l)
fi
if [ "$NTD" -lt 5712 ]; then
  echo "[error] $CIRUN/TRAINDATADIR has $NTD entries after cp -rs, expected >= 5712" >&2
  exit 2
fi
echo "[bp-ci] TRAINDATADIR: $NTD entries (symlinks into \$FULL + anything new)"

# --- (3) work-item lists ------------------------------------------------------------------------
# gtd: ATAC-seq is the ONLY blind mark with no traindata in $FULL. `stage.sh`'s gtd stage reads
# `<mark><TAB><chrom>` and passes -c <chrom>; with one field it would take the whole-genome branch
# instead, which is the ~7.6 h path this list exists to avoid.
while read -r C; do
  printf 'ATAC-seq\t%s\n' "$C"
done < <(cut -f1 "$CIRUN/input/chrominfo.txt") > "$CIRUN/lists/gtd.txt"

# train/apply: `<input sample><TAB><mark>`, exactly as `submit.sh` builds them.
cut -f1,3 "$CIRUN/input/targets.tsv" > "$CIRUN/lists/train.txt"
cp "$CIRUN/lists/train.txt" "$CIRUN/lists/apply.txt"

NGTD=$(wc -l < "$CIRUN/lists/gtd.txt")
NTRAIN=$(wc -l < "$CIRUN/lists/train.txt")
NAPPLY=$(wc -l < "$CIRUN/lists/apply.txt")
echo "[bp-ci] lists: gtd $NGTD | train $NTRAIN | apply $NAPPLY"

# --- (4) the chain ------------------------------------------------------------------------------
ENV_COMMON="ALL,CI_RUN=$CIRUN,CI_REPO=$CI_REPO,CI_STORE=$CI_STORE,CI_PY=$CI_PY,CI_JAR=$CI_JAR,CI_CHROMS=all"

# `CI_THROTTLE` caps how many array tasks of one stage run at once. GenerateTrainData and Apply
# each hold ONE OPEN gzip reader per compendium track — 267 of them — so an unthrottled array puts
# thousands of concurrent handles on Lustre and draws transient
# "Cannot send after transport endpoint shutdown" opens that look like missing files.
sub() {  # sub <stage> <time> <mem> <mx> [afterok-jobid]
  local stage=$1 time=$2 mem=$3 mx=$4 dep=${5:-}
  local n; n=$(wc -l < "$CIRUN/lists/$stage.txt")
  local depflag=(); [ -n "$dep" ] && depflag=(--dependency=afterok:"$dep")
  sbatch --parsable --account="$ACCT" --nodes=1 --ntasks=1 --cpus-per-task=1 \
         --job-name="bp_ci_$stage" --array="0-$((n - 1))%$CI_THROTTLE" \
         --time="$time" --mem="$mem" \
         --output="$CIRUN/logs/%x_%A_%a.out" --error="$CIRUN/logs/%x_%A_%a.err" \
         "${depflag[@]}" --export="$ENV_COMMON,CI_STAGE=$stage,CI_MX=$mx" "$STAGE_SH"
}

JOB_GTD=$(sub gtd   12:00:00 24000M 20000M)
JOB_TRAIN=$(sub train 6:00:00 24000M 20000M "$JOB_GTD")
JOB_APPLY=$(sub apply 12:00:00 16000M 12000M "$JOB_TRAIN")

# --- (5) collect --------------------------------------------------------------------------------
# Written, not inlined, so the exact command that produced $OUTROOT/ChromImpute stays on disk next
# to the run. `\$(...)` is deliberately deferred to collect time.
cat > "$CIRUN/collect.sh" <<EOF
#!/bin/bash
# after bp_ci_apply: Apply's wigs -> the RIVALS_PLAN.md §4.1 prediction root. Throwaway preview.
set -euo pipefail
PYTHONPATH=$CI_REPO/src $CI_PY $COLLECT_PY --store $CI_STORE \\
    --targets $CIRUN/input/targets.tsv --impute-dir $CIRUN/OUTPUTIMPUTEDIR \\
    --pred-root $OUTROOT/ChromImpute \\
    --chroms \$(cut -f1 $CIRUN/input/chrominfo.txt | paste -sd, -) \\
    --jar $CI_JAR \\
    --notes "blind preview 2026-09-01: regime.eic_test 51 B_ targets on all 23 chroms; predictors from t51 full_genome recipe (CANDII_t51 fd33b71), ATAC-seq traindata added; store manifest pre_t78p3 6c0e0c3e at fit time, live c9a95e4e at apply time; throwaway"
EOF
chmod +x "$CIRUN/collect.sh"

JOB_COLLECT=$(sbatch --parsable --account="$ACCT" --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --job-name=bp_ci_collect --time=4:00:00 --mem=16000M \
    --output="$CIRUN/logs/%x_%j.out" --error="$CIRUN/logs/%x_%j.err" \
    --dependency=afterok:"$JOB_APPLY" "$CIRUN/collect.sh")

# --- (6) the job-id record ----------------------------------------------------------------------
JOBIDS=$OUTROOT/jobs/chromimpute.jobids
{
  printf 'gtd\t%s\n'     "$JOB_GTD"
  printf 'train\t%s\n'   "$JOB_TRAIN"
  printf 'apply\t%s\n'   "$JOB_APPLY"
  printf 'collect\t%s\n' "$JOB_COLLECT"
} | tee "$JOBIDS"

echo "[bp-ci] job ids written to $JOBIDS"
echo "[bp-ci] watch with: squeue -u \$USER -n bp_ci_gtd,bp_ci_train,bp_ci_apply,bp_ci_collect"
