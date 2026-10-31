#!/bin/bash
# t128 phase 3: train one run of the two-arm table (plan/T126_ENCDEC_D2.md "Phase 3: PI decisions,
# 2026-10-27") on a 10 GB MIG slice on Fir, depth 7.5M held out of training, 15,000 steps, then
# score its trained pairs, its held-out pairs, shuffle, swap and depth law. Both arms are old rung D
# (form D): Dbigg = g hidden 256, f 32 channels; Dbigf = g hidden 64, f 64 channels (n_theta 512).
# One array task = one run; 48 tasks. This is slurm/t128p2/train.sh with the t128p3 task table,
# tools/t128p3/train.py, a 24 h limit and the t128p3 out dir. No t118, t125, t126, t128 or t128p2
# script, run or out dir is touched.
#
# WHAT ONE TASK DOES.
#   1. python3 $KIT/tools/t128p3/train.py train-index <manifest> <covariates.tsv> <cache_dir> \
#          <out_dir>/runs <index> [--max-steps $6] [--eval-every $7]
#                                                  -> <out_dir>/runs/<run_name>/ckpt.pt, TRAIN_DONE
#   2. python3 $KIT/tools/t118/ladder/score.py trained <manifest> <covariates.tsv> <cache_dir> \
#          <blacklist> <out_dir>/runs/<run_name> --workers 4   -> scores.json, figdata.npz, SCORE_DONE
# A task whose SCORE_DONE exists exits 0 without work.
#
# THE TASK TABLE. Index -> run comes from `python3 tools/t128p3/pairs.py tasks <manifest>` (48 rows:
# 2 arms x 8 g x 3 seeds, arm slowest; 9 = Dbigg_C19M16_counts_real_s0, 33 = Dbigf_C19M16_...).
# The first task with a venv writes it to <out_dir>/tasks.tsv; later tasks read that copy only to
# exit early, then check it against a fresh table (mismatch exits 2). The out dir must be under a
# t128p3 directory; t118, t125, t126, t128 (phase 1) and t128p2 directories are refused.
#
# NO --export. Every input is a POSITIONAL argument; the task index is $SLURM_ARRAY_TASK_ID.
#   OUT=/project/def-maxwl/mforooz/t128p3
#   CACHE=/project/def-maxwl/mforooz/t118/ladder/cache   (t118's count cache, read-only here)
#   BLACKLIST=/project/def-maxwl/mforooz/DATA_CANDI_MERGED/hg38_blacklist_v2.bed   (required)
#
# THE VENV is t118's pinned recipe (python/3.10.13, requirements-fir.txt, then
# requirements-pypi.txt from $KIT/wheels), built per task in $SLURM_TMPDIR.
#
# RESOURCES. One 10 GB MIG slice, 4 cores, 16 GB, 24 h (15,000 steps at up to ~5 s/step for the
# 64-channel trunk; the submitter may override --time per arm from the smoke's s/step); Fir nodes
# fc10713 (CUDA busy) and fc10715 (slow) excluded.
#
# DRY_RUN=1 prints the two python command lines (nothing when SCORE_DONE exists) and exits 0
# before any module load (needs numpy, scipy, torch through ${PYTHON:-python3}).
#
# Usage, from the Fir login node, after snapshotting the repo to $KIT:
#   smoke (one run per arm, 300 steps, eval every 100, its own tree):
#     mkdir -p $OUT/smoke/logs
#     TRAIN=$(sbatch --parsable --array=9,33%2 --time=3:00:00 \
#         --output=$OUT/smoke/logs/%x_%A_%a.out \
#         $KIT/slurm/t128p3/train.sh $KIT $PRODUCTS $CACHE $OUT/smoke $BLACKLIST 300 100)
#   full (one array per arm, each chained to its own law array):
#     mkdir -p $OUT/logs
#     TRAIN=$(sbatch --parsable --array=0-23%12 --output=$OUT/logs/%x_%A_%a.out \
#         $KIT/slurm/t128p3/train.sh $KIT $PRODUCTS $CACHE $OUT $BLACKLIST)      # Dbigg
#     TRAIN=$(sbatch --parsable --array=24-47%12 --output=$OUT/logs/%x_%A_%a.out \
#         $KIT/slurm/t128p3/train.sh $KIT $PRODUCTS $CACHE $OUT $BLACKLIST)      # Dbigf
# then, at once after each, slurm/t128p3/law.sh on the same array with
# --dependency=aftercorr:$TRAIN on the LIVE train array (aftercorr on a finished array is refused).
#SBATCH --account=def-maxwl
#SBATCH --job-name=t128p3_train
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --exclude=fc10713,fc10715
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M

set -euo pipefail

BLACKLIST_SHA256_PREFIX=31c69342
USAGE="usage: t128p3/train.sh <kit_dir> <products_dir> <cache_dir> <out_dir> <blacklist.bed> [max_steps [eval_every]]"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
CACHE="${3:?$USAGE}"
OUT="${4:?$USAGE}"
BLACKLIST="${5:?$USAGE}"
MAX_STEPS="${6:-}"
EVAL_EVERY="${7:-}"
IDX="${SLURM_ARRAY_TASK_ID:?array task only}"
[[ "$IDX" =~ ^[0-9]+$ ]] || { echo "SLURM_ARRAY_TASK_ID '$IDX' is not an index" >&2; exit 2; }
[[ -z "$MAX_STEPS" || "$MAX_STEPS" =~ ^[0-9]+$ ]] || { echo "max_steps '$MAX_STEPS' is not an integer" >&2; exit 2; }
[[ -z "$EVAL_EVERY" || "$EVAL_EVERY" =~ ^[0-9]+$ ]] || { echo "eval_every '$EVAL_EVERY' is not an integer" >&2; exit 2; }
case "$OUT" in
  */t118*|*/t125*|*/t126*|*/t128|*/t128/*|*/t128p2|*/t128p2/*) echo "out dir $OUT is under a t118/t125/t126/t128/t128p2 directory; t128p3 writes only under t128p3/" >&2; exit 2 ;;
esac
case "$OUT" in
  */t128p3|*/t128p3/*) ;;
  *) echo "out dir $OUT is not under a t128p3 directory" >&2; exit 2 ;;
esac
PY_MODULES="python/3.10.13"
MANIFEST="$PRODUCTS/MANIFEST.tsv"
COVARIATES="$KIT/tools/t118/covariates.tsv"
RUNS="$OUT/runs"
TASKS_TSV="$OUT/tasks.tsv"
EXTRA=""
[ -z "$MAX_STEPS" ] || EXTRA="$EXTRA --max-steps $MAX_STEPS"
[ -z "$EVAL_EVERY" ] || EXTRA="$EXTRA --eval-every $EVAL_EVERY"

# The run_name of task $IDX, read from a pairs.py task table on stdin (columns found by header).
run_name_from() {
  awk -F'\t' -v i="$IDX" 'NR == 1 { for (c = 1; c <= NF; c++) h[$c] = c; next }
                          $h["index"] == i { print $h["run_name"] }'
}

if [ "${DRY_RUN:-0}" = 1 ]; then
  TOOLS="$KIT"
  [ -f "$TOOLS/tools/t128p3/pairs.py" ] || TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  TABLE_MANIFEST="$MANIFEST"
  [ -f "$TABLE_MANIFEST" ] || TABLE_MANIFEST="$TOOLS/tests/fixtures/t112_meta/MANIFEST.tsv"
  TABLE=$(PYTHONPATH="$TOOLS/src${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-python3}" \
          "$TOOLS/tools/t128p3/pairs.py" tasks "$TABLE_MANIFEST")
  RUN=$(run_name_from <<< "$TABLE")
  [ -n "$RUN" ] || { echo "no task $IDX in the pairs.py table" >&2; exit 2; }
  if [ -f "$RUNS/$RUN/SCORE_DONE" ]; then echo "skip $RUN: SCORE_DONE exists" >&2; exit 0; fi
  echo "python3 $KIT/tools/t128p3/train.py train-index $MANIFEST $COVARIATES $CACHE $RUNS $IDX$EXTRA"
  echo "python3 $KIT/tools/t118/ladder/score.py trained $MANIFEST $COVARIATES $CACHE $BLACKLIST $RUNS/$RUN --workers 4"
  exit 0
fi

[ -f "$KIT/tools/t128p3/train.py" ] || { echo "no tools/t128p3/train.py under $KIT" >&2; exit 2; }
[ -f "$KIT/tools/t118/ladder/score.py" ] || { echo "no tools/t118/ladder/score.py under $KIT" >&2; exit 2; }
[ -f "$COVARIATES" ] || { echo "no $COVARIATES" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "no $MANIFEST" >&2; exit 2; }
[ -d "$CACHE" ] || { echo "no cache dir $CACHE (run ladder_cache.sh first)" >&2; exit 2; }
[ -s "$BLACKLIST" ] || { echo "blacklist $BLACKLIST missing or empty" >&2; exit 2; }
BL_SHA=$(sha256sum "$BLACKLIST" | cut -d' ' -f1)
case "$BL_SHA" in
  "$BLACKLIST_SHA256_PREFIX"*) ;;
  *) echo "blacklist sha256 $BL_SHA does not start $BLACKLIST_SHA256_PREFIX" >&2; exit 2 ;;
esac
KIT=$(cd "$KIT" && pwd)
COVARIATES="$KIT/tools/t118/covariates.tsv"
mkdir -p "$RUNS"

echo "=== t128p3 train idx=$IDX job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX host=$(hostname) $(date -u)"
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || git -C "$KIT" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "blacklist sha256: $BL_SHA"
echo "step overrides:${EXTRA:- none}"

# Early exit before the venv, only from a table an earlier task already wrote with pairs.py.
if [ -f "$TASKS_TSV" ]; then
  RUN=$(run_name_from < "$TASKS_TSV")
  if [ -n "$RUN" ] && [ -f "$RUNS/$RUN/SCORE_DONE" ]; then
    echo "skip $RUN: SCORE_DONE exists"; exit 0
  fi
fi

set +u; module load $PY_MODULES; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
ls "$KIT"/wheels/x_transformers-*-py3-none-any.whl >/dev/null 2>&1 \
  || { echo "no x_transformers wheel in $KIT/wheels" >&2; exit 2; }
pip install --no-index -r "$KIT/requirements-fir.txt"
pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
export PYTHONPATH="$KIT/src"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
if [ -f "$KIT/GIT_SHA" ]; then export T118_GIT_SHA=$(cat "$KIT/GIT_SHA"); fi
python3 -c "import candi.metrics, candi.bench.distributional, candi.store.genome; print('venv ok', candi.__file__)"
# the library must come from this kit, not from some other install (tests/test_slurm_kit_pin.py)
case "$(python3 -c 'import candi; print(candi.__file__)')" in
  "$KIT"/src/*) ;;
  *) echo "candi does not import from $KIT/src" >&2; exit 3 ;;
esac

# The authoritative task table, from pairs.py; a stale shared copy is an error, not a guess.
TMP_TABLE="$TASKS_TSV.tmp.${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX"
python3 "$KIT/tools/t128p3/pairs.py" tasks "$MANIFEST" > "$TMP_TABLE"
# The run comes from this task's own table. The shared copy is created once and never replaced:
# replacing it under 40 concurrent readers on /project gave "Stale file handle" (8 failed tasks,
# job 22657297). cmp exit 1 = the tables differ (an error); exit 2 = a read problem (not fatal).
RUN=$(run_name_from < "$TMP_TABLE")
if [ -f "$TASKS_TSV" ]; then
  CMP_RC=0; cmp -s "$TMP_TABLE" "$TASKS_TSV" || CMP_RC=$?
  if [ "$CMP_RC" = 1 ]; then
    echo "$TASKS_TSV differs from this kit's pairs.py table; remove it or fix the kit" >&2
    rm -f "$TMP_TABLE"; exit 2
  fi
else
  mv -n "$TMP_TABLE" "$TASKS_TSV" 2>/dev/null || true
fi
rm -f "$TMP_TABLE"
[ -n "$RUN" ] || { echo "no task $IDX in the pairs.py table" >&2; exit 2; }
echo "run: $RUN"
if [ -f "$RUNS/$RUN/SCORE_DONE" ]; then echo "skip $RUN: SCORE_DONE exists"; exit 0; fi

STEP_ARGS=()
[ -z "$MAX_STEPS" ] || STEP_ARGS+=(--max-steps "$MAX_STEPS")
[ -z "$EVAL_EVERY" ] || STEP_ARGS+=(--eval-every "$EVAL_EVERY")
python3 "$KIT/tools/t128p3/train.py" train-index "$MANIFEST" "$COVARIATES" "$CACHE" "$RUNS" "$IDX" \
    ${STEP_ARGS[@]+"${STEP_ARGS[@]}"}
echo "=== trained $(date -u)"
python3 "$KIT/tools/t118/ladder/score.py" trained "$MANIFEST" "$COVARIATES" "$CACHE" "$BLACKLIST" \
    "$RUNS/$RUN" --workers 4

echo "=== done $RUN $(date -u)"
