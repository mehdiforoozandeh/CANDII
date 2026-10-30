#!/bin/bash
# t136 (h28): train one run of the 42-row arm-to-arm grid (plan/T136_ARM2ARM_GRID.md) on a 10 GB
# MIG slice on Nibi, depth 7.5M held out of training, then score its trained pairs, held-out pairs,
# shuffle, swap and depth law. Arms: Darm1 and Darm2, both phase 2's old D (g hidden 64, f 32
# channels, n_theta 256, 12,000 steps) plus extra arm->arm training pairs (Darm1: depth 15M <->
# depth 3.75M; Darm2: every arm->arm pair without the held-out 7.5M product). 7 per-track g x seeds
# 0, 1, 2. This is slurm/t128p3/train_3b.sh with `tools/t136/pairs.py tasks`,
# `tools/t136/train.py train-index` and the out dir t136. No t128p2/t128p3 run or out dir is touched.
#
# WHAT ONE TASK DOES.
#   1. python3 $KIT/tools/t136/train.py train-index <manifest> <covariates.tsv> <cache_dir> \
#          <out_dir>/runs <index> [--max-steps $6] [--eval-every $7]
#                                                  -> <out_dir>/runs/<run_name>/ckpt.pt, TRAIN_DONE
#   2. python3 $KIT/tools/t118/ladder/score.py trained <manifest> <covariates.tsv> <cache_dir> \
#          <blacklist> <out_dir>/runs/<run_name>   -> scores.json, figdata.npz, SCORE_DONE
#      (no --workers: score.py's default_workers() reads the CPU affinity, i.e. the 2 allocated cores)
# A task whose SCORE_DONE exists exits 0 without work.
#
# THE TASK TABLE. Index -> run comes from `python3 tools/t136/pairs.py tasks <manifest>` (42 rows:
# 2 arms x 7 g x 3 seeds, arm slowest; 0 = Darm1_C07M20_counts_real_s0, 21 =
# Darm2_C07M20_counts_real_s0). The first task with a venv writes it to <out_dir>/tasks.tsv; later
# tasks read that copy only to exit early, then check it against a fresh table (mismatch exits 2).
# The out dir must be t136 or under it; any other directory is refused.
#
# NO --export. Every input is a POSITIONAL argument; the task index is $SLURM_ARRAY_TASK_ID.
# All paths are on Nibi:
#   KIT=/project/def-maxwl/mforooz/t136/code/K<n>
#   PRODUCTS=/project/def-maxwl/mforooz/t112_cf/products
#   CACHE=/project/def-maxwl/mforooz/t118/ladder/cache   (t118's count cache, read-only here)
#   OUT=/project/def-maxwl/mforooz/t136
#   BLACKLIST=/project/def-maxwl/mforooz/EIC_REPRO/002/scripts/hg38_blacklist_v2.bed   (required)
#
# THE VENV is t118's pinned recipe as slurm/t126/train.sh runs it on Nibi (module python/3.10.13,
# requirements-fir.txt, then requirements-pypi.txt from $KIT/wheels), built per task in $SLURM_TMPDIR.
#
# RESOURCES (PI ruling 2026-10-29). One 10 GB MIG slice, 2 cores, 10000M, 7 h, on Nibi; nodes
# c128, c166, c537, c371 (the t126 list) and c329 excluded.
#
# RUNTIME RISK. Validation scores EVERY training pair over chr22 at each eval (every 250 steps):
# Darm2 has 342 pairs per histone track against 36 for Darm1/old D, so one eval costs ~9.5x more.
# t127's old-D logs show 25-106 s for the 36-pair step-0 eval, so a Darm2 eval may take ~4-17 min.
# Run SMOKE row 30 (Darm2_C19M16_counts_real_s0) FIRST, read the `seconds` column of its
# runs/<run>/train_log.tsv, and only then submit the full array.
#
# DRY_RUN=1 prints the two python command lines (nothing when SCORE_DONE exists) and exits 0
# before any module load (needs numpy, scipy, torch through ${PYTHON:-python3}).
#
# Usage, from the Nibi login node, after snapshotting the repo to $KIT:
#     mkdir -p $OUT/logs
#     TRAIN=$(sbatch --parsable --array=0-20%12 --output=$OUT/logs/%x_%A_%a.out \
#         $KIT/slurm/t136/train.sh $KIT $PRODUCTS $CACHE $OUT $BLACKLIST)      # Darm1
#     TRAIN=$(sbatch --parsable --array=21-41%12 --output=$OUT/logs/%x_%A_%a.out \
#         $KIT/slurm/t136/train.sh $KIT $PRODUCTS $CACHE $OUT $BLACKLIST)      # Darm2
#     sbatch --parsable --array=30 --output=$OUT/logs/%x_%A_%a.out \
#         $KIT/slurm/t136/train.sh $KIT $PRODUCTS $CACHE $OUT/smoke $BLACKLIST   # Darm2 smoke, first
# then, at once after each, slurm/t136/law.sh on the same array with
# --dependency=aftercorr:$TRAIN on the LIVE train array (aftercorr on a finished array is refused).
#SBATCH --account=def-maxwl
#SBATCH --job-name=t136_train
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --exclude=c128,c166,c537,c371,c329
#SBATCH --time=7:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=10000M

set -euo pipefail

BLACKLIST_SHA256_PREFIX=31c69342
USAGE="usage: t136/train.sh <kit_dir> <products_dir> <cache_dir> <out_dir> <blacklist.bed> [max_steps [eval_every]]"
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
  */t136|*/t136/*) ;;
  *) echo "out dir $OUT is not t136 or under it; t136 writes only there" >&2; exit 2 ;;
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
  [ -f "$TOOLS/tools/t136/pairs.py" ] || TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  TABLE_MANIFEST="$MANIFEST"
  [ -f "$TABLE_MANIFEST" ] || TABLE_MANIFEST="$TOOLS/tests/fixtures/t112_meta/MANIFEST.tsv"
  TABLE=$(PYTHONPATH="$TOOLS/src${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-python3}" \
          "$TOOLS/tools/t136/pairs.py" tasks "$TABLE_MANIFEST")
  RUN=$(run_name_from <<< "$TABLE")
  [ -n "$RUN" ] || { echo "no task $IDX in the pairs.py table" >&2; exit 2; }
  if [ -f "$RUNS/$RUN/SCORE_DONE" ]; then echo "skip $RUN: SCORE_DONE exists" >&2; exit 0; fi
  echo "python3 $KIT/tools/t136/train.py train-index $MANIFEST $COVARIATES $CACHE $RUNS $IDX$EXTRA"
  echo "python3 $KIT/tools/t118/ladder/score.py trained $MANIFEST $COVARIATES $CACHE $BLACKLIST $RUNS/$RUN"
  exit 0
fi

[ -f "$KIT/tools/t136/train.py" ] || { echo "no tools/t136/train.py under $KIT" >&2; exit 2; }
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

echo "=== t136 train idx=$IDX job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX host=$(hostname) $(date -u)"
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
python3 "$KIT/tools/t136/pairs.py" tasks "$MANIFEST" > "$TMP_TABLE"
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
python3 "$KIT/tools/t136/train.py" train-index "$MANIFEST" "$COVARIATES" "$CACHE" "$RUNS" "$IDX" \
    ${STEP_ARGS[@]+"${STEP_ARGS[@]}"}
echo "=== trained $(date -u)"
python3 "$KIT/tools/t118/ladder/score.py" trained "$MANIFEST" "$COVARIATES" "$CACHE" "$BLACKLIST" \
    "$RUNS/$RUN"

echo "=== done $RUN $(date -u)"
