#!/bin/bash
# t125 part 2: train one learned h (the map from a count and the control to P = -log10 p) on a
# 10 GB MIG slice,
# either on the true X' (table "true": 12 learned h x 3 seeds = 36 tasks) or on one f's predicted
# NB (table "fnb": the 64 seed-0 real f runs x 12 learned h x fnb_seeds; 768 tasks at fnb_seeds 1,
# PI decision (ii) b; 2304 at fnb_seeds 3). One array task = one h.
#
# WHAT ONE TASK DOES (runs_dir = <out_root>/runs_<table>):
#   python3 $KIT/tools/t125/hmap/train.py train-index <manifest> <covariates.tsv> <cache_dir> \
#       <ctl_cache> <runs_dir> <table> <index> [--eta-runs <eta_runs_dir> --fnb-seeds <N>]
#           -> <runs_dir>/<h_name>/config.json, ckpt.pt, train_log.tsv, timing.json, TRAIN_DONE
# The "fnb" table always passes --eta-runs (train.py resolves the row's fnb:<f_run> to
# <eta_runs_dir>/<f_run>) and --fnb-seeds. A task whose TRAIN_DONE exists exits 0 without work. An
# fnb task whose f run has no TRAIN_DONE exits 3, so an unfinished eta run shows up as a failed
# task instead of a crash inside train.py.
#
# THE TASK TABLE. Index -> h comes from `python3 tools/t125/hmap/tasks.py true <manifest>` or
# `tasks.py fnb <manifest> --fnb-seeds N` (true index 0 = h_affine_r1_push_true_s0; fnb index 0 =
# h_affine_r1_push_A_C07M20_counts_real_s0_s0). The first task with a venv writes it to
# <out_root>/tasks_true.tsv or <out_root>/tasks_fnb_s<N>.tsv; a later task reads that copy BEFORE
# its venv only to exit early when its h is done, then checks it against a fresh tasks.py table (a
# mismatch exits 2). The copy is created once and never replaced (stale file handles under
# concurrent readers on /project, t118 job 22657297). An <out_root> or <eta_runs_dir> under a t118
# directory is refused (exit 2): t118 runs carry the same run names as t125's f runs.
#
# NO --export. Every input is a POSITIONAL argument; the task index is $SLURM_ARRAY_TASK_ID.
# Every output is under <out_root>:
#   OUT_H=/project/def-maxwl/mforooz/t125/h
#   CTL=/project/def-maxwl/mforooz/t125/ctl_cache     (h_ctl_cache.sh builds it)
#   CACHE=/project/def-maxwl/mforooz/t118/ladder/cache (t118's count and pval cache, read-only here)
#   ETA_RUNS=/project/def-maxwl/mforooz/t125/eta/runs  (fnb only; the f runs, read-only here)
#
# THE VENV is the pinned recipe of slurm/t118/row2_train.sh (python/3.10.13, requirements-fir.txt,
# then requirements-pypi.txt from $KIT/wheels), built per task in $SLURM_TMPDIR; PYTHONPATH is
# "$KIT/src" and the t125 CLIs put tools/t118 and tools/t125 on sys.path themselves.
#
# RESOURCES (t125-P2h). One 10 GB MIG slice of an H100, 8 cores (OMP/MKL threads 8; train.py
# draws the windows of an f-NB batch in 8 threads and takes log1p on the host in up to 8), 40000M
# host memory, 1 day (partition b3). Nodes c128, c166, c537 and c371 are excluded. Memory: the N4
# true-X' fits (K10, CPU) peaked at 31.2 GiB of 31.25. No --device is passed: train.py picks its
# default device, the slice. The Nibi smoke (N2, kit K10) found both fnb fits over 3 h (and over
# 12 h, job 24083625) on 8 CPU cores; t125-P2h builds the f-NB chunks on the slice with the same
# numbers (bit for bit on a CPU device; on the slice the features are bit for bit too). Measured
# on the slice for f A_C19M16_counts_real_s0 (38 pairs): a training step 0.26-0.64 s, a chr22
# validation pass 346 s, so a fit of 4000 steps and 17 passes about 2.2 h (the widest-support
# C19M16 f; the row-2 f are cheaper). 1 day leaves room (a longer request starts days later, see
# h_chain.sh). The 36 true-X' fits of N4 ran on CPU with kit K10; this script reruns a true fit on
# the slice (float32 results then differ from a CPU run in the last bits).
# SCOPE (PI 2026-10-17): the fnb fits are for the 8 real seed-0 f of track C19M16 only (A, B, C,
# D, A2, B2, C2, D2): 96 fits, fnb indices 36-47,132-143,228-239,324-335,420-431,516-527,
# 612-623,708-719 (fnb index = f index x 12 + h version; f index 3, 11, ..., 59).
#
# DRY_RUN=1 prints the train command line (nothing when TRAIN_DONE exists; a note on stderr when
# the fnb f run has no TRAIN_DONE, where a real task would exit 3) and exits 0 before any module
# load. It resolves the task table with ${PYTHON:-python3} (needs numpy, scipy, torch: run it with
# the candii env active or PYTHON set) from $KIT when $KIT holds tools/t125/hmap/tasks.py, else
# from the checkout this script sits in; and from <products_dir>/MANIFEST.tsv when it exists, else
# from that checkout's tests/fixtures/t112_meta/MANIFEST.tsv (the same file, md5-pinned).
#
# Usage, from the Nibi login node, after snapshotting the repo to $KIT (kit K9 and up). Every job
# id goes into plan/T125_STATUS.md:
#   OUT_H=/project/def-maxwl/mforooz/t125/h; mkdir -p $OUT_H/logs    # SLURM opens --output first
#   CTL=/project/def-maxwl/mforooz/t125/ctl_cache; CACHE=/project/def-maxwl/mforooz/t118/ladder/cache
#   # h on the true X' (36):
#   sbatch --test-only --array=0-35%36 --exclude=c128,c166,c537,c371 --output=$OUT_H/logs/%x_%A_%a.out \
#       $KIT/slurm/t125/h_train.sh $KIT $PRODUCTS $CACHE $CTL $OUT_H true
#   sbatch --parsable --array=0-35%36 --exclude=c128,c166,c537,c371 --output=$OUT_H/logs/%x_%A_%a.out \
#       $KIT/slurm/t125/h_train.sh $KIT $PRODUCTS $CACHE $CTL $OUT_H true
#   # h on f's NB, the 96 in scope (the 8 C19M16 f; the table has 768 at one h seed):
#   sbatch --parsable --array=36-47,132-143,228-239,324-335,420-431,516-527,612-623,708-719%40 \
#       --exclude=c128,c166,c537,c371 --output=$OUT_H/logs/%x_%A_%a.out \
#       $KIT/slurm/t125/h_train.sh $KIT $PRODUCTS $CACHE $CTL $OUT_H fnb /project/def-maxwl/mforooz/t125/eta/runs 1
# %40 is safe because each task builds its own venv in $SLURM_TMPDIR. Never chain a job onto an
# array that may have finished: verify finished work by its TRAIN_DONE files.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t125_htrain
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --exclude=c128,c166,c537,c371
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40000M

set -euo pipefail

USAGE="usage: h_train.sh <kit_dir> <products_dir> <cache_dir> <ctl_cache> <out_root> <true|fnb> [eta_runs_dir] [fnb_seeds 1|3]"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
CACHE="${3:?$USAGE}"
CTL="${4:?$USAGE}"
OUT_ROOT="${5:?$USAGE}"
TABLE="${6:?$USAGE}"
ETA_RUNS="${7:-}"
FNB_SEEDS="${8:-1}"
IDX="${SLURM_ARRAY_TASK_ID:?array task only}"
[[ "$IDX" =~ ^[0-9]+$ ]] || { echo "SLURM_ARRAY_TASK_ID '$IDX' is not an index" >&2; exit 2; }
case "$TABLE" in
  true|fnb) ;;
  *) echo "table '$TABLE' is not true or fnb; $USAGE" >&2; exit 2 ;;
esac
case "$OUT_ROOT" in
  */t118|*/t118/*) echo "out root $OUT_ROOT is under a t118 directory; t125 writes only under t125/" >&2; exit 2 ;;
esac
if [ "$TABLE" = fnb ]; then
  [ -n "$ETA_RUNS" ] || { echo "table fnb needs <eta_runs_dir>; $USAGE" >&2; exit 2; }
  case "$FNB_SEEDS" in
    1|3) ;;
    *) echo "fnb_seeds '$FNB_SEEDS' is not 1 or 3" >&2; exit 2 ;;
  esac
  case "$ETA_RUNS" in
    */t118|*/t118/*) echo "eta runs $ETA_RUNS are under a t118 directory; h is trained on t125's f runs" >&2; exit 2 ;;
  esac
  TABLE_ARGS=(fnb --fnb-seeds "$FNB_SEEDS")
  EXTRA_ARGS=(--eta-runs "$ETA_RUNS" --fnb-seeds "$FNB_SEEDS")
  TASKS_TSV="$OUT_ROOT/tasks_fnb_s$FNB_SEEDS.tsv"
else
  [ -z "$ETA_RUNS" ] || { echo "table true takes no <eta_runs_dir>; $USAGE" >&2; exit 2; }
  TABLE_ARGS=(true)
  EXTRA_ARGS=()
  TASKS_TSV="$OUT_ROOT/tasks_true.tsv"
fi
PY_MODULES="python/3.10.13"
MANIFEST="$PRODUCTS/MANIFEST.tsv"
COVARIATES="$KIT/tools/t118/covariates.tsv"
RUNS="$OUT_ROOT/runs_$TABLE"

# "<h_name>\t<source>" of task $IDX, read from a tasks.py table on stdin (columns found by header).
row_from() {
  awk -F'\t' -v i="$IDX" 'NR == 1 { for (c = 1; c <= NF; c++) h[$c] = c; next }
                          $h["index"] == i { print $h["h_name"] "\t" $h["source"] }'
}
# The f run of an fnb source ("fnb:<f_run>"), empty for "true".
f_run_of() { case "$1" in fnb:*) echo "${1#fnb:}" ;; *) echo "" ;; esac; }

if [ "${DRY_RUN:-0}" = 1 ]; then
  TOOLS="$KIT"
  [ -f "$TOOLS/tools/t125/hmap/tasks.py" ] || TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  TABLE_MANIFEST="$MANIFEST"
  [ -f "$TABLE_MANIFEST" ] || TABLE_MANIFEST="$TOOLS/tests/fixtures/t112_meta/MANIFEST.tsv"
  ROWS=$(PYTHONPATH="$TOOLS/src${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-python3}" \
         "$TOOLS/tools/t125/hmap/tasks.py" "${TABLE_ARGS[0]}" "$TABLE_MANIFEST" "${TABLE_ARGS[@]:1}")
  ROW=$(row_from <<< "$ROWS")
  [ -n "$ROW" ] || { echo "no task $IDX in the tasks.py $TABLE table" >&2; exit 2; }
  H=${ROW%%$'\t'*}; F_RUN=$(f_run_of "${ROW#*$'\t'}")
  if [ -f "$RUNS/$H/TRAIN_DONE" ]; then echo "skip $H: TRAIN_DONE exists" >&2; exit 0; fi
  if [ -n "$F_RUN" ] && [ ! -f "$ETA_RUNS/$F_RUN/TRAIN_DONE" ]; then
    echo "note: no $ETA_RUNS/$F_RUN/TRAIN_DONE; a real task would exit 3" >&2
  fi
  echo "python3 $KIT/tools/t125/hmap/train.py train-index $MANIFEST $COVARIATES $CACHE $CTL $RUNS $TABLE $IDX${EXTRA_ARGS[*]:+ ${EXTRA_ARGS[*]}}"
  exit 0
fi

[ -f "$KIT/tools/t125/hmap/train.py" ] || { echo "no tools/t125/hmap/train.py under $KIT" >&2; exit 2; }
[ -f "$KIT/tools/t125/hmap/tasks.py" ] || { echo "no tools/t125/hmap/tasks.py under $KIT" >&2; exit 2; }
[ -f "$COVARIATES" ] || { echo "no $COVARIATES" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "no $MANIFEST" >&2; exit 2; }
[ -d "$CACHE" ] || { echo "no cache dir $CACHE (t118 ladder_cache.sh builds it)" >&2; exit 2; }
[ -d "$CTL" ] || { echo "no control cache $CTL (h_ctl_cache.sh builds it)" >&2; exit 2; }
if [ "$TABLE" = fnb ]; then [ -d "$ETA_RUNS" ] || { echo "no eta runs dir $ETA_RUNS" >&2; exit 2; }; fi
KIT=$(cd "$KIT" && pwd)
COVARIATES="$KIT/tools/t118/covariates.tsv"
mkdir -p "$RUNS"

echo "=== t125 h_train table=$TABLE idx=$IDX job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX host=$(hostname) $(date -u)"
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || git -C "$KIT" rev-parse HEAD 2>/dev/null || echo unknown)"

# Early exits before the venv, only from a table an earlier task already wrote with tasks.py.
if [ -f "$TASKS_TSV" ]; then
  ROW=$(row_from < "$TASKS_TSV")
  if [ -n "$ROW" ]; then
    H=${ROW%%$'\t'*}; F_RUN=$(f_run_of "${ROW#*$'\t'}")
    if [ -f "$RUNS/$H/TRAIN_DONE" ]; then echo "skip $H: TRAIN_DONE exists"; exit 0; fi
    if [ -n "$F_RUN" ] && [ ! -f "$ETA_RUNS/$F_RUN/TRAIN_DONE" ]; then
      echo "no $ETA_RUNS/$F_RUN/TRAIN_DONE: f run $F_RUN did not finish" >&2; exit 3
    fi
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
if [ -f "$KIT/GIT_SHA" ]; then export T125_GIT_SHA=$(cat "$KIT/GIT_SHA"); fi
python3 -c "import candi.metrics, candi.bench.distributional, candi.store.genome; print('venv ok', candi.__file__)"
# the library must come from this kit, not from some other install (tests/test_slurm_kit_pin.py)
case "$(python3 -c 'import candi; print(candi.__file__)')" in
  "$KIT"/src/*) ;;
  *) echo "candi does not import from $KIT/src" >&2; exit 3 ;;
esac

# The authoritative task table, from tasks.py; a stale shared copy is an error, not a guess.
TMP_TABLE="$TASKS_TSV.tmp.${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX"
python3 "$KIT/tools/t125/hmap/tasks.py" "${TABLE_ARGS[0]}" "$MANIFEST" "${TABLE_ARGS[@]:1}" > "$TMP_TABLE"
# cmp exit 1 = the tables differ (an error); exit 2 = a read problem (not fatal).
ROW=$(row_from < "$TMP_TABLE")
if [ -f "$TASKS_TSV" ]; then
  CMP_RC=0; cmp -s "$TMP_TABLE" "$TASKS_TSV" || CMP_RC=$?
  if [ "$CMP_RC" = 1 ]; then
    echo "$TASKS_TSV differs from this kit's tasks.py table; remove it or fix the kit" >&2
    rm -f "$TMP_TABLE"; exit 2
  fi
else
  mv -n "$TMP_TABLE" "$TASKS_TSV" 2>/dev/null || true
fi
rm -f "$TMP_TABLE"
[ -n "$ROW" ] || { echo "no task $IDX in the tasks.py $TABLE table" >&2; exit 2; }
H=${ROW%%$'\t'*}; F_RUN=$(f_run_of "${ROW#*$'\t'}")
echo "h: $H${F_RUN:+ (f run $F_RUN)}"
if [ -f "$RUNS/$H/TRAIN_DONE" ]; then echo "skip $H: TRAIN_DONE exists"; exit 0; fi
if [ -n "$F_RUN" ] && [ ! -f "$ETA_RUNS/$F_RUN/TRAIN_DONE" ]; then
  echo "no $ETA_RUNS/$F_RUN/TRAIN_DONE: f run $F_RUN did not finish" >&2; exit 3
fi

python3 "$KIT/tools/t125/hmap/train.py" train-index "$MANIFEST" "$COVARIATES" "$CACHE" "$CTL" \
    "$RUNS" "$TABLE" "$IDX" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "=== done $H $(date -u)"
