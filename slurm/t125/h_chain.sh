#!/bin/bash
# t125 part 2: score every h IN THE CHAIN on one f — the f's predicted NB on each of its training
# pairs pushed through each h, CRPS exact through the push-forward — on a 10 GB MIG slice. One
# array task = one f: the 64 seed-0 real f runs of rows 1 and 2 (eta.pairs.f_runs_seed0, task
# order; index 0 = A_C07M20_counts_real_s0).
#
# WHAT ONE TASK DOES.
#   python3 $KIT/tools/t125/hmap/score.py chain <manifest> <covariates.tsv> <cache_dir> <ctl_cache> \
#       <blacklist> <eta_runs_dir>/<f_run> <out_root>/chain --h <spec> [--h <spec> ...] --workers 4 \
#       --mix-merge 256
#           -> chain/<f_run>/<h_name>/scores.json per h, chain/<f_run>/CHAIN_DONE
# --mix-merge 256 (PI 2026-10-16): the own-noise h's mixture CRPS over at most 256 moment-matched
# components per bin (each such record says crps_mix_merged); push-only h stay exact.
# The h list is `python3 tools/t125/hmap/tasks.py chain-list <f_run> --fnb-seeds N` (50 at N = 1:
# the 2 formulas, the 36 h trained on the true X', this f's 12 h trained on its NB), each name
# turned into a spec: h_formula_bin / h_formula_frag -> formula:bin / formula:frag (the learned
# h_formula_affine_* and h_formula_mlp_* are run dirs, not formulas); *_true_s<seed> -> <out_root>/runs_true/<name>;
# *_<f_run>_s<seed> -> <out_root>/runs_fnb/<name>. A task whose CHAIN_DONE lists every h of the
# task exits 0 without work. A task whose f run or any learned h run dir has no TRAIN_DONE exits
# 3, so an unfinished train task shows up as a failed chain task instead of a partial chain.
#
# THE H SET (9th argument, PI 2026-10-16). "all" (default): the 50 h above, for the 16 real f
# with one g across tracks (the repair reading; their fnb h must exist). "true": only the 2
# formulas and the 36 true-X' h (38), no fnb h needed: the per-track f, if time allows. A later
# "all" task on an f that a "true" task chained scores only its 12 fnb h (h with a scores.json
# are not scored again) and rewrites CHAIN_DONE with all 50 names.
#
# THE TASK TABLE. Index -> f run comes from `python3 tools/t125/hmap/tasks.py fnb <manifest>`:
# the f runs in the order of their first row (that table is built from eta.pairs.f_runs_seed0).
# The first task with a venv writes "index f_run" to <out_root>/tasks_chain.tsv; a later task
# reads that copy BEFORE its venv only to exit early, then checks it against a fresh table (a
# mismatch exits 2). The copy is created once and never replaced. An <out_root> or <eta_runs_dir>
# under a t118 directory is refused (exit 2): t118 runs carry the same run names as t125's f runs.
#
# NO --export. Every input is a POSITIONAL argument; the task index is $SLURM_ARRAY_TASK_ID.
# Every output is under <out_root>/chain:
#   OUT_H=/project/def-maxwl/mforooz/t125/h
#   CTL=/project/def-maxwl/mforooz/t125/ctl_cache     (h_ctl_cache.sh builds it)
#   CACHE=/project/def-maxwl/mforooz/t118/ladder/cache (t118's count and pval cache, read-only here)
#   ETA_RUNS=/project/def-maxwl/mforooz/t125/eta/runs  (the f runs, read-only here)
#
# THE VENV is the pinned recipe of slurm/t118/row2_train.sh (python/3.10.13, requirements-fir.txt,
# then requirements-pypi.txt from $KIT/wheels), built per task in $SLURM_TMPDIR; PYTHONPATH is
# "$KIT/src" and the t125 CLIs put tools/t118 and tools/t125 on sys.path themselves.
#
# RESOURCES. One 10 GB MIG slice of an H100, 4 cores, 40000M host memory, 6 h. Nodes c128, c166,
# c537 and c371 are excluded. Peak memory from `sacct -o MaxRSS` (no /usr/bin/time on compute
# nodes). Since t125-P2h the slice does the scoring itself (score.py's fast path: every
# per-(bin, k) step in torch on the device, the same numbers as the CPU scorer up to summation
# order); the jobs run in one process on CUDA, so --workers 4 is unused there. The cost grows with
# f's NB support: a row-2 f (support about 10 counts) fits in the 6 h; a row-1 f (about 220-300
# counts on chr19 + chr21) does not, above all for the rung-5 own-noise h, whose mixture CRPS is
# O(support^2) per bin (measured in t125-P2h).
#
# DRY_RUN=1 prints the chain command line (nothing when CHAIN_DONE lists every h; a note on stderr for
# each missing TRAIN_DONE, where a real task would exit 3) and exits 0 before any module load. The
# tables are resolved as in h_train.sh: ${PYTHON:-python3}, $KIT or else this script's checkout,
# <products_dir>/MANIFEST.tsv or else tests/fixtures/t112_meta/MANIFEST.tsv.
#
# Usage, from the Nibi login node, for the f's whose 12 fnb h AND the 36 true h all have
# TRAIN_DONE (check the files; never a dependency on an array that may have finished). Submit in
# batches as the fits finish; every batch's index list and job id goes into plan/T125_STATUS.md:
#   OUT_H=/project/def-maxwl/mforooz/t125/h; mkdir -p $OUT_H/logs
#   CTL=/project/def-maxwl/mforooz/t125/ctl_cache; CACHE=/project/def-maxwl/mforooz/t118/ladder/cache
#   ETA_RUNS=/project/def-maxwl/mforooz/t125/eta/runs
#   sbatch --test-only --array=0-63%40 --exclude=c128,c166,c537,c371 --output=$OUT_H/logs/%x_%A_%a.out \
#       $KIT/slurm/t125/h_chain.sh $KIT $PRODUCTS $CACHE $CTL $BLACKLIST $ETA_RUNS $OUT_H 1
#   sbatch --parsable --array=$READY_INDICES%40 --exclude=c128,c166,c537,c371 \
#       --output=$OUT_H/logs/%x_%A_%a.out \
#       $KIT/slurm/t125/h_chain.sh $KIT $PRODUCTS $CACHE $CTL $BLACKLIST $ETA_RUNS $OUT_H 1
#   # a per-track f with the formulas and the 36 true-X' h only:
#   sbatch --parsable --array=$INDICES%40 --exclude=c128,c166,c537,c371 \
#       --output=$OUT_H/logs/%x_%A_%a.out \
#       $KIT/slurm/t125/h_chain.sh $KIT $PRODUCTS $CACHE $CTL $BLACKLIST $ETA_RUNS $OUT_H 1 true
# %40 is safe because each task builds its own venv in $SLURM_TMPDIR.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t125_hchain
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --exclude=c128,c166,c537,c371
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40000M

set -euo pipefail

BLACKLIST_SHA256_PREFIX=31c69342
USAGE="usage: h_chain.sh <kit_dir> <products_dir> <cache_dir> <ctl_cache> <blacklist.bed> <eta_runs_dir> <out_root> [fnb_seeds 1|3] [h_set all|true]"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
CACHE="${3:?$USAGE}"
CTL="${4:?$USAGE}"
BLACKLIST="${5:?$USAGE}"
ETA_RUNS="${6:?$USAGE}"
OUT_ROOT="${7:?$USAGE}"
FNB_SEEDS="${8:-1}"
H_SET="${9:-all}"
IDX="${SLURM_ARRAY_TASK_ID:?array task only}"
[[ "$IDX" =~ ^[0-9]+$ ]] || { echo "SLURM_ARRAY_TASK_ID '$IDX' is not an index" >&2; exit 2; }
case "$FNB_SEEDS" in
  1|3) ;;
  *) echo "fnb_seeds '$FNB_SEEDS' is not 1 or 3" >&2; exit 2 ;;
esac
case "$H_SET" in
  all) N_EXPECT=$((38 + 12 * FNB_SEEDS)) ;;
  true) N_EXPECT=38 ;;
  *) echo "h_set '$H_SET' is not all or true; $USAGE" >&2; exit 2 ;;
esac
MIX_MERGE=256
case "$OUT_ROOT" in
  */t118|*/t118/*) echo "out root $OUT_ROOT is under a t118 directory; t125 writes only under t125/" >&2; exit 2 ;;
esac
case "$ETA_RUNS" in
  */t118|*/t118/*) echo "eta runs $ETA_RUNS are under a t118 directory; the chain scores t125's f runs" >&2; exit 2 ;;
esac
PY_MODULES="python/3.10.13"
MANIFEST="$PRODUCTS/MANIFEST.tsv"
COVARIATES="$KIT/tools/t118/covariates.tsv"
RUNS_TRUE="$OUT_ROOT/runs_true"
RUNS_FNB="$OUT_ROOT/runs_fnb"
CHAIN="$OUT_ROOT/chain"
TASKS_TSV="$OUT_ROOT/tasks_chain.tsv"

# "index f_run" for every f run of a tasks.py fnb table on stdin, in the order of its first row.
f_runs_from() {
  awk -F'\t' 'NR == 1 { for (c = 1; c <= NF; c++) h[$c] = c; print "index\tf_run"; next }
              { s = $h["source"]; sub(/^fnb:/, "", s); if (!(s in seen)) { seen[s] = 1; print n++ "\t" s } }'
}
# The f run of task $IDX, from an "index f_run" table on stdin.
f_run_from() { awk -F'\t' -v i="$IDX" 'NR > 1 && $1 == i { print $2 }'; }
# The h_spec of one chain-list name for f run $F_RUN; empty for a name of no known kind.
spec_of() {
  case "$1" in
    h_formula_bin|h_formula_frag) echo "formula:${1#h_formula_}" ;;   # not h_formula_affine_*/mlp_*
    *_true_s[0-9]*) echo "$RUNS_TRUE/$1" ;;
    *_"$F_RUN"_s[0-9]*) echo "$RUNS_FNB/$1" ;;
    *) echo "" ;;
  esac
}
# Fill SPECS (the --h arguments) and MISSING (learned run dirs without TRAIN_DONE) from the
# chain-list names on stdin.
build_specs() {
  SPECS=(); MISSING=()
  local name spec
  while IFS= read -r name; do
    [ -n "$name" ] || continue
    spec=$(spec_of "$name")
    [ -n "$spec" ] || { echo "chain-list name $name is no formula, true or $F_RUN h" >&2; exit 2; }
    SPECS+=(--h "$spec")
    case "$spec" in
      formula:*) ;;
      *) [ -f "$spec/TRAIN_DONE" ] || MISSING+=("$spec") ;;
    esac
  done
}

# The chain-list names on stdin, kept for $H_SET: "true" drops this f's fnb h.
names_for_set() {
  local name
  while IFS= read -r name; do
    [ -n "$name" ] || continue
    case "$H_SET:$name" in
      true:h_formula_bin|true:h_formula_frag|true:*_true_s[0-9]*|all:*) echo "$name" ;;
    esac
  done
}
# 0 when $CHAIN/$F_RUN/CHAIN_DONE lists every name on stdin.
chain_done_covers() {
  local done="$CHAIN/$F_RUN/CHAIN_DONE" name
  [ -f "$done" ] || return 1
  while IFS= read -r name; do
    [ -n "$name" ] || continue
    grep -qxF "$name" "$done" || return 1
  done
  return 0
}
# Before the venv (no h list yet): CHAIN_DONE has at least as many names as this h set needs.
chain_done_count_ok() {
  [ -f "$CHAIN/$F_RUN/CHAIN_DONE" ] && [ "$(grep -c . "$CHAIN/$F_RUN/CHAIN_DONE")" -ge "$N_EXPECT" ]
}

if [ "${DRY_RUN:-0}" = 1 ]; then
  TOOLS="$KIT"
  [ -f "$TOOLS/tools/t125/hmap/tasks.py" ] || TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  TABLE_MANIFEST="$MANIFEST"
  [ -f "$TABLE_MANIFEST" ] || TABLE_MANIFEST="$TOOLS/tests/fixtures/t112_meta/MANIFEST.tsv"
  PY="${PYTHON:-python3}"
  TPP="$TOOLS/src${PYTHONPATH:+:$PYTHONPATH}"
  F_RUN=$(PYTHONPATH="$TPP" "$PY" "$TOOLS/tools/t125/hmap/tasks.py" fnb "$TABLE_MANIFEST" \
          | f_runs_from | f_run_from)
  [ -n "$F_RUN" ] || { echo "no task $IDX in the f-run table" >&2; exit 2; }
  NAMES=$(PYTHONPATH="$TPP" "$PY" "$TOOLS/tools/t125/hmap/tasks.py" chain-list "$F_RUN" \
          --fnb-seeds "$FNB_SEEDS" | names_for_set)
  if chain_done_covers <<< "$NAMES"; then echo "skip $F_RUN: CHAIN_DONE lists every h" >&2; exit 0; fi
  build_specs <<< "$NAMES"
  [ -f "$ETA_RUNS/$F_RUN/TRAIN_DONE" ] \
    || echo "note: no $ETA_RUNS/$F_RUN/TRAIN_DONE; a real task would exit 3" >&2
  [ "${#MISSING[@]}" = 0 ] \
    || echo "note: ${#MISSING[@]} h run dirs lack TRAIN_DONE; a real task would exit 3" >&2
  echo "python3 $KIT/tools/t125/hmap/score.py chain $MANIFEST $COVARIATES $CACHE $CTL $BLACKLIST $ETA_RUNS/$F_RUN $CHAIN ${SPECS[*]} --workers 4 --mix-merge $MIX_MERGE"
  exit 0
fi

[ -f "$KIT/tools/t125/hmap/score.py" ] || { echo "no tools/t125/hmap/score.py under $KIT" >&2; exit 2; }
[ -f "$KIT/tools/t125/hmap/tasks.py" ] || { echo "no tools/t125/hmap/tasks.py under $KIT" >&2; exit 2; }
[ -f "$COVARIATES" ] || { echo "no $COVARIATES" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "no $MANIFEST" >&2; exit 2; }
[ -d "$CACHE" ] || { echo "no cache dir $CACHE (t118 ladder_cache.sh builds it)" >&2; exit 2; }
[ -d "$CTL" ] || { echo "no control cache $CTL (h_ctl_cache.sh builds it)" >&2; exit 2; }
[ -d "$ETA_RUNS" ] || { echo "no eta runs dir $ETA_RUNS" >&2; exit 2; }
[ -s "$BLACKLIST" ] || { echo "blacklist $BLACKLIST missing or empty" >&2; exit 2; }
BL_SHA=$(sha256sum "$BLACKLIST" | cut -d' ' -f1)
case "$BL_SHA" in
  "$BLACKLIST_SHA256_PREFIX"*) ;;
  *) echo "blacklist sha256 $BL_SHA does not start $BLACKLIST_SHA256_PREFIX" >&2; exit 2 ;;
esac
KIT=$(cd "$KIT" && pwd)
COVARIATES="$KIT/tools/t118/covariates.tsv"
mkdir -p "$CHAIN"

echo "=== t125 h_chain idx=$IDX job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX host=$(hostname) $(date -u)"
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || git -C "$KIT" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "blacklist sha256: $BL_SHA"

# Early exits before the venv, only from a table an earlier task already wrote.
if [ -f "$TASKS_TSV" ]; then
  F_RUN=$(f_run_from < "$TASKS_TSV")
  if [ -n "$F_RUN" ]; then
    if chain_done_count_ok; then echo "skip $F_RUN: CHAIN_DONE lists $N_EXPECT h or more"; exit 0; fi
    [ -f "$ETA_RUNS/$F_RUN/TRAIN_DONE" ] || { echo "no $ETA_RUNS/$F_RUN/TRAIN_DONE: f run $F_RUN did not finish" >&2; exit 3; }
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
if [ -f "$KIT/GIT_SHA" ]; then export T125_GIT_SHA=$(cat "$KIT/GIT_SHA"); fi
python3 -c "import candi.metrics, candi.bench.distributional, candi.store.genome; print('venv ok', candi.__file__)"
# the library must come from this kit, not from some other install (tests/test_slurm_kit_pin.py)
case "$(python3 -c 'import candi; print(candi.__file__)')" in
  "$KIT"/src/*) ;;
  *) echo "candi does not import from $KIT/src" >&2; exit 3 ;;
esac

# The authoritative f-run table, from tasks.py; a stale shared copy is an error, not a guess.
TMP_TABLE="$TASKS_TSV.tmp.${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX"
python3 "$KIT/tools/t125/hmap/tasks.py" fnb "$MANIFEST" | f_runs_from > "$TMP_TABLE"
# cmp exit 1 = the tables differ (an error); exit 2 = a read problem (not fatal).
F_RUN=$(f_run_from < "$TMP_TABLE")
if [ -f "$TASKS_TSV" ]; then
  CMP_RC=0; cmp -s "$TMP_TABLE" "$TASKS_TSV" || CMP_RC=$?
  if [ "$CMP_RC" = 1 ]; then
    echo "$TASKS_TSV differs from this kit's tasks.py f-run table; remove it or fix the kit" >&2
    rm -f "$TMP_TABLE"; exit 2
  fi
else
  mv -n "$TMP_TABLE" "$TASKS_TSV" 2>/dev/null || true
fi
rm -f "$TMP_TABLE"
[ -n "$F_RUN" ] || { echo "no task $IDX in the f-run table" >&2; exit 2; }
echo "f run: $F_RUN"
[ -f "$ETA_RUNS/$F_RUN/TRAIN_DONE" ] || { echo "no $ETA_RUNS/$F_RUN/TRAIN_DONE: f run $F_RUN did not finish" >&2; exit 3; }

NAMES=$(python3 "$KIT/tools/t125/hmap/tasks.py" chain-list "$F_RUN" --fnb-seeds "$FNB_SEEDS" | names_for_set)
if chain_done_covers <<< "$NAMES"; then echo "skip $F_RUN: CHAIN_DONE lists every h"; exit 0; fi
build_specs <<< "$NAMES"
if [ "${#MISSING[@]}" != 0 ]; then
  echo "${#MISSING[@]} h run dirs lack TRAIN_DONE; train them first:" >&2
  printf '  %s\n' "${MISSING[@]}" >&2
  exit 3
fi
echo "h set $H_SET: $((${#SPECS[@]} / 2)) h"

python3 "$KIT/tools/t125/hmap/score.py" chain "$MANIFEST" "$COVARIATES" "$CACHE" "$CTL" \
    "$BLACKLIST" "$ETA_RUNS/$F_RUN" "$CHAIN" "${SPECS[@]}" --workers 4 --mix-merge "$MIX_MERGE"

echo "=== done $F_RUN $(date -u)"
