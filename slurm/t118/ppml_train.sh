#!/bin/bash
# t127, PPML vs NB: one array task = one run of `pairs.tasks_ppml()` (126 rows: rungs A, A0, D x
# losses nb, ppml x the 7 tracks x seeds 0-2; counts, real covariates, per-track g). The task trains,
# scores its trained pairs (score.py trained --extras), then runs the law pass on the never-trained
# arm->arm pairs (score.py law --extras), in that order, in the same job. Each step is skipped when
# its marker exists (TRAIN_DONE, SCORE_DONE, LAW_DONE), so a resubmission redoes only what is missing.
#
# Copied from ladder_train.sh + ladder_law.sh (row 1) with these differences:
#   - the table is `pairs.py tasks --ppml` and train.py runs `train-index --ppml`;
#   - score.py gets --extras (report-only keys: mu_sum_over_y_sum, cover50, cover90);
#   - the law pass runs in the same task, not as an aftercorr-chained array (row 1's aftercorr
#     chain never released a task while the train array ran);
#   - CPU by default (16 cores, OMP/MKL threads = cores, as row2_train_cpu.sh), --device auto, so the
#     same script trains on a GPU when submitted with --gres; 64 GB, 8 h.
#   - extra arguments after <out_dir> go to train.py (the smoke passes --max-steps / --eval-every).
# Training settings are train.py's DEFAULTS, the same as row 1.
#
# NO --export: every input is positional; the index is $SLURM_ARRAY_TASK_ID.
# Usage, from the Nibi login node:
#   OUT=/project/def-maxwl/mforooz/t118/ppml_vs_nb; mkdir -p $OUT/logs
#   CACHE=/project/def-maxwl/mforooz/t118/ladder/cache   (row 1's cache, read-only here)
#   PRODUCTS=/project/def-maxwl/mforooz/t112_cf/products
#   sbatch --parsable --array=0-125%40 --output=$OUT/logs/%x_%A_%a.out \
#       $KIT/slurm/t118/ppml_train.sh $KIT $PRODUCTS $CACHE $OUT
#SBATCH --account=def-maxwl
#SBATCH --job-name=t127_ppml
#SBATCH --exclude=c128,c166,c537
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M

set -euo pipefail

BLACKLIST=/project/def-maxwl/mforooz/EIC_REPRO/002/scripts/hg38_blacklist_v2.bed
BLACKLIST_SHA256_PREFIX=31c69342
USAGE="usage: ppml_train.sh <kit_dir> <products_dir> <cache_dir> <out_dir> [train.py args...]"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
CACHE="${3:?$USAGE}"
OUT="${4:?$USAGE}"
shift 4
TRAIN_ARGS=("$@")
IDX="${SLURM_ARRAY_TASK_ID:?array task only}"
[[ "$IDX" =~ ^[0-9]+$ ]] || { echo "SLURM_ARRAY_TASK_ID '$IDX' is not an index" >&2; exit 2; }
PY_MODULES="python/3.10.13"
MANIFEST="$PRODUCTS/MANIFEST.tsv"
RUNS="$OUT/runs"
TASKS_TSV="$OUT/tasks.tsv"
WORKERS="${SLURM_CPUS_PER_TASK:-16}"

run_name_from() {
  awk -F'\t' -v i="$IDX" 'NR == 1 { for (c = 1; c <= NF; c++) h[$c] = c; next }
                          $h["index"] == i { print $h["run_name"] }'
}

[ -f "$KIT/tools/t118/ladder/train.py" ] || { echo "no tools/t118/ladder/train.py under $KIT" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "no $MANIFEST" >&2; exit 2; }
[ -d "$CACHE" ] || { echo "no cache dir $CACHE" >&2; exit 2; }
[ -s "$BLACKLIST" ] || { echo "blacklist $BLACKLIST missing or empty" >&2; exit 2; }
BL_SHA=$(sha256sum "$BLACKLIST" | cut -d' ' -f1)
case "$BL_SHA" in
  "$BLACKLIST_SHA256_PREFIX"*) ;;
  *) echo "blacklist sha256 $BL_SHA does not start $BLACKLIST_SHA256_PREFIX" >&2; exit 2 ;;
esac
KIT=$(cd "$KIT" && pwd)
COVARIATES="$KIT/tools/t118/covariates.tsv"
[ -f "$COVARIATES" ] || { echo "no $COVARIATES" >&2; exit 2; }
mkdir -p "$RUNS"

echo "=== t127 ppml_train idx=$IDX job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX host=$(hostname) $(date -u)"
echo "kit git sha: $(cat "$KIT/GIT_SHA" 2>/dev/null || echo unknown)"
echo "train args: ${TRAIN_ARGS[*]:-none}"

if [ -f "$TASKS_TSV" ]; then
  RUN=$(run_name_from < "$TASKS_TSV")
  if [ -n "$RUN" ] && [ -f "$RUNS/$RUN/LAW_DONE" ]; then echo "skip $RUN: LAW_DONE exists"; exit 0; fi
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
export OMP_NUM_THREADS="$WORKERS" MKL_NUM_THREADS="$WORKERS"
if [ -f "$KIT/GIT_SHA" ]; then export T118_GIT_SHA=$(cat "$KIT/GIT_SHA"); fi
case "$(python3 -c 'import candi; print(candi.__file__)')" in
  "$KIT"/src/*) ;;
  *) echo "candi does not import from $KIT/src" >&2; exit 3 ;;
esac

TMP_TABLE="$TASKS_TSV.tmp.${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX"
python3 "$KIT/tools/t118/ladder/pairs.py" tasks "$MANIFEST" --ppml > "$TMP_TABLE"
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
[ -n "$RUN" ] || { echo "no task $IDX in the ppml table" >&2; exit 2; }
echo "run: $RUN"

python3 "$KIT/tools/t118/ladder/train.py" train-index "$MANIFEST" "$COVARIATES" "$CACHE" "$RUNS" \
    "$IDX" --ppml --device auto "${TRAIN_ARGS[@]}"
echo "=== trained $(date -u)"
if [ ! -f "$RUNS/$RUN/SCORE_DONE" ]; then
  python3 "$KIT/tools/t118/ladder/score.py" trained "$MANIFEST" "$COVARIATES" "$CACHE" "$BLACKLIST" \
      "$RUNS/$RUN" --workers "$WORKERS" --extras
fi
echo "=== scored $(date -u)"
if [ ! -f "$RUNS/$RUN/LAW_DONE" ]; then
  python3 "$KIT/tools/t118/ladder/score.py" law "$MANIFEST" "$COVARIATES" "$CACHE" "$BLACKLIST" \
      "$RUNS/$RUN" --workers "$WORKERS" --extras
fi
echo "=== done $RUN $(date -u)"
