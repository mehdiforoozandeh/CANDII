#!/bin/bash
# t180 (chunk KF): one build-list row -> verified measurement directories. The bulk build's driver.
#
# Submit one priority (run on a login node; prints every skipped and submitted row, and job ids):
#   bash <kit>/slurm/onion_data/build_row.sh submit <kit> <build list> <priority|all> \
#       [build concurrency=50] [stage concurrency=2] [row spec, e.g. 1-20,33]
#   Safe to run twice: a row is submitted only when some expected measurement has no
#   provenance.json, its run dir holds no NOT_APPLICABLE, and no onion_* array task of that row is
#   queued or running (`build_list.py pending`). Rows are grouped by time limit: a row whose
#   projected wall (catalog reads x the KF smoke rates) exceeds 18 h gets 3 or 7 days.
#   Per time-limit group:
#   -> 1. a stage array (stage_fastq.sh, 2 CPU): the row's FASTQs or native BAMs, md5-checked
#      2. a build array (this script, 16 CPU), task i after stage task i (aftercorr)
#   then 3. for the submitted rows whose products include signal: a signal pass of this script
#      after every build array (afterany); a ChIP row whose control is not completely built exits
#      75 in step 2 and finishes here; a row already built exits 0 at once
#
# One row (an array task, or by hand):
#   sbatch [--array=<rows>%N] <kit>/slurm/onion_data/build_row.sh <kit> <build list> [row]
#   0. a row whose expected measurements all exist (or NOT_APPLICABLE) exits 0 at once
#   1. align: chip/atac rows -> fastq_uniform.py run; dnase rows -> fastq_dnase.py run; native
#      rows (pipeline = native-redacted-bam) have nothing to align
#   2. views: align_views.py run --row (base, halves, br<i>, depth arms, nodedup, mapq0, or the
#      row's FASTQ-level arm; signal only when the row asks), then align_views.py verify on the
#      experiment's measurements
# A cgroup sampler logs the job's peak anonymous memory and page cache every 30 s ([mem] lines):
# the job's MaxRSS includes page cache (KF: 60 of 67 GB was file cache, 0 OOM events).
# Env: ONION_DROP_BAMS (default 1) deletes a keep_bam = 0 row's run-dir BAMs once every row that
# reads that run dir has its measurements (align_views.bam_readers_pending; set 0 to keep them);
# ONION_VIEWS_ARGS adds align_views.py run options (e.g. --name-tokens --check-refilter);
# ONION_WORK, ONION_OUT, ONION_FASTQ_ROOT move the three roots (tests only).
# Submit-only overrides (D6x), for a resubmission of rows that ran out of time or memory:
#   ONION_TIME=3-00:00:00 replaces the projected time limit, ONION_MEM=128000M the header --mem,
#   on every build array and the signal pass of that submit, e.g.
#   ONION_TIME=3-00:00:00 bash build_row.sh submit <kit> <build list> 1 50 2 56
#   ONION_SBATCH_TEST=1 prints the sbatch lines instead of submitting (tests only).
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_build
#SBATCH --output=/project/def-maxwl/mforooz/ONION/logs/build_%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --gres=none

set -uo pipefail
EIC=/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline   # read only: Caper venv, images, WDLs
WORK="${ONION_WORK:-/scratch/mforooz/onion/work}"
OUT="${ONION_OUT:-/scratch/mforooz/onion/measurements}"
FQ="${ONION_FASTQ_ROOT:-/scratch/mforooz/onion/fastq}"

if [ "${1:-}" = submit ]; then
  KIT="${2:?usage: build_row.sh submit <kit> <build list> <priority|all> [conc] [stage conc] [rows]}"
  BL="${3:?build list}"; PRI="${4:?priority (a number, or all)}"; CONC="${5:-50}"; SCONC="${6:-2}"
  SPEC="${7:-}"
  KIT=$(cd "$KIT" && pwd); BL=$(cd "$(dirname "$BL")" && pwd)/$(basename "$BL")
  FILES=/project/def-maxwl/mforooz/ONION/catalog/encode/files.tsv   # read counts: time limits
  RES=(); [ -n "${ONION_MEM:-}" ] && RES+=(--mem="$ONION_MEM")
  SB=(sbatch); [ "${ONION_SBATCH_TEST:-0}" = 1 ] && SB=(echo sbatch)
  if [ "${ONION_SBATCH_TEST:-0}" != 1 ]; then
    python3 "$KIT/tools/onion_data/fastq_uniform.py" check-jars || exit 3
    ACTIVE=$(squeue -u "$USER" -h -r -n onion_build,onion_signal,onion_stage -o '%K' 2>/dev/null \
             | grep -E '^[0-9]+$' | sort -un | paste -sd, -)
  else
    ACTIVE=""
  fi
  args=(--build-list "$BL" --priority "$PRI" --out "$OUT" --work "$WORK")
  [ -n "$SPEC" ] && args+=(--rows "$SPEC")
  [ -n "$ACTIVE" ] && args+=(--active "$ACTIVE")
  [ -f "$FILES" ] && args+=(--files "$FILES")
  PLAN=$(python3 "$KIT/tools/onion_data/build_list.py" pending "${args[@]}") || exit 2
  [ "${ONION_SBATCH_TEST:-0}" = 1 ] || mkdir -p /project/def-maxwl/mforooz/ONION/logs
  [ -n "$PLAN" ] || { echo "priority $PRI${SPEC:+ rows $SPEC}: nothing to submit"; exit 0; }
  BIDS=(); SIG=""
  while IFS=$'\t' read -r LIM ROWS; do
    if [ "$LIM" = signal ]; then SIG="$ROWS"; continue; fi
    LIM="${ONION_TIME:-$LIM}"
    SID=$("${SB[@]}" --parsable --array="${ROWS}%${SCONC}" "$KIT/slurm/onion_data/stage_fastq.sh" \
          "$KIT" "$BL") || exit 3
    BID=$("${SB[@]}" --parsable --time="$LIM" ${RES[@]+"${RES[@]}"} --array="${ROWS}%${CONC}" \
          --dependency="aftercorr:${SID}" \
          --kill-on-invalid-dep=yes "$KIT/slurm/onion_data/build_row.sh" "$KIT" "$BL") || exit 3
    echo "rows $ROWS (limit $LIM${ONION_MEM:+, mem $ONION_MEM}): stage array $SID (%$SCONC)  build array $BID (%$CONC)"
    BIDS+=("$BID")
  done <<< "$PLAN"
  if [ -n "$SIG" ]; then
    DEP="afterany:$(IFS=:; echo "${BIDS[*]}")"
    SIGT=(); [ -n "${ONION_TIME:-}" ] && SIGT=(--time="$ONION_TIME")
    GID=$("${SB[@]}" --parsable ${SIGT[@]+"${SIGT[@]}"} ${RES[@]+"${RES[@]}"} --array="${SIG}%${CONC}" --dependency="$DEP" \
          --job-name=onion_signal "$KIT/slurm/onion_data/build_row.sh" "$KIT" "$BL") || exit 3
    echo "signal pass $GID rows $SIG (after $DEP)"
  fi
  exit 0
fi

KIT="${1:?usage: build_row.sh <kit dir> <build list> [row]}"
BL="${2:?usage: build_row.sh <kit dir> <build list> [row]}"
ROW="${3:-${SLURM_ARRAY_TASK_ID:-}}"
[ -n "$ROW" ] || { echo "build_row.sh: no row (argument 3 or array index)"; exit 2; }
KIT=$(cd "$KIT" && pwd)
[ -n "${SLURM_TMPDIR:-}" ] || { echo "no SLURM_TMPDIR: refusing to run"; exit 3; }
field() { awk -F'\t' -v r="$ROW" -v k="$1" 'NR==1{for(i=1;i<=NF;i++)c[$i]=i; next} $c["row"]==r{print $c[k]}' "$BL"; }
EXP=$(field experiment); PIPE=$(field pipeline); CLS=$(field assay_class)
[ -n "$EXP" ] || { echo "build_row.sh: row $ROW not in $BL"; exit 2; }
# 0. built already (a signal-pass task whose build task succeeded, or a rerun): nothing to do
TODO=$(python3 "$KIT/tools/onion_data/build_list.py" pending --build-list "$BL" --rows "$ROW" \
       --out "$OUT" --work "$WORK") || { echo "build_row.sh: pending check failed"; exit 2; }
if [ -z "$TODO" ]; then
  echo "=== row $ROW $EXP: every expected measurement exists (or NOT_APPLICABLE); exit 0"
  exit 0
fi
case "$PIPE" in
  native-redacted-bam) KIND=native ;;
  dnase-seq-pipeline*) KIND=dnase ;;
  *) KIND=fastq ;;
esac
echo "=== onion_build job ${SLURM_JOB_ID:-} array ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} row $ROW $EXP $CLS $KIND host $(hostname) $(date -u)"
cat "$KIT/.kit_stamp" 2>/dev/null || true
df -h "$SLURM_TMPDIR" | tail -n 1

# --- memory sampler: the job cgroup's anon (process) and file (page cache) bytes -------------
CG=$(awk -F: '{print $3}' /proc/self/cgroup | sed -E 's#/step_[^/]+.*##')
CGD="/sys/fs/cgroup$CG"
memloop() {
  local pa=0 pf=0 a f
  while sleep 30; do
    a=$(awk '$1=="anon"{print $2}' "$CGD/memory.stat" 2>/dev/null || echo 0)
    f=$(awk '$1=="file"{print $2}' "$CGD/memory.stat" 2>/dev/null || echo 0)
    [ "${a:-0}" -gt "$pa" ] && pa=$a
    [ "${f:-0}" -gt "$pf" ] && pf=$f
    echo "$pa $pf" > "$SLURM_TMPDIR/mempeak"
  done
}
memloop & MEMPID=$!
finish() {
  kill "$MEMPID" 2>/dev/null
  read -r PA PF < "$SLURM_TMPDIR/mempeak" 2>/dev/null || { PA=0; PF=0; }
  OOM=$(awk '$1=="oom_kill"{print $2}' "$CGD/memory.events" 2>/dev/null)
  echo "[mem] peak_anon_gb=$(awk -v x="$PA" 'BEGIN{printf "%.2f", x/2^30}') peak_page_cache_gb=$(awk -v x="$PF" 'BEGIN{printf "%.2f", x/2^30}') oom_kill=${OOM:-?} (30 s samples of $CGD)"
  mkdir -p "$WORK/$EXP"
  printf '{"row": "%s", "job": "%s", "peak_anon_bytes": %s, "peak_page_cache_bytes": %s, "oom_kill": "%s", "rc": %s}\n' \
    "$ROW" "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}_${SLURM_ARRAY_TASK_ID:-}" "${PA:-0}" "${PF:-0}" "${OOM:-}" "$1" \
    > "$WORK/$EXP/JOBMEM.row$ROW.json"
  echo "[onion-build] exit=$1"
  exit "$1"
}

# --- 1. align ---------------------------------------------------------------------------------
T0=$(date +%s)
if [ "$KIND" = fastq ]; then
  ( source "$EIC/runner/env.sh"
    export APPTAINER_TMPDIR="$SLURM_TMPDIR/apptainer_tmp"; mkdir -p "$APPTAINER_TMPDIR"
    export PYTHONUNBUFFERED=1
    python "$KIT/tools/onion_data/fastq_uniform.py" run --build-list "$BL" --row "$ROW" \
      --work "$WORK" --tmp "$SLURM_TMPDIR/align" --cpus "${SLURM_CPUS_PER_TASK:-16}" --fastq-root "$FQ" )
  rc=$?
elif [ "$KIND" = dnase ]; then
  ( set +u
    module load StdEnv/2023 python/3.11.5 java/17.0.6 apptainer/1.3.5
    source "$EIC/caper_venv/bin/activate"
    set -u
    export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
    export APPTAINER_CACHEDIR="$SLURM_TMPDIR/apptainer_cache" APPTAINER_TMPDIR="$SLURM_TMPDIR/apptainer_tmp"
    mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
    cd "$KIT" && python tools/onion_data/fastq_dnase.py run --build-list "$BL" --row "$ROW" \
      --work "$WORK" --tmp "$SLURM_TMPDIR/align" --fastq-root "$FQ" )
  rc=$?
else
  rc=0
fi
T1=$(date +%s)
echo "=== align step exit=$rc wall_s=$((T1 - T0)) $(date -u)"
[ "$rc" -eq 0 ] || finish "$rc"
rm -rf "$SLURM_TMPDIR/align"
# a row whose arm does not exist for the experiment (NOT_APPLICABLE, written by the align step)
# builds nothing: no views and no verify (verify fails on an experiment with no measurement yet)
UNIT=$(field unit); ARM=$(field arm)
RD="$WORK/$EXP/$ARM"; [ "$UNIT" = pooled ] || RD="$WORK/$EXP/$UNIT.$ARM"
if [ "$KIND" != native ] && [ -f "$RD/NOT_APPLICABLE" ]; then
  echo "=== row $ROW: $RD/NOT_APPLICABLE; nothing to build, verify skipped"
  finish 0
fi

# --- 2. views ---------------------------------------------------------------------------------
set +u; module load apptainer/1.3.5 python/3.10.13; set -u
{ virtualenv --no-download "$SLURM_TMPDIR/kitvenv" &&
  source "$SLURM_TMPDIR/kitvenv/bin/activate" &&
  pip install -q --no-index --upgrade pip &&
  pip install -q --no-index -r "$KIT/requirements-fir.txt" &&
  pip install -q --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt" &&
  pip install -q --no-index pyBigWig; } || { echo "kit venv setup failed"; finish 4; }
cd "$KIT" && source "$KIT/slurm/_kit_pin.sh"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
DROP=(); [ "${ONION_DROP_BAMS:-1}" = 1 ] && DROP=(--drop-bams)
python tools/onion_data/align_views.py run --build-list "$BL" --row "$ROW" --work "$WORK" \
  --out "$OUT" --fastq-root "$FQ" --tmp "$SLURM_TMPDIR/views" "${DROP[@]}" ${ONION_VIEWS_ARGS:-}
rc=$?
T2=$(date +%s)
echo "=== views step exit=$rc wall_s=$((T2 - T1)) $(date -u)"
[ "$rc" -eq 0 ] || finish "$rc"
python tools/onion_data/align_views.py verify --measurements "$OUT" --experiment "$EXP"
rc=$?
echo "=== verify exit=$rc $(date -u)"
finish "$rc"
