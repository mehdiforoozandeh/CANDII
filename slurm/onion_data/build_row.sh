#!/bin/bash
# t180 (chunk KF): one build-list row -> verified measurement directories. The bulk build's driver.
#
# Submit one priority (run on a login node; prints the job ids):
#   bash <kit>/slurm/onion_data/build_row.sh submit <kit> <build list> <priority> \
#       [build concurrency=50] [stage concurrency=2]
#   -> 1. a stage array (stage_fastq.sh, 2 CPU): the row's FASTQs or native BAMs, md5-checked
#      2. a build array (this script, 16 CPU), task i after stage task i (aftercorr)
#      3. for the rows whose products include signal: a signal pass of this script after the
#         whole build array (afterany); a ChIP row whose control was not built yet exits 75 in
#         step 2 and finishes here (every other step skips what exists)
#
# One row (an array task, or by hand):
#   sbatch [--array=<rows>%N] <kit>/slurm/onion_data/build_row.sh <kit> <build list> [row]
#   1. align: chip/atac rows -> fastq_uniform.py run; dnase rows -> fastq_dnase.py run; native
#      rows (pipeline = native-redacted-bam) have nothing to align
#   2. views: align_views.py run --row (base, halves, br<i>, depth arms, nodedup, mapq0, or the
#      row's FASTQ-level arm; signal only when the row asks), then align_views.py verify on the
#      experiment's measurements
# A cgroup sampler logs the job's peak anonymous memory and page cache every 30 s ([mem] lines):
# the job's MaxRSS includes page cache (KF: 60 of 67 GB was file cache, 0 OOM events).
# Env: ONION_DROP_BAMS=1 deletes a keep_bam = 0 row's BAMs once its views exist;
# ONION_VIEWS_ARGS adds align_views.py run options (e.g. --name-tokens --check-refilter);
# ONION_WORK, ONION_OUT, ONION_FASTQ_ROOT move the three roots (tests only).
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
  KIT="${2:?usage: build_row.sh submit <kit> <build list> <priority> [conc] [stage conc]}"
  BL="${3:?build list}"; PRI="${4:?priority}"; CONC="${5:-50}"; SCONC="${6:-2}"
  ROWS=$(python3 "$KIT/tools/onion_data/build_list.py" rows --build-list "$BL" --priority "$PRI") || exit 2
  SIG=$(awk -F'\t' -v p="$PRI" 'NR==1{for(i=1;i<=NF;i++)c[$i]=i; next}
        $c["priority"]==p && $c["expected_measurements"] ~ /signal/ {printf "%s%s", s, $c["row"]; s=","}' "$BL")
  mkdir -p /project/def-maxwl/mforooz/ONION/logs
  SID=$(sbatch --parsable --array="${ROWS}%${SCONC}" "$KIT/slurm/onion_data/stage_fastq.sh" "$KIT" "$BL") || exit 3
  BID=$(sbatch --parsable --array="${ROWS}%${CONC}" --dependency="aftercorr:${SID}" \
        --kill-on-invalid-dep=yes "$KIT/slurm/onion_data/build_row.sh" "$KIT" "$BL") || exit 3
  echo "priority $PRI rows $ROWS"
  echo "stage array $SID (%$SCONC)  build array $BID (%$CONC)"
  if [ -n "$SIG" ]; then
    GID=$(sbatch --parsable --array="${SIG}%${CONC}" --dependency="afterany:${BID}" \
          --job-name=onion_signal "$KIT/slurm/onion_data/build_row.sh" "$KIT" "$BL") || exit 3
    echo "signal pass $GID rows $SIG"
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
DROP=(); [ "${ONION_DROP_BAMS:-0}" = 1 ] && DROP=(--drop-bams)
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
