#!/bin/bash
# Close-out item A4: tools/onion_data/qc_fill.py `run` over one CHUNK of measurements.tsv rows per
# array task, with every record kept off /scratch. Line k+1 of <rows file> (k = SLURM_ARRAY_TASK_ID)
# is a row spec ("3-8,11"). The records go to $SLURM_TMPDIR, and the task ends by writing ONE tar
#   <dest dir>/rec_<pass>_<k>.tar      (pass = whole | pr1)
# so a pass of N chunks adds N tars to /project, not one file per row.
#
#   sbatch --array=0-<N-1>%<p> --output=<dest>/logs/<pass>_%A_%a.out \
#       <kit>/slurm/onion_data/qc_fill_chunks.sh <kit dir> <tables dir> <rows file> <dest dir> <pass>
#
# Submitted by tools/minicandi/configs/ablations/qc_on_input/submit_qc_on_input.sh (MODE=fill).
# Environment and the read-set checks are those of slurm/onion_data/qc_fill.sh (kit venv, apptainer
# 1.3.5, every intermediate in $SLURM_TMPDIR). Header: 4 CPUs and 16000M as qc_fill.sh (peak RSS
# 11.7 GB measured); --time 06:00:00 = 20 rows x the longest measured row (588 s, Nibi job 27235942)
# = 3.3 h, with margin.
#SBATCH --account=def-maxwl
#SBATCH --job-name=a4_qc_fill
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --gres=none

set -uo pipefail
USAGE="usage: qc_fill_chunks.sh <kit dir> <tables dir> <rows file> <dest dir> <whole|pr1>"
KIT="${1:?$USAGE}"; TABLES="${2:?$USAGE}"; ROWS_FILE="${3:?$USAGE}"; DEST="${4:?$USAGE}"
PASS="${5:?$USAGE}"
K="${SLURM_ARRAY_TASK_ID:?qc_fill_chunks.sh runs as an array task}"
case "$PASS" in whole) HALF=() ;; pr1) HALF=(--half pr1) ;; *) echo "$USAGE" >&2; exit 2 ;; esac
KIT=$(cd "$KIT" && pwd)
SPEC=$(sed -n "$((K + 1))p" "$ROWS_FILE")
[ -n "$SPEC" ] || { echo "no line $((K + 1)) in $ROWS_FILE" >&2; exit 2; }
TAR="$DEST/rec_${PASS}_${K}.tar"
[ ! -e "$TAR" ] || { echo "$TAR exists: this chunk is done (never overwritten)"; exit 0; }
echo "=== A4 qc_fill pass=$PASS chunk=$K rows=$SPEC job ${SLURM_ARRAY_JOB_ID:-none}_$K host=$(hostname) $(date -u)"
cat "$KIT/.kit_stamp" 2>/dev/null || true
df -h "$SLURM_TMPDIR" | tail -1

set +u; module load apptainer/1.3.5 python/3.10.13; set -u
{ virtualenv --no-download "$SLURM_TMPDIR/venv" &&
  source "$SLURM_TMPDIR/venv/bin/activate" &&
  pip install -q --no-index --upgrade pip &&
  pip install -q --no-index -r "$KIT/requirements-fir.txt" &&
  pip install -q --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"; } ||
  { echo "qc_fill_chunks.sh: venv setup failed"; echo "[onion-run] exit=3"; exit 3; }
source "$SLURM_TMPDIR/venv/bin/activate"
cd "$KIT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
source "$KIT/slurm/_kit_pin.sh"

REC="$SLURM_TMPDIR/rec"
mkdir -p "$REC"
python tools/onion_data/qc_fill.py run --tables "$TABLES" --rows "$SPEC" ${HALF[@]+"${HALF[@]}"} \
    --work "$SLURM_TMPDIR/qc" --out "$REC" --nth "${SLURM_CPUS_PER_TASK:-4}" --sort-mem 6G
rc=$?
n=$(ls "$REC"/*.json 2>/dev/null | wc -l)
tar -C "$REC" -cf "$TAR.partial" . && mv "$TAR.partial" "$TAR" ||
  { echo "tar to $TAR failed"; echo "[onion-run] exit=4"; exit 4; }
echo "=== $n records -> $TAR (qc_fill exit $rc: 1 means a refused or failed row, recorded) $(date -u)"
echo "[onion-run] exit=$rc"
exit $rc
