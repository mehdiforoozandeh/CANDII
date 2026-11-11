#!/bin/bash
# Chunk QCt: run tools/onion_data/qc_fill.py inside an ONION code kit, one measurements.tsv row
# per array task (0-based data row index = SLURM_ARRAY_TASK_ID).
#
#   sbatch --array=<rows from `qc_fill.py coverage --missing-rows`>%50 --output=<log>_%A_%a \
#       <kit>/slurm/onion_data/qc_fill.sh <kit dir> <tables dir> <out dir>
#
# Environment: the kit venv of slurm/onion_data/align_views.sh (requirements-fir.txt and the kit
# wheels, in $SLURM_TMPDIR) and apptainer 1.3.5 for the pipeline SIFs. Every intermediate (the
# concatenated tagAlign, the xcor dir, the sort temp files) stays in $SLURM_TMPDIR.
# The kit root is an argument (SLURM runs a spool copy of this file); nothing goes via --export.
# Header values are the plan's QCa request (4 CPUs) and an estimate (16000M, 4 h): xcor on a
# 15-25 M read subsample plus one sort of the pre-dedup reads. Re-measure with `seff <job id>`.
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_qc_fill
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --gres=none

set -uo pipefail
KIT="${1:?usage: qc_fill.sh <kit dir> <tables dir> <out dir>}"
TABLES="${2:?usage: qc_fill.sh <kit dir> <tables dir> <out dir>}"
OUT="${3:?usage: qc_fill.sh <kit dir> <tables dir> <out dir>}"
ROW="${SLURM_ARRAY_TASK_ID:?qc_fill.sh runs as an array task (--array=<rows>)}"
KIT=$(cd "$KIT" && pwd)
[ -f "$KIT/tools/onion_data/qc_fill.py" ] || { echo "no qc_fill.py in $KIT" >&2; exit 2; }
echo "=== onion qc_fill row $ROW job ${SLURM_ARRAY_JOB_ID:-none}_$ROW host=$(hostname) kit=$KIT $(date -u)"
cat "$KIT/.kit_stamp" 2>/dev/null || true
df -h "$SLURM_TMPDIR" | tail -1

set +u; module load apptainer/1.3.5 python/3.10.13; set -u
{ virtualenv --no-download "$SLURM_TMPDIR/venv" &&
  source "$SLURM_TMPDIR/venv/bin/activate" &&
  pip install -q --no-index --upgrade pip &&
  pip install -q --no-index -r "$KIT/requirements-fir.txt" &&
  pip install -q --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"; } ||
  { echo "qc_fill.sh: venv setup failed"; echo "[onion-run] exit=3"; exit 3; }
source "$SLURM_TMPDIR/venv/bin/activate"
cd "$KIT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
source "$KIT/slurm/_kit_pin.sh"

python tools/onion_data/qc_fill.py run --tables "$TABLES" --rows "$ROW" \
    --work "$SLURM_TMPDIR/qc" --out "$OUT" --nth "${SLURM_CPUS_PER_TASK:-4}" --sort-mem 6G
rc=$?
echo "=== done $(date -u)"
echo "[onion-run] exit=$rc"
exit $rc
