#!/bin/bash
# D7 (t168): one build-list row from FASTQ through the ENCODE-DCC pipeline in alignment-only mode.
#
#   sbatch [--array=<rows>%<n>] slurm/onion_data/fastq_uniform.sh <kit dir> <build list> [row]
#
# Row: the third argument, else $SLURM_ARRAY_TASK_ID. Caper runs on its Local backend inside this
# job; Cromwell's execution tree is under $SLURM_TMPDIR and dies with the job. Only the keepers
# reach /scratch/mforooz/onion/work/<experiment>/<arm>/ (tools/onion_data/fastq_uniform.py).
# Afterwards, if the kit holds tools/onion_data/align_views.py (chunk D5) and ONION_VIEWS is not 0,
# the experiment's counts are built from those keepers with the kit's own venv.
# Cost record: `sacct -j <job> --format=JobID,Elapsed,TotalCPU,AllocTRES%60,MaxRSS,State`.
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_fastq
#SBATCH --output=/project/def-maxwl/mforooz/ONION/logs/fastq_%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --gres=none

set -uo pipefail
KIT="${1:?usage: fastq_uniform.sh <kit dir> <build list> [row]}"
BL="${2:?usage: fastq_uniform.sh <kit dir> <build list> [row]}"
ROW="${3:-${SLURM_ARRAY_TASK_ID:-}}"
[ -n "$ROW" ] || { echo "fastq_uniform.sh: no row (argument 3 or array index)"; exit 2; }
WORK=/scratch/mforooz/onion/work
EIC=/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline   # read only: venv, images, WDLs

echo "=== onion_fastq job ${SLURM_JOB_ID:-none} array ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} row $ROW host $(hostname) $(date -u)"
cat "$KIT/.kit_stamp" 2>/dev/null || true
echo "SLURM_TMPDIR=${SLURM_TMPDIR:-unset}"; df -h "${SLURM_TMPDIR:-/tmp}" | tail -n 1
[ -n "${SLURM_TMPDIR:-}" ] || { echo "no SLURM_TMPDIR: refusing to run Cromwell elsewhere"; exit 3; }

# Caper venv, java, apptainer (EIC_REPRO/003_pipeline/runner/env.sh, sourced, never piped)
source "$EIC/runner/env.sh"
export APPTAINER_TMPDIR="$SLURM_TMPDIR/apptainer_tmp"; mkdir -p "$APPTAINER_TMPDIR"
export PYTHONUNBUFFERED=1
python "$KIT/tools/onion_data/fastq_uniform.py" run --build-list "$BL" --row "$ROW" \
  --work "$WORK" --tmp "$SLURM_TMPDIR" --cpus "${SLURM_CPUS_PER_TASK:-16}"
rc=$?
echo "=== fastq_uniform.py exit=$rc $(date -u)"
[ "$rc" -eq 0 ] || { echo "[onion-fastq] exit=$rc"; exit "$rc"; }

if [ "${ONION_VIEWS:-1}" != 0 ] && [ -f "$KIT/tools/onion_data/align_views.py" ]; then
  EXP=$(awk -F'\t' -v r="$ROW" 'NR==1{for(i=1;i<=NF;i++)c[$i]=i; next} $c["row"]==r{print $c["experiment"]}' "$BL")
  deactivate 2>/dev/null || true
  set +u; module load python/3.10.13; set -u
  { virtualenv --no-download "$SLURM_TMPDIR/kitvenv" &&
    source "$SLURM_TMPDIR/kitvenv/bin/activate" &&
    pip install -q --no-index --upgrade pip &&
    pip install -q --no-index -r "$KIT/requirements-fir.txt" &&
    pip install -q --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"; } ||
    { echo "kit venv setup failed"; echo "[onion-fastq] exit=4"; exit 4; }
  cd "$KIT" && source "$KIT/slurm/_kit_pin.sh"
  python tools/onion_data/align_views.py run --build-list "$BL" --experiment "$EXP" \
    --work "$WORK" --out /scratch/mforooz/onion/measurements
  rc=$?
  echo "=== align_views.py exit=$rc $(date -u)"
else
  echo "=== align_views.py not run (absent from kit or ONION_VIEWS=0)"
fi
echo "[onion-fastq] exit=$rc"
exit "$rc"
