#!/bin/bash
# Longer Python job inside an ONION code kit (t152, chunk K0). Submitted by
# tools/onion_kit/remote_run.sh, which sets --cpus-per-task, --mem, --time (and the GPU slice
# with --gpu) on the sbatch line; the header values below are only the fallbacks.
#
#   sbatch --output=<log> $KIT/slurm/onion_kit/run.sh <kit dir> <python args...>
#
# Builds a venv in $SLURM_TMPDIR from requirements-fir.txt and the kit's wheels/, pins `candi`
# to the kit (slurm/_kit_pin.sh), cds into the kit and runs `python <python args>`.
# The last log line is `[onion-run] exit=<rc>`. Once the environment is ready the log gets
# `[onion-run] env-ready host=<node>` before the payload.
# Retry: when the venv setup fails (e.g. a CVMFS / I/O error on the node) the job resubmits itself
# once, with the same sbatch options ($ONION_RESUBMIT, exported by remote_run.sh) and this node
# added to the exclude list ($ONION_EXCLUDE_NODES), logs `[onion-run] retried-as=<job id>
# host=<node>` and exits 3. A retried job ($ONION_RETRY_OF set) never retries again.
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_run
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --gres=none

set -uo pipefail
KIT="${1:?usage: run.sh <kit dir> <python args...>}"
shift
[ $# -gt 0 ] || { echo "run.sh: no python args" >&2; echo "[onion-run] exit=2"; exit 2; }
echo "=== onion run job $SLURM_JOB_ID host=$(hostname) kit=$KIT $(date -u)"
cat "$KIT/.kit_stamp" 2>/dev/null || true
echo "=== python $*"
set +u; module load python/3.10.13; set -u
{ virtualenv --no-download "$SLURM_TMPDIR/venv" &&
  source "$SLURM_TMPDIR/venv/bin/activate" &&
  pip install -q --no-index --upgrade pip &&
  pip install -q --no-index -r "$KIT/requirements-fir.txt" &&
  pip install -q --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"; } || {
  echo "run.sh: venv setup failed on $(hostname)"
  if [ -z "${ONION_RETRY_OF:-}" ] && [ -n "${ONION_RESUBMIT:-}" ]; then
    host="$(hostname -s 2>/dev/null || hostname)"
    ex="${ONION_EXCLUDE_NODES:+$ONION_EXCLUDE_NODES,}$host"
    # ONION_RESUBMIT is a word list of sbatch options (no spaces inside a value)
    # shellcheck disable=SC2086
    new="$( (deactivate 2>/dev/null || true
             ONION_RETRY_OF="$SLURM_JOB_ID" ONION_EXCLUDE_NODES="$ex" sbatch --parsable \
               --export=ALL $ONION_RESUBMIT --exclude="$ex" "$KIT/slurm/onion_kit/run.sh" "$KIT" "$@") 2>&1 |
           tail -1 | cut -d';' -f1)"
    if [[ "$new" =~ ^[0-9]+$ ]]; then echo "[onion-run] retried-as=$new host=$host"
    else echo "run.sh: the retry was not submitted: $new"; fi
  fi
  echo "[onion-run] exit=3"; exit 3; }
source "$SLURM_TMPDIR/venv/bin/activate"
cd "$KIT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
source "$KIT/slurm/_kit_pin.sh"
echo "[onion-run] env-ready host=$(hostname)"
python "$@"
rc=$?
echo "=== done $(date -u)"
echo "[onion-run] exit=$rc"
exit $rc
