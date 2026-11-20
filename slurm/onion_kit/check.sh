#!/bin/bash
# Short Python check inside an ONION code kit (t152, chunk K0). Submitted by
# tools/onion_kit/remote_check.sh; not meant to be submitted by hand.
#
#   sbatch --output=<log> $KIT/slurm/onion_kit/check.sh <kit dir> <python args...>
#
# Builds a venv in $SLURM_TMPDIR from requirements-fir.txt and the kit's wheels/, pins `candi`
# to the kit (slurm/_kit_pin.sh), cds into the kit and runs `python <python args>`.
# Setup output goes to <log>.setup; the log itself holds only the check's own stdout and stderr,
# then one last line `[onion-check] exit=<rc>` that remote_check.sh reads.
# remote_check.sh may override --time, --cpus-per-task and --mem on the sbatch line.
# Once the environment is ready the log gets `[onion-check] env-ready host=<node>` before the payload.
# Retry: when the venv setup fails (e.g. a CVMFS / I/O error on the node) the job resubmits itself
# once, with the same sbatch options ($ONION_RESUBMIT, exported by remote_check.sh) and this node
# added to the exclude list ($ONION_EXCLUDE_NODES), logs `[onion-check] retried-as=<job id>
# host=<node>` and exits 3. A retried job ($ONION_RETRY_OF set) never retries again.
# A job sbatch'ed directly (no $ONION_RESUBMIT, e.g. a dependent guardrail job) rebuilds those
# options and its exclude list from `scontrol show job` and retries the same way (F3). Prefer
# `remote_run.sh --after <job>` for dependent jobs: it also applies the cluster's exclude list.
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_check
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --gres=none

set -uo pipefail
KIT="${1:?usage: check.sh <kit dir> <python args...>}"
shift
[ $# -gt 0 ] || { echo "check.sh: no python args" >&2; echo "[onion-check] exit=2"; exit 2; }
SETUP_LOG="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *StdOut=//p').setup"
[ "$SETUP_LOG" != .setup ] || SETUP_LOG="$SLURM_TMPDIR/setup.log"

setup() {
  echo "=== onion check job $SLURM_JOB_ID host=$(hostname) kit=$KIT $(date -u)"
  cat "$KIT/.kit_stamp" 2>/dev/null || true
  set +u; module load python/3.10.13; set -u
  virtualenv --no-download "$SLURM_TMPDIR/venv" &&
  source "$SLURM_TMPDIR/venv/bin/activate" &&
  pip install --no-index --upgrade pip &&
  pip install --no-index -r "$KIT/requirements-fir.txt" &&
  pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
}
# A job submitted by raw sbatch (e.g. a guardrail job chained with --dependency=afterok, P2a) has
# no ONION_RESUBMIT: rebuild the options from the job record, so it retries too (F3, DEFECTS 1).
# The dependency is not carried over: by now it has held, and Fir refuses afterok on an old job.
options_from_job() {   # scontrol record on stdin -> line 1 sbatch options, line 2 ExcNodeList
  local rec name tl cpt mem out exc gres
  rec="$(tr ' ' '\n')"
  field() { printf '%s\n' "$rec" | sed -n "s/^$1=//p" | head -1; }
  name="$(field JobName)"; tl="$(field TimeLimit)"; cpt="$(field 'CPUs\/Task')"
  mem="$(field MinMemoryNode)"; out="$(field StdOut)"
  exc="$(field ExcNodeList)"; gres="$(field TresPerNode)"
  [ -n "$tl" ] && [ -n "$cpt" ] && [ -n "$mem" ] || return 1
  # the log path: put %j back where this job's id was, else add a suffix
  if [ -n "$out" ]; then out="${out//$SLURM_JOB_ID/%j}"; [[ "$out" == *%j* ]] || out="$out.retry_%j"; fi
  case "$gres" in gres/*) gres="--gres=${gres#gres/}" ;; *) gres="--gres=none" ;; esac
  printf '%s\n' "${name:+--job-name=$name }--time=$tl --cpus-per-task=$cpt --mem=$mem $gres${out:+ --output=$out}"
  [ "$exc" = "(null)" ] && exc=""
  printf '%s\n' "$exc"
}
if [ -z "${ONION_RESUBMIT:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  if derived="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | options_from_job)"; then
    ONION_RESUBMIT="$(printf '%s\n' "$derived" | sed -n 1p)"
    [ -n "${ONION_EXCLUDE_NODES:-}" ] || ONION_EXCLUDE_NODES="$(printf '%s\n' "$derived" | sed -n 2p)"
  fi
fi
retry_elsewhere() {
  [ -z "${ONION_RETRY_OF:-}" ] && [ -n "${ONION_RESUBMIT:-}" ] || return 1
  local host ex new
  host="$(hostname -s 2>/dev/null || hostname)"
  ex="${ONION_EXCLUDE_NODES:+$ONION_EXCLUDE_NODES,}$host"
  # ONION_RESUBMIT is a word list of sbatch options (no spaces inside a value)
  # shellcheck disable=SC2086
  new="$( (deactivate 2>/dev/null || true
           ONION_RETRY_OF="$SLURM_JOB_ID" ONION_EXCLUDE_NODES="$ex" sbatch --parsable --export=ALL \
             $ONION_RESUBMIT --exclude="$ex" "$KIT/slurm/onion_kit/check.sh" "$KIT" "$@") 2>&1 |
         tail -1 | cut -d';' -f1)"
  [[ "$new" =~ ^[0-9]+$ ]] || { echo "check.sh: the retry was not submitted: $new"; return 1; }
  echo "[onion-check] retried-as=$new host=$host"
}
if ! setup > "$SETUP_LOG" 2>&1; then
  echo "check.sh: venv setup failed on $(hostname); last lines of $SETUP_LOG:"
  tail -20 "$SETUP_LOG"
  retry_elsewhere "$@"
  echo "[onion-check] exit=3"
  exit 3
fi
source "$SLURM_TMPDIR/venv/bin/activate"
cd "$KIT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"   # not the node's cores (F3)
source "$KIT/slurm/_kit_pin.sh" >> "$SETUP_LOG" 2>&1 || { echo "check.sh: kit pin failed"; tail -5 "$SETUP_LOG"; echo "[onion-check] exit=4"; exit 4; }
echo "[onion-check] env-ready host=$(hostname)"
python "$@"
rc=$?
echo "[onion-check] exit=$rc"
exit $rc
