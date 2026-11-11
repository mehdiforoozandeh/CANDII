#!/bin/bash
# Run a short Python check inside an ONION code kit as an sbatch job, wait for it, print its
# output and return its exit code (t152, chunk K0). Push the kit first with push_kit.sh.
#
#   bash tools/onion_kit/remote_check.sh [--cluster fir|nibi] [--time hh:mm:ss] [--cpus N]
#        [--mem MB] [--max-wait SECONDS] [--exclude node1,node2] <kit id> <python args...>
#   bash tools/onion_kit/remote_check.sh [--cluster fir|nibi] [--max-wait SECONDS] --wait <job id>
#
# e.g.  remote_check.sh K0 -c "import candi; print(candi.__file__)"
#       remote_check.sh main --time 02:00:00 --cpus 8 --mem 32000 -m pytest tests/ -q
# The job runs slurm/onion_kit/check.sh from /project/def-maxwl/mforooz/onion_kit/<kit id> with
# `cd <kit>` and `python <python args>`. Defaults: 00:30:00, 2 CPU, 8000 MB, --gres=none.
# The job id prints at once. The wait is bounded (default 1800 s). If the job has not finished by
# then, the script prints the `--wait <job id>` line to re-run and exits 75; never resubmit.
# Exit codes: the check's own code; 69 tunnel down; 75 still queued or running; 2 bad usage.
# Cluster: --cluster, else $ONION_CLUSTER, else ./.orchestrate/cluster (cwd), else the umbrella
# checkout's .orchestrate/cluster. ONION_DRY_RUN=1 prints the submit command and submits nothing.
# Nodes: sbatch gets --exclude=<default list for the cluster, plus --exclude> (see exclude_list);
# the sbatch line prints before submission.
# Retry: a job whose venv setup fails before the payload starts (no `[onion-check] env-ready` line;
# e.g. a CVMFS / I/O error) resubmits itself once on another node (slurm/onion_kit/check.sh); this
# script prints both job ids and waits on the retry, also in the --wait form.
set -euo pipefail

UMBRELLA_ORCH="/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/.claude/worktrees/onion/.orchestrate"
UMBRELLA_CLUSTER_FILE="$UMBRELLA_ORCH/cluster"
KITS=/project/def-maxwl/mforooz/onion_kit
LOGS=/project/def-maxwl/mforooz/ONION/logs/onion_kit
USAGE="usage: remote_check.sh [--cluster fir|nibi] [--time hh:mm:ss] [--cpus N] [--mem MB] [--max-wait S] [--exclude nodes] <kit id> <python args...>
       remote_check.sh [--cluster fir|nibi] [--max-wait S] --wait <job id>"

die() { echo "remote_check: $*" >&2; exit 2; }

# Exclude list: the default per cluster (--exclude adds to it): $ONION_EXCLUDE (a node list; set
# but empty means none), else ./.orchestrate/exclude_<cluster>, else the umbrella checkout's
# .orchestrate/exclude_<cluster>. A file lists node names separated by commas, spaces or newlines;
# `#` starts a comment.
nodes_of() { sed 's/#.*//' | tr ', \t' '\n\n\n' | sed '/^$/d'; }
exclude_list() {   # <cluster> <extra nodes...> -> one comma-separated, de-duplicated list
  local c="$1" f=""; shift
  { if [ -n "${ONION_EXCLUDE+x}" ]; then printf '%s\n' "$ONION_EXCLUDE"
    elif [ -f ".orchestrate/exclude_$c" ]; then f=".orchestrate/exclude_$c"
    elif [ -f "$UMBRELLA_ORCH/exclude_$c" ]; then f="$UMBRELLA_ORCH/exclude_$c"; fi
    [ -z "$f" ] || cat "$f"
    printf '%s\n' "$@"; } | nodes_of | awk '!seen[$0]++' | paste -sd, -
}

CLUSTER="" TIME=00:30:00 CPUS=2 MEM=8000 MAX_WAIT=1800 WAIT_JOB="" EXCLUDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cluster) CLUSTER="${2:?$USAGE}"; shift 2 ;;
    --time) TIME="${2:?$USAGE}"; shift 2 ;;
    --cpus) CPUS="${2:?$USAGE}"; shift 2 ;;
    --mem) MEM="${2:?$USAGE}"; shift 2 ;;
    --max-wait) MAX_WAIT="${2:?$USAGE}"; shift 2 ;;
    --wait) WAIT_JOB="${2:?$USAGE}"; shift 2 ;;
    --exclude) EXCLUDE="${2?$USAGE}"; shift 2 ;;
    -h|--help) echo "$USAGE"; exit 0 ;;
    *) break ;;
  esac
done
[[ "$TIME" =~ ^[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]] || die "bad --time '$TIME' (hh:mm:ss)"
[[ "$CPUS" =~ ^[1-9][0-9]*$ ]] || die "bad --cpus '$CPUS'"
[[ "$MEM" =~ ^[1-9][0-9]*$ ]] || die "bad --mem '$MEM' (MB, integer)"
[[ "$MAX_WAIT" =~ ^[0-9]+$ ]] || die "bad --max-wait '$MAX_WAIT' (seconds)"
[[ "$EXCLUDE" =~ ^[A-Za-z0-9._,-]*$ ]] || die "bad --exclude '$EXCLUDE' (node names, comma-separated)"
if [ -n "$WAIT_JOB" ]; then
  [[ "$WAIT_JOB" =~ ^[0-9]+$ ]] || die "bad job id '$WAIT_JOB'"
else
  [ $# -ge 2 ] || die "$USAGE"
  KIT_ID="$1"; shift
  [[ "$KIT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "bad kit id '$KIT_ID'"
fi

if [ -z "$CLUSTER" ]; then
  if [ -n "${ONION_CLUSTER:-}" ]; then CLUSTER="$ONION_CLUSTER"
  elif [ -f .orchestrate/cluster ]; then CLUSTER="$(tr -d '[:space:]' < .orchestrate/cluster)"
  elif [ -f "$UMBRELLA_CLUSTER_FILE" ]; then CLUSTER="$(tr -d '[:space:]' < "$UMBRELLA_CLUSTER_FILE")"
  else die "no cluster: pass --cluster, set ONION_CLUSTER, or write .orchestrate/cluster"; fi
fi
case "$CLUSTER" in fir|nibi) ;; *) die "unknown cluster '$CLUSTER' (fir|nibi)" ;; esac

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 "$CLUSTER")

if [ -z "$WAIT_JOB" ]; then
  KIT="$KITS/$KIT_ID"
  QARGS=""; for a in "$@"; do QARGS="$QARGS $(printf '%q' "$a")"; done
  EXCL="$(exclude_list "$CLUSTER" "$EXCLUDE")"
  [[ "$EXCL" =~ ^[A-Za-z0-9._,-]*$ ]] || die "bad exclude list '$EXCL' (node names only)"
  OPTS="--job-name=onion_check_$KIT_ID --time=$TIME --cpus-per-task=$CPUS --mem=${MEM}M --gres=none --output=$LOGS/check_%j.out"
  SBATCH="sbatch --parsable --export=ALL $OPTS${EXCL:+ --exclude=$EXCL} '$KIT/slurm/onion_kit/check.sh' '$KIT'$QARGS"
  SUBMIT="test -f '$KIT/slurm/onion_kit/check.sh' || { echo 'no kit $KIT on $CLUSTER: run push_kit.sh $KIT_ID' >&2; exit 2; }
mkdir -p '$LOGS' && export ONION_RESUBMIT='$OPTS' ONION_EXCLUDE_NODES='$EXCL' && unset ONION_RETRY_OF && $SBATCH"
  if [ "${ONION_DRY_RUN:-0}" = 1 ]; then echo "cluster=$CLUSTER"; echo "$SUBMIT"; exit 0; fi
  echo "$SBATCH"
  ssh -O check "$CLUSTER" >/dev/null 2>&1 || {
    echo "remote_check: no live connection to $CLUSTER. Run: hpc up $CLUSTER" >&2; exit 69; }
  ERR="$(mktemp "${TMPDIR:-/tmp}/onion_check_err.XXXXXX")"
  JOB="$("${SSH[@]}" "$SUBMIT" 2>"$ERR" | tail -1 | cut -d';' -f1)" || true
  [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "remote_check: sbatch did not return a job id: '$JOB'" >&2
    grep -v -E 'nodejs|deprecated|recommend|at your own|messages associated|^-+$|^$' "$ERR" >&2; rm -f "$ERR"; exit 1; }
  rm -f "$ERR"
  echo "submitted check job $JOB on $CLUSTER (kit $KIT_ID); log $LOGS/check_$JOB.out"
else
  JOB="$WAIT_JOB"
  if [ "${ONION_DRY_RUN:-0}" = 1 ]; then echo "cluster=$CLUSTER"; echo "wait $JOB"; exit 0; fi
  ssh -O check "$CLUSTER" >/dev/null 2>&1 || {
    echo "remote_check: no live connection to $CLUSTER. Run: hpc up $CLUSTER" >&2; exit 69; }
  echo "waiting on check job $JOB on $CLUSTER"
fi

RETRIED=0
while :; do
  LOG="$LOGS/check_$JOB.out"
  START=$(date +%s) STATE=""
  while :; do
    Q="$("${SSH[@]}" "squeue -h -j $JOB -o %T 2>/dev/null || true" 2>/dev/null)" || {
      echo "remote_check: lost the connection to $CLUSTER. Run: hpc up $CLUSTER; then re-run:" >&2
      echo "  bash tools/onion_kit/remote_check.sh --cluster $CLUSTER --wait $JOB" >&2; exit 69; }
    STATE="$(printf '%s\n' "$Q" | head -1 | tr -d '[:space:]')"
    [ -z "$STATE" ] && break
    if [ $(( $(date +%s) - START )) -ge "$MAX_WAIT" ]; then
      echo "remote_check: job $JOB still $STATE after ${MAX_WAIT}s. Do not resubmit. Re-run:"
      echo "  bash tools/onion_kit/remote_check.sh --cluster $CLUSTER --wait $JOB"
      exit 75
    fi
    sleep 15
  done

  OUT="$("${SSH[@]}" "cat '$LOG' 2>/dev/null || true" 2>/dev/null)"
  NEW="$(printf '%s\n' "$OUT" | sed -n 's/^\[onion-check\] retried-as=\([0-9]*\) .*$/\1/p' | tail -1)"
  if [ -n "$NEW" ] && [ "$RETRIED" = 0 ]; then
    HOST="$(printf '%s\n' "$OUT" | sed -n 's/^\[onion-check\] retried-as=[0-9]* host=//p' | tail -1)"
    printf '%s\n' "$OUT" | grep -v '^\[onion-check\] ' || true
    echo "remote_check: job $JOB died in environment setup on $HOST; retried once as job $NEW (excluding $HOST). Jobs: $JOB $NEW"
    JOB="$NEW" RETRIED=1
    continue
  fi
  break
done
RC="$(printf '%s\n' "$OUT" | sed -n 's/^\[onion-check\] exit=\([0-9]*\)$/\1/p' | tail -1)"
printf '%s\n' "$OUT" | grep -v '^\[onion-check\] ' || true
if [ -n "$RC" ] && ! printf '%s\n' "$OUT" | grep -q '^\[onion-check\] env-ready'; then
  echo "remote_check: job $JOB died in environment setup before the payload started (no retry left)" >&2
fi
if [ -z "$RC" ]; then
  ACCT="$("${SSH[@]}" "sacct -n -X -j $JOB -o State,ExitCode -P 2>/dev/null | head -1" 2>/dev/null || true)"
  echo "remote_check: job $JOB ended without an exit line (sacct: ${ACCT:-none}); setup log tail:" >&2
  "${SSH[@]}" "tail -15 '$LOG.setup'" 2>/dev/null >&2 || true
  RC=1
fi
exit "$RC"
