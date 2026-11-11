#!/bin/bash
# Submit a longer Python job inside an ONION code kit and print its job id (t152, chunk K0).
# It does not wait. Push the kit first with push_kit.sh.
#
#   bash tools/onion_kit/remote_run.sh [--cluster fir|nibi] [--gpu] [--name NAME]
#        [--exclude node1,node2] <kit id> <cpus> <mem MB> <hh:mm:ss> <python args...>
#
# e.g.  remote_run.sh D1 4 16000 02:00:00 tools/onion_data/encode_catalog.py --out ...
# The job runs slurm/onion_kit/run.sh from /project/def-maxwl/mforooz/onion_kit/<kit id> with
# `cd <kit>` and `python <python args>`; log /project/def-maxwl/mforooz/ONION/logs/onion_kit/
# run_<kit id>_<job id>.out, last line `[onion-run] exit=<rc>`. CPU jobs get --gres=none; --gpu
# asks for one 1g.10gb H100 slice (and excludes Fir nodes fc10713,fc10715). Keep 4000 MB per CPU
# on CPU jobs. Stdout: one line `<job id>` first, then a line naming the log.
# Cluster: --cluster, else $ONION_CLUSTER, else ./.orchestrate/cluster (cwd), else the umbrella
# checkout's .orchestrate/cluster. Exit 69: tunnel down. ONION_DRY_RUN=1 prints the submit line.
# Nodes: sbatch gets --exclude=<default list for the cluster, plus --exclude> (see exclude_list;
# --gpu on Fir always adds fc10713,fc10715); the sbatch line prints before submission.
# Retry: a job whose venv setup fails before the payload starts (no `[onion-run] env-ready` line;
# e.g. a CVMFS / I/O error) resubmits itself once on another node (slurm/onion_kit/run.sh). This
# script does not wait, so the retry's job id is in the first job's log, as
# `[onion-run] retried-as=<job id> host=<node>`; the retry logs to run_<kit id>_<retry id>.out.
set -euo pipefail

UMBRELLA_ORCH="/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/.claude/worktrees/onion/.orchestrate"
UMBRELLA_CLUSTER_FILE="$UMBRELLA_ORCH/cluster"
KITS=/project/def-maxwl/mforooz/onion_kit
LOGS=/project/def-maxwl/mforooz/ONION/logs/onion_kit
USAGE="usage: remote_run.sh [--cluster fir|nibi] [--gpu] [--name NAME] [--exclude nodes] <kit id> <cpus> <mem MB> <hh:mm:ss> <python args...>"

die() { echo "remote_run: $*" >&2; exit 2; }

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

CLUSTER="" GPU=0 NAME="" EXCLUDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cluster) CLUSTER="${2:?$USAGE}"; shift 2 ;;
    --gpu) GPU=1; shift ;;
    --name) NAME="${2:?$USAGE}"; shift 2 ;;
    --exclude) EXCLUDE="${2?$USAGE}"; shift 2 ;;
    -h|--help) echo "$USAGE"; exit 0 ;;
    -*) die "unknown option $1. $USAGE" ;;
    *) break ;;
  esac
done
[ $# -ge 5 ] || die "$USAGE"
KIT_ID="$1" CPUS="$2" MEM="$3" TIME="$4"; shift 4
[[ "$KIT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "bad kit id '$KIT_ID'"
[[ "$CPUS" =~ ^[1-9][0-9]*$ ]] || die "bad cpus '$CPUS'"
[[ "$MEM" =~ ^[1-9][0-9]*$ ]] || die "bad mem '$MEM' (MB, integer)"
[[ "$TIME" =~ ^([0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]] || die "bad time '$TIME' ([d-]hh:mm:ss)"
[ -z "$NAME" ] || [[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "bad --name '$NAME'"
[[ "$EXCLUDE" =~ ^[A-Za-z0-9._,-]*$ ]] || die "bad --exclude '$EXCLUDE' (node names, comma-separated)"
NAME="${NAME:-onion_$KIT_ID}"

if [ -z "$CLUSTER" ]; then
  if [ -n "${ONION_CLUSTER:-}" ]; then CLUSTER="$ONION_CLUSTER"
  elif [ -f .orchestrate/cluster ]; then CLUSTER="$(tr -d '[:space:]' < .orchestrate/cluster)"
  elif [ -f "$UMBRELLA_CLUSTER_FILE" ]; then CLUSTER="$(tr -d '[:space:]' < "$UMBRELLA_CLUSTER_FILE")"
  else die "no cluster: pass --cluster, set ONION_CLUSTER, or write .orchestrate/cluster"; fi
fi
case "$CLUSTER" in fir|nibi) ;; *) die "unknown cluster '$CLUSTER' (fir|nibi)" ;; esac

GPU_BAD=""
if [ "$GPU" = 1 ]; then
  GRES="--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
  [ "$CLUSTER" = fir ] && GPU_BAD="fc10713,fc10715"
else
  GRES="--gres=none"
  [ $(( MEM / CPUS )) -le 4000 ] || echo "remote_run: note: ${MEM} MB on $CPUS CPUs is over 4000 MB per CPU" >&2
fi

KIT="$KITS/$KIT_ID"
QARGS=""; for a in "$@"; do QARGS="$QARGS $(printf '%q' "$a")"; done
EXCL="$(exclude_list "$CLUSTER" "$GPU_BAD" "$EXCLUDE")"
[[ "$EXCL" =~ ^[A-Za-z0-9._,-]*$ ]] || die "bad exclude list '$EXCL' (node names only)"
OPTS="--job-name=$NAME --time=$TIME --cpus-per-task=$CPUS --mem=${MEM}M $GRES --output=$LOGS/run_${KIT_ID}_%j.out"
SBATCH="sbatch --parsable --export=ALL $OPTS${EXCL:+ --exclude=$EXCL} '$KIT/slurm/onion_kit/run.sh' '$KIT'$QARGS"
SUBMIT="test -f '$KIT/slurm/onion_kit/run.sh' || { echo 'no kit $KIT on $CLUSTER: run push_kit.sh $KIT_ID' >&2; exit 2; }
mkdir -p '$LOGS' && export ONION_RESUBMIT='$OPTS' ONION_EXCLUDE_NODES='$EXCL' && unset ONION_RETRY_OF && $SBATCH"
if [ "${ONION_DRY_RUN:-0}" = 1 ]; then echo "cluster=$CLUSTER"; echo "$SUBMIT"; exit 0; fi
echo "$SBATCH" >&2

ssh -O check "$CLUSTER" >/dev/null 2>&1 || {
  echo "remote_run: no live connection to $CLUSTER. Run: hpc up $CLUSTER" >&2; exit 69; }
ERR="$(mktemp "${TMPDIR:-/tmp}/onion_run_err.XXXXXX")"
JOB="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$CLUSTER" "$SUBMIT" 2>"$ERR" | tail -1 | cut -d';' -f1)" || true
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "remote_run: sbatch did not return a job id: '$JOB'" >&2
  grep -v -E 'nodejs|deprecated|recommend|at your own|messages associated|^-+$|^$' "$ERR" >&2; rm -f "$ERR"; exit 1; }
rm -f "$ERR"
echo "$JOB"
echo "submitted $NAME on $CLUSTER; log $LOGS/run_${KIT_ID}_$JOB.out (a setup retry's job id appears there as [onion-run] retried-as=<id>)"
