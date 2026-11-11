#!/bin/bash
# Submit a longer Python job inside an ONION code kit and print its job id (t152, chunk K0).
# It does not wait. Push the kit first with push_kit.sh.
#
#   bash tools/onion_kit/remote_run.sh [--cluster fir|nibi] [--gpu] [--name NAME]
#        <kit id> <cpus> <mem MB> <hh:mm:ss> <python args...>
#
# e.g.  remote_run.sh D1 4 16000 02:00:00 tools/onion_data/encode_catalog.py --out ...
# The job runs slurm/onion_kit/run.sh from /project/def-maxwl/mforooz/onion_kit/<kit id> with
# `cd <kit>` and `python <python args>`; log /project/def-maxwl/mforooz/ONION/logs/onion_kit/
# run_<kit id>_<job id>.out, last line `[onion-run] exit=<rc>`. CPU jobs get --gres=none; --gpu
# asks for one 1g.10gb H100 slice (and excludes Fir nodes fc10713,fc10715). Keep 4000 MB per CPU
# on CPU jobs. Stdout: one line `<job id>` first, then a line naming the log.
# Cluster: --cluster, else $ONION_CLUSTER, else ./.orchestrate/cluster (cwd), else the umbrella
# checkout's .orchestrate/cluster. Exit 69: tunnel down. ONION_DRY_RUN=1 prints the submit line.
set -euo pipefail

UMBRELLA_CLUSTER_FILE="/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/.claude/worktrees/onion/.orchestrate/cluster"
KITS=/project/def-maxwl/mforooz/onion_kit
LOGS=/project/def-maxwl/mforooz/ONION/logs/onion_kit
USAGE="usage: remote_run.sh [--cluster fir|nibi] [--gpu] [--name NAME] <kit id> <cpus> <mem MB> <hh:mm:ss> <python args...>"

die() { echo "remote_run: $*" >&2; exit 2; }

CLUSTER="" GPU=0 NAME=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cluster) CLUSTER="${2:?$USAGE}"; shift 2 ;;
    --gpu) GPU=1; shift ;;
    --name) NAME="${2:?$USAGE}"; shift 2 ;;
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
NAME="${NAME:-onion_$KIT_ID}"

if [ -z "$CLUSTER" ]; then
  if [ -n "${ONION_CLUSTER:-}" ]; then CLUSTER="$ONION_CLUSTER"
  elif [ -f .orchestrate/cluster ]; then CLUSTER="$(tr -d '[:space:]' < .orchestrate/cluster)"
  elif [ -f "$UMBRELLA_CLUSTER_FILE" ]; then CLUSTER="$(tr -d '[:space:]' < "$UMBRELLA_CLUSTER_FILE")"
  else die "no cluster: pass --cluster, set ONION_CLUSTER, or write .orchestrate/cluster"; fi
fi
case "$CLUSTER" in fir|nibi) ;; *) die "unknown cluster '$CLUSTER' (fir|nibi)" ;; esac

if [ "$GPU" = 1 ]; then
  GRES="--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
  [ "$CLUSTER" = fir ] && GRES="$GRES --exclude=fc10713,fc10715"
else
  GRES="--gres=none"
  [ $(( MEM / CPUS )) -le 4000 ] || echo "remote_run: note: ${MEM} MB on $CPUS CPUs is over 4000 MB per CPU" >&2
fi

KIT="$KITS/$KIT_ID"
QARGS=""; for a in "$@"; do QARGS="$QARGS $(printf '%q' "$a")"; done
SUBMIT="test -f '$KIT/slurm/onion_kit/run.sh' || { echo 'no kit $KIT on $CLUSTER: run push_kit.sh $KIT_ID' >&2; exit 2; }
mkdir -p '$LOGS' && sbatch --parsable --job-name='$NAME' --time=$TIME --cpus-per-task=$CPUS --mem=${MEM}M $GRES --output='$LOGS/run_${KIT_ID}_%j.out' '$KIT/slurm/onion_kit/run.sh' '$KIT'$QARGS"
if [ "${ONION_DRY_RUN:-0}" = 1 ]; then echo "cluster=$CLUSTER"; echo "$SUBMIT"; exit 0; fi

ssh -O check "$CLUSTER" >/dev/null 2>&1 || {
  echo "remote_run: no live connection to $CLUSTER. Run: hpc up $CLUSTER" >&2; exit 69; }
ERR="$(mktemp "${TMPDIR:-/tmp}/onion_run_err.XXXXXX")"
JOB="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$CLUSTER" "$SUBMIT" 2>"$ERR" | tail -1 | cut -d';' -f1)" || true
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "remote_run: sbatch did not return a job id: '$JOB'" >&2
  grep -v -E 'nodejs|deprecated|recommend|at your own|messages associated|^-+$|^$' "$ERR" >&2; rm -f "$ERR"; exit 1; }
rm -f "$ERR"
echo "$JOB"
echo "submitted $NAME on $CLUSTER; log $LOGS/run_${KIT_ID}_$JOB.out"
