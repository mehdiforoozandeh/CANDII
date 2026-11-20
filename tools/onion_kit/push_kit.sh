#!/bin/bash
# Push this checkout as a code kit to the ONION cluster (t152 programme, chunk K0).
#
#   bash tools/onion_kit/push_kit.sh [--cluster fir|nibi] <kit id>
#   -> /project/def-maxwl/mforooz/onion_kit/<kit id>/   (same path on both clusters)
#
# What is sent: the files git would see in this checkout (tracked, plus untracked files that are
# not ignored), minus .git, .claude, .orchestrate and cruxvault/raw. Ignored files (golden refs,
# __pycache__) never go. A file that left the checkout also leaves the kit (--delete-excluded),
# so `pytest tests/` on the cluster never runs a test that no longer exists here.
# `wheels/` in the kit is a symlink to the per-cluster /project/def-maxwl/mforooz/onion_kit/wheels.
#
# Cluster: --cluster, else $ONION_CLUSTER, else ./.orchestrate/cluster (cwd), else the umbrella
# checkout's .orchestrate/cluster. Exit 69: the ssh tunnel is down (run `hpc up <cluster>`).
# ONION_DRY_RUN=1 prints what would run and touches nothing.
set -euo pipefail

UMBRELLA_CLUSTER_FILE="/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/.claude/worktrees/onion/.orchestrate/cluster"
KITS=/project/def-maxwl/mforooz/onion_kit
USAGE="usage: push_kit.sh [--cluster fir|nibi] <kit id>"

die() { echo "push_kit: $*" >&2; exit 2; }

CLUSTER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cluster) CLUSTER="${2:?$USAGE}"; shift 2 ;;
    -h|--help) echo "$USAGE"; exit 0 ;;
    -*) die "unknown option $1. $USAGE" ;;
    *) break ;;
  esac
done
[ $# -eq 1 ] || die "$USAGE"
KIT_ID="$1"
[[ "$KIT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "bad kit id '$KIT_ID' (letters, digits, . _ -)"
[ "$KIT_ID" != wheels ] && [ "$KIT_ID" != logs ] || die "kit id '$KIT_ID' is reserved"

if [ -z "$CLUSTER" ]; then
  if [ -n "${ONION_CLUSTER:-}" ]; then CLUSTER="$ONION_CLUSTER"
  elif [ -f .orchestrate/cluster ]; then CLUSTER="$(tr -d '[:space:]' < .orchestrate/cluster)"
  elif [ -f "$UMBRELLA_CLUSTER_FILE" ]; then CLUSTER="$(tr -d '[:space:]' < "$UMBRELLA_CLUSTER_FILE")"
  else die "no cluster: pass --cluster, set ONION_CLUSTER, or write .orchestrate/cluster"; fi
fi
case "$CLUSTER" in fir|nibi) ;; *) die "unknown cluster '$CLUSTER' (fir|nibi)" ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$KITS/$KIT_ID"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)

# rsync filter: every kept file and each of its parent directories, then exclude the rest.
FILTER="$(mktemp "${TMPDIR:-/tmp}/onion_kit_filter.XXXXXX")"
trap 'rm -f "$FILTER"' EXIT
echo "P /wheels" > "$FILTER"
git -C "$ROOT" ls-files -co --exclude-standard \
  | grep -v -E '^(\.git/|\.claude/|\.orchestrate/|cruxvault/raw/)' \
  | while IFS= read -r f; do [ -e "$ROOT/$f" ] && printf '%s\n' "$f"; done \
  | awk '{ n = split($0, a, "/"); p = "";
           for (i = 1; i < n; i++) { p = p a[i] "/"; if (!(p in seen)) { seen[p] = 1; print "+ /" p } }
           print "+ /" $0 }
         END { print "- *" }' >> "$FILTER"
NFILES=$(grep -c '^+ /.*[^/]$' "$FILTER" || true)
STAMP="kit=$KIT_ID git=$(git -C "$ROOT" rev-parse --short HEAD) dirty=$(git -C "$ROOT" status --porcelain | wc -l | tr -d ' ') pushed=$(date -u +%Y-%m-%dT%H:%M:%SZ) from=$ROOT"

RSYNC=(rsync -a --no-g --no-p --delete --delete-excluded --filter="merge $FILTER"
       -e "ssh -q -o BatchMode=yes -o ConnectTimeout=10" "$ROOT/" "$CLUSTER:$DEST/")
if [ "${ONION_DRY_RUN:-0}" = 1 ]; then
  echo "cluster=$CLUSTER"
  echo "dest=$DEST"
  echo "files=$NFILES"
  echo "${RSYNC[*]}"
  exit 0
fi

ssh -O check "$CLUSTER" >/dev/null 2>&1 || {
  echo "push_kit: no live connection to $CLUSTER. Run: hpc up $CLUSTER" >&2; exit 69; }
"${SSH[@]}" "$CLUSTER" "mkdir -p '$DEST'" 2>/dev/null || { echo "push_kit: cannot create $DEST on $CLUSTER" >&2; exit 1; }
"${RSYNC[@]}"
"${SSH[@]}" "$CLUSTER" "set -e; cd '$DEST'; ln -sfn '$KITS/wheels' wheels; printf '%s\n' '$STAMP' > .kit_stamp;
  n=\$(ls '$KITS/wheels' 2>/dev/null | wc -l); [ \"\$n\" -gt 0 ] || echo 'push_kit: WARNING $KITS/wheels is empty on $CLUSTER'" 2>/dev/null
echo "pushed $NFILES files to $CLUSTER:$DEST ($STAMP)"
