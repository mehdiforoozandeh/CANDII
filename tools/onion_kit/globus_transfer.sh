#!/usr/bin/env bash
# Globus transfer of the t112 counterfactual testbed scratch tree, Nibi -> Fir (crux task t155).
#
#   bash tools/onion_kit/globus_transfer.sh route    # check for an authenticated Globus CLI on Nibi, then Fir
#   bash tools/onion_kit/globus_transfer.sh submit   # submit the transfer if a route exists, else print NO_ROUTE
#   bash tools/onion_kit/globus_transfer.sh status   # task id and state, or NO_ROUTE / NOT_SUBMITTED
#   bash tools/onion_kit/globus_transfer.sh verify [--sample N]
#                                                    # file count, total bytes, sha256 on a sample, both sides
#
# Never authenticates: it only uses a Globus CLI session that already exists on a
# cluster login node. Data moves endpoint to endpoint; nothing transits this laptop.
# `verify` reads both trees over `hpc run`, so it also checks a transfer the PI ran
# in the browser. It deletes nothing.
set -u

SRC_EP_NAME='alliancecan#nibi'
DST_EP_NAME='alliancecan#fir-globus'
SRC_PATH='/scratch/mforooz/t112_cf/'
DST_PATH='/project/def-maxwl/mforooz/ONION_import/t112_cf/'
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RES="$ROOT/cruxvault/results/t155"
TASKFILE="$RES/TRANSFER.md"   # holds a line "Globus task id: <uuid>" after submit
GLOBUS_PRE='module load globus-cli >/dev/null 2>&1; command -v globus >/dev/null 2>&1 || { echo NO_CLI; exit 0; };'

# Print the first cluster (nibi, fir) whose login node has an authenticated globus CLI.
find_route() {
  local c out
  for c in nibi fir; do
    out=$(hpc run "$c" "$GLOBUS_PRE globus whoami 2>&1 | head -1" 2>/dev/null)
    rc=$?
    [ $rc -eq 69 ] && { echo "TUNNEL_DOWN $c" >&2; return 69; }
    case "$out" in
      NO_CLI*|*"MissingLoginError"*|*"No authentication"*|*"login"*|"") echo "  $c: $out" >&2 ;;
      *) echo "$c"; return 0 ;;
    esac
  done
  return 1
}

task_id() { [ -s "$TASKFILE" ] && sed -n 's/^Globus task id: *\([0-9a-f-]\{36\}\).*/\1/p' "$TASKFILE" | head -1; }

cmd_route() {
  local r; r=$(find_route); local rc=$?
  [ $rc -eq 69 ] && exit 69
  if [ $rc -eq 0 ]; then echo "ROUTE $r"; else echo "NO_ROUTE"; exit 3; fi
}

cmd_submit() {
  if [ -n "$(task_id)" ]; then echo "already submitted: $(task_id)"; exit 0; fi
  local r; r=$(find_route); local rc=$?
  [ $rc -eq 69 ] && exit 69
  [ $rc -ne 0 ] && { echo "NO_ROUTE (see $RES/NO_ROUTE.md)"; exit 3; }
  hpc run "$r" "$GLOBUS_PRE
    S=\$(globus endpoint search '$SRC_EP_NAME' --filter-scope all --jq 'DATA[0].id' --format unix)
    D=\$(globus endpoint search '$DST_EP_NAME' --filter-scope all --jq 'DATA[0].id' --format unix)
    echo \"source endpoint: \$S\"; echo \"destination endpoint: \$D\"
    globus transfer \"\$S:$SRC_PATH\" \"\$D:$DST_PATH\" --recursive \
      --sync-level checksum --verify-checksum --label 't155 t112_cf nibi to fir' \
      --jq 'task_id' --format unix"
}

cmd_status() {
  local t; t=$(task_id)
  if [ -z "$t" ]; then
    if [ -s "$RES/NO_ROUTE.md" ]; then echo "NO_ROUTE: no authenticated Globus route; see $RES/NO_ROUTE.md"
    else echo "NOT_SUBMITTED"; fi
    exit 0
  fi
  local r; r=$(find_route) || { echo "task $t: NO_ROUTE to query it"; exit 3; }
  hpc run "$r" "$GLOBUS_PRE globus task show $t --jq '[status,files,files_transferred,bytes_transferred,faults]' --format unix"
}

# One side: "files <n> bytes <b>" and sha256 of the sampled relative paths.
side_stats() { # cluster root samplefile
  hpc run "$1" "cd '$2' 2>/dev/null || { echo MISSING; exit 0; }
    find . -type f -printf '%s\n' | awk '{n++;s+=\$1}END{printf \"files %d bytes %d\n\",n,s}'
    while read -r p; do sha256sum \"\$p\" 2>/dev/null || echo \"MISSING  \$p\"; done <<'EOF'
$(cat "$3")
EOF"
}

cmd_verify() {
  local n=20; [ "${1:-}" = "--sample" ] && n="$2"
  local tmp; tmp=$(mktemp -d)
  # Deterministic sample: every k-th file of the sorted source list, files under 2 GB only.
  hpc run nibi "cd '$SRC_PATH' && find . -type f -size -2G | LC_ALL=C sort" > "$tmp/all" || exit $?
  awk -v n="$n" '{a[NR]=$0}END{k=int(NR/n); if(k<1)k=1; c=0; for(i=1;i<=NR&&c<n;i+=k){print a[i]; c++}}' "$tmp/all" > "$tmp/sample"
  side_stats nibi "$SRC_PATH" "$tmp/sample" > "$tmp/src" || exit $?
  side_stats fir "$DST_PATH" "$tmp/sample" > "$tmp/dst" || exit $?
  echo "source      $(head -1 "$tmp/src")"; echo "destination $(head -1 "$tmp/dst")"
  local ok=1
  [ "$(head -1 "$tmp/src")" = "$(head -1 "$tmp/dst")" ] || { echo "COUNT/BYTES MISMATCH"; ok=0; }
  # A sampled source line (sha256 + path) with no identical destination line is a mismatch.
  local diffs; diffs=$(grep -vxFf <(tail -n +2 "$tmp/dst") <(tail -n +2 "$tmp/src") | wc -l | tr -d ' ')
  echo "sha256 sample: $(wc -l < "$tmp/sample" | tr -d ' ') files, $diffs mismatched"
  [ "$diffs" -eq 0 ] || ok=0
  local t; t=$(task_id); [ -n "$t" ] && echo "globus task $t: $(cmd_status 2>&1 | tail -1)"
  rm -rf "$tmp"
  if [ $ok -eq 1 ]; then echo "VERIFY PASS"; else echo "VERIFY FAIL"; exit 1; fi
}

case "${1:-}" in
  route) cmd_route ;;
  submit) cmd_submit ;;
  status) cmd_status ;;
  verify) shift; cmd_verify "$@" ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
