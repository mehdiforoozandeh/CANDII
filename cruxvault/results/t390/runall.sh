#!/bin/bash
# t390 results (EIC entry for a count-space model). Rebuilds this folder:
#
#   bash runall.sh
#
# No arguments, no environment variables. Today the folder holds the observed-data ladder probe
# (ladder_probe/): LADDER_PROBE.md is hand-written prose (not an output); the five TSVs beside it
# are outputs of tools/eic_entry/collect.py on small files pulled from Nibi. The cluster arrays
# and the jobs that made them are listed in LADDER_PROBE.md; they live under
# /scratch/mforooz/eicx/ladder (the build brief's location, not /project/<task-id>: a deviation
# from AGENTS.md 4.11 rule 4, recorded here). Rebuilding the arrays: tools/eic_entry/run_probe.sh.
#
# Steps:
#   1. pull      ladder_probe/*.tsv missing -> rsync the small files (scores, qc, placements,
#                sacct, job reports) from Nibi into a temporary folder
#   2. collect   run collect.py on them -> ladder_probe/{scores_all23,placement,rung_compare,checks,resources}.tsv
#   3. manifest  MANIFEST.tsv missing -> hash every output
#   4. check     always: re-hash every MANIFEST.tsv row, exit 1 on any mismatch
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=/Users/mforooz/miniforge3/envs/candii/bin/python
REPO=$(git rev-parse --show-toplevel)
REMOTE=nibi:/scratch/mforooz/eicx/ladder/
OUTS=(ladder_probe/scores_all23.tsv ladder_probe/placement.tsv ladder_probe/rung_compare.tsv
      ladder_probe/checks.tsv ladder_probe/resources.tsv)

sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
        else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

missing=0; for f in "${OUTS[@]}"; do [ -e "$f" ] || missing=1; done
if [ "$missing" = 1 ]; then
  TMP=$(mktemp -d)
  rsync -a -e 'ssh -o BatchMode=yes' --include='*/' --include='scores/*/*.tsv' --include='qc/*.json' \
    --include='placement_*.tsv' --include='sacct.tsv' --include='rungs/*/*.json' --include='peaks/*.json' \
    --include='starts/*.json' --exclude='code/' --exclude='score_cache/' --exclude='*' "$REMOTE" "$TMP/" \
    || { echo "rsync failed: run \`hpc up nibi\`"; exit 69; }
  echo "1 pull: $TMP"
  "$PY" "$REPO/tools/eic_entry/collect.py" "$TMP" ladder_probe > /dev/null
  echo "2 collect: wrote ${OUTS[*]}"
  rm -rf "$TMP"
fi

if [ ! -s MANIFEST.tsv ]; then
  printf 'path\tbytes\tsha256\twhere\n' > MANIFEST.tsv
  for f in "${OUTS[@]}"; do printf '%s\t%s\t%s\tgit\n' "$f" "$(wc -c < "$f" | tr -d ' ')" "$(sha "$f")" >> MANIFEST.tsv; done
  echo "3 manifest: wrote MANIFEST.tsv"
fi

bad=0; n=0
while IFS=$'\t' read -r path bytes want where; do
  [ "$path" = path ] && continue
  n=$((n + 1))
  if [ "$(sha "$path")" != "$want" ]; then echo "MISMATCH $path ($where)"; bad=$((bad + 1)); fi
done < MANIFEST.tsv
echo "4 check: $((n - bad)) of $n outputs match MANIFEST.tsv"
[ "$bad" = 0 ]
