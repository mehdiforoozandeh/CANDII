#!/bin/bash
# Notes of p-value and peaks from the predicted NB (close-out item E2). Rebuilds this folder:
#
#   bash runall.sh
#
# No arguments, no environment variables. The folder holds hand-written notes only (E2_RESULTS.md, FORMULA_EXTENSION.md),
# which are not outputs (AGENTS.md 4.11 rule 3), so MANIFEST.tsv has no output row and the script
# only checks it. The numbers in the notes are read from runs and tables of the item, named in
# the notes. Frozen: a finding cites this folder. A redo is a new task folder.
#
# Steps (no cluster call):
#   1. manifest   MANIFEST.tsv   missing -> write the header (no outputs)
#   2. check      always: re-hash every MANIFEST.tsv row, print MISMATCH, exit 1 on any
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

sha() {  # sha <file> -> hex digest (macOS shasum or GNU sha256sum)
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# 1. manifest
if [ ! -s MANIFEST.tsv ]; then
  printf 'path\tbytes\tsha256\twhere\n' > MANIFEST.tsv
  echo "1 manifest: wrote MANIFEST.tsv"
fi

# 2. check
bad=0; n=0
while IFS=$'\t' read -r path bytes want where; do
  [ "$path" = path ] && continue
  n=$((n + 1))
  if [ "$(sha "$path")" != "$want" ]; then echo "MISMATCH $path ($where)"; bad=$((bad + 1)); fi
done < MANIFEST.tsv
echo "2 check: $((n - bad)) of $n outputs match MANIFEST.tsv"
[ "$bad" = 0 ]
