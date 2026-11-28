#!/bin/bash
# t390 post-prediction pileup probe (variants A, B, C, D of `ladder_probe.py postpile_route`). Rebuilds
# this folder:
#
#   bash runall.sh
#
# No arguments, no environment variables. POSTPILE.md is hand-written prose (not an output).
# postpile.tsv is the output of tools/eic_entry/collect_postpile.py on small files pulled from Nibi
# /scratch/mforooz/eicx/ladder (qc/*.json, postpile/*/*/*.npy.json). The arrays and the jobs that made
# them are listed in POSTPILE.md; they were made by `CHECK=1 tools/eic_entry/run_probe.sh pp <exp>`.
#
# Steps:
#   1. pull      postpile.tsv missing -> rsync the small files from Nibi into a temporary folder
#   2. collect   collect_postpile.py -> postpile.tsv
#   3. manifest  MANIFEST.tsv missing -> hash every output
#   4. check     always: re-hash every MANIFEST.tsv row, exit 1 on any mismatch
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=/Users/mforooz/miniforge3/envs/candii/bin/python
REPO=$(git rev-parse --show-toplevel)
REMOTE=nibi:/scratch/mforooz/eicx/ladder/
OUTS=(postpile.tsv)

sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
        else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

missing=0; for f in "${OUTS[@]}"; do [ -e "$f" ] || missing=1; done
if [ "$missing" = 1 ]; then
  TMP=$(mktemp -d)
  rsync -a -e 'ssh -o BatchMode=yes' --include='*/' --include='qc/*.json' --include='postpile/*/*/*.npy.json' \
    --exclude='code/' --exclude='score_cache/' --exclude='rungs/' --exclude='rungs2/' --exclude='*' "$REMOTE" "$TMP/" \
    || { echo "rsync failed: run \`hpc up nibi\`"; exit 69; }
  echo "1 pull: $TMP"
  "$PY" "$REPO/tools/eic_entry/collect_postpile.py" "$TMP" postpile.tsv > /dev/null
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
