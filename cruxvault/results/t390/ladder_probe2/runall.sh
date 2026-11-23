#!/bin/bash
# t390 ladder probe 2 (rungs L7t, L7o, L6f, L7f). Rebuilds this folder:
#
#   bash runall.sh
#
# No arguments, no environment variables. PROBE2.md is hand-written prose (not an output). The six
# TSVs are outputs of tools/eic_entry/collect2.py on small files pulled from Nibi
# /scratch/mforooz/eicx/ladder (scores2/, qc/, peaks/*.json, sacct2.tsv), plus probe 1's
# ../ladder_probe/*.tsv. The arrays and the jobs that made them are listed in PROBE2.md; they were
# made by tools/eic_entry/run_probe.sh p2 <exp>.
#
# Steps:
#   1. pull      any output missing -> rsync the small files from Nibi into a temporary folder
#   2. collect   collect2.py -> {scores_all23,placement,rung_compare,checks,peaks,resources}.tsv
#   3. manifest  MANIFEST.tsv missing -> hash every output
#   4. check     always: re-hash every MANIFEST.tsv row, exit 1 on any mismatch
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=/Users/mforooz/miniforge3/envs/candii/bin/python
REPO=$(git rev-parse --show-toplevel)
REMOTE=nibi:/scratch/mforooz/eicx/ladder/
OUTS=(scores_all23.tsv placement.tsv rung_compare.tsv checks.tsv peaks.tsv resources.tsv)

sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
        else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

missing=0; for f in "${OUTS[@]}"; do [ -e "$f" ] || missing=1; done
if [ "$missing" = 1 ]; then
  TMP=$(mktemp -d)
  rsync -a -e 'ssh -o BatchMode=yes' --include='*/' --include='scores2/*/*.tsv' --include='scores2/*.tsv' \
    --include='qc/*.json' --include='peaks/*.json' --include='sacct2.tsv' \
    --exclude='code/' --exclude='score_cache/' --exclude='rungs/' --exclude='rungs2/' --exclude='*' "$REMOTE" "$TMP/" \
    || { echo "rsync failed: run \`hpc up nibi\`"; exit 69; }
  echo "1 pull: $TMP"
  "$PY" "$REPO/tools/eic_entry/collect2.py" "$TMP" ../ladder_probe . > /dev/null
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
