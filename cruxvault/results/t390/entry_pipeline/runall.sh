#!/bin/bash
# t390 entry pipeline test (observed counts in place of miniCANDI predictions). Rebuilds this folder:
#
#   bash runall.sh
#
# No arguments, no environment variables. ENTRY_PIPELINE.md is hand-written prose (not an output).
# The outputs below are small files written on Nibi by tools/eic_entry/run_entry.sh test (jobs in
# ENTRY_PIPELINE.md) under /scratch/mforooz/eicx/entry/test (purge-exposed); the arrays stay there.
#
# Steps:
#   1. pull      any output missing -> copy the small files from Nibi
#   2. manifest  MANIFEST.tsv missing -> hash every output
#   3. check     always: re-hash every MANIFEST.tsv row, exit 1 on any mismatch
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REMOTE=/scratch/mforooz/eicx/entry/test
OUTS=(board_moesm2.tsv board_moesm4.tsv SUMMARY.md check_C19M22.tsv check_C40M17.tsv check_C12M02.tsv
      sacct.tsv)

sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
        else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

missing=0; for f in "${OUTS[@]}"; do [ -e "$f" ] || missing=1; done
if [ "$missing" = 1 ]; then
  ssh -o BatchMode=yes nibi "cd $REMOTE && tar cf - ${OUTS[*]}" | tar xf - \
    || { echo "pull failed: run \`hpc up nibi\`"; exit 69; }
  echo "1 pull: ${OUTS[*]}"
fi

if [ ! -s MANIFEST.tsv ]; then
  printf 'path\tbytes\tsha256\twhere\n' > MANIFEST.tsv
  for f in "${OUTS[@]}"; do printf '%s\t%s\t%s\tgit\n' "$f" "$(wc -c < "$f" | tr -d ' ')" "$(sha "$f")" >> MANIFEST.tsv; done
  echo "2 manifest: wrote MANIFEST.tsv"
fi

bad=0; n=0
while IFS=$'\t' read -r path bytes want where; do
  [ "$path" = path ] && continue
  n=$((n + 1))
  if [ "$(sha "$path")" != "$want" ]; then echo "MISMATCH $path ($where)"; bad=$((bad + 1)); fi
done < MANIFEST.tsv
echo "3 check: $((n - bad)) of $n outputs match MANIFEST.tsv"
[ "$bad" = 0 ]
