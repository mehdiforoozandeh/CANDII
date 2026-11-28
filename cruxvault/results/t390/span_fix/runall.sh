#!/bin/bash
# t390 span_fix (SPAN_EXTRA 24 -> 25, the store count rule). Rebuilds this folder:
#
#   bash runall.sh
#
# No arguments, no environment variables. SPAN_FIX.md is hand-written prose (not an output). The
# TSVs are outputs of tools/eic_entry/collect_spanfix.py on small files pulled from Nibi
# /scratch/mforooz/eicx/spanfix (purge-exposed; arrays stay there), made by
#   ENTRY_ROOT=/scratch/mforooz/eicx/spanfix bash run_entry.sh test /scratch/mforooz/eicx/spanfix/entry <exp>
#   bash run_probe.sh sf <exp> <entry job>;  bash run_probe.sh sfid C19M22 <entry job>
# (jobs in SPAN_FIX.md). The "before" columns come from the committed tables of ../ladder_probe,
# ../moesm2 and ../entry_pipeline.
#
# Steps:
#   1. pull      any output missing -> copy the small files from Nibi into a temporary folder
#   2. collect   collect_spanfix.py -> the TSVs below
#   3. manifest  MANIFEST.tsv missing -> hash every output
#   4. check     always: re-hash every MANIFEST.tsv row, exit 1 on any mismatch
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=/Users/mforooz/miniforge3/envs/candii/bin/python
REPO=$(git rev-parse --show-toplevel)
REMOTE=/scratch/mforooz/eicx/spanfix
OUTS=(nmse.tsv placement.tsv combined.tsv checks.tsv resources.tsv)

sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
        else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

missing=0; for f in "${OUTS[@]}"; do [ -e "$f" ] || missing=1; done
if [ "$missing" = 1 ]; then
  TMP=$(mktemp -d)
  ssh -o BatchMode=yes nibi "cd $REMOTE && tar cf - qc scores/placement_qn_*.tsv scores/placement_se_*.tsv \
entry/board_moesm2.tsv entry/board_moesm4.tsv entry/check_*.tsv sacct.tsv" | tar xf - -C "$TMP" \
    || { echo "pull failed: run \`hpc up nibi\`"; exit 69; }
  echo "1 pull: $TMP"
  "$PY" "$REPO/tools/eic_entry/collect_spanfix.py" "$TMP" .
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
