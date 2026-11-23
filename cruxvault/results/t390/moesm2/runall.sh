#!/bin/bash
# t390 MOESM2 board (original 2019 paired-end blind truth, no normalization). Rebuilds this folder:
#
#   bash runall.sh
#
# No arguments, no environment variables. MOESM2.md is hand-written prose (not an output). The TSVs
# are outputs of tools/eic_entry/moesm2.py collect on small files pulled from Nibi:
#   /scratch/mforooz/eicx/moesm2/{scores,verify,qc,logs,sacct.tsv} and score_cache/pe/scores/team_*
#   (made by tools/eic_entry/run_moesm2.sh submit; jobs in MOESM2.md), and Max's cached pe scores
#   /project/def-maxwl/maxwl/epi_imputation/s04/scores/pe/<entrant>/<exp>.csv (read-only).
# The published table (1.99 MB, not committed) is read on Nibi at
#   /project/def-maxwl/mforooz/ONION/eic/published/13059_2023_2915_MOESM2_ESM.csv (sha256 pinned in
#   tools/onion_eval/eic_score.py MOESM2_SHA256).
#
# Steps:
#   1. pull      any output missing -> copy the small files from Nibi into a temporary folder
#   2. collect   moesm2.py collect -> the TSVs below
#   3. manifest  MANIFEST.tsv missing -> hash every output
#   4. check     always: re-hash every MANIFEST.tsv row, exit 1 on any mismatch
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=/Users/mforooz/miniforge3/envs/candii/bin/python
REPO=$(git rev-parse --show-toplevel)
OUTS=(repro_max.tsv repro_ours.tsv board_check.tsv scores_all23.tsv placement.tsv
      placement_combined.tsv checks.tsv resources.tsv)
SSH=(ssh -o BatchMode=yes nibi)

sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
        else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

missing=0; for f in "${OUTS[@]}"; do [ -e "$f" ] || missing=1; done
if [ "$missing" = 1 ]; then
  TMP=$(mktemp -d)
  mkdir -p "$TMP/max_pe" "$TMP/cache_scores"
  { "${SSH[@]}" 'cd /scratch/mforooz/eicx/moesm2 && tar cf - scores verify qc logs sacct.tsv' | tar xf - -C "$TMP" &&
    "${SSH[@]}" 'cd /scratch/mforooz/eicx/moesm2/score_cache/pe/scores && tar cf - team_*' | tar xf - -C "$TMP/cache_scores" &&
    "${SSH[@]}" 'cd /project/def-maxwl/maxwl/epi_imputation/s04/scores/pe && tar cf - */C19M22.csv */C40M17.csv */C12M02.csv' | tar xf - -C "$TMP/max_pe" &&
    "${SSH[@]}" 'cat /project/def-maxwl/mforooz/ONION/eic/published/13059_2023_2915_MOESM2_ESM.csv' > "$TMP/MOESM2.csv"; } \
    || { echo "pull failed: run \`hpc up nibi\`"; exit 69; }
  echo "1 pull: $TMP"
  "$PY" "$REPO/tools/eic_entry/moesm2.py" collect --src "$TMP" --dst . --published "$TMP/MOESM2.csv"
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
