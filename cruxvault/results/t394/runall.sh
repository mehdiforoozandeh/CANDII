#!/bin/bash
# t394: why the read-start miniCANDI loses to the read-count version (low-compute checks). Rebuilds this
# folder:
#
#   bash runall.sh
#
# No arguments, no environment variables. Every step skips when its output exists, so a rerun of
# the finished folder only checks MANIFEST.tsv. Frozen once a finding cites it. A redo is a new
# task folder.
#
# Steps (cluster: Fir; this script writes only in this folder):
#   1. pull       each missing output -> rsync from $SRC on Fir
#   2. manifest   MANIFEST.tsv        missing -> hash every output
#   3. check      always: re-hash every MANIFEST.tsv row, print MISMATCH, exit 1 on any
# Producers:
#   conv_sparsity/: tools/t392/conv_sparsity.py (measure, model, link, summarize) on Fir, jobs 6026174,
#     6026439, 6026962, 6027889, 6027890, 6027896-98 (0.37 CPU-h); job scripts in t392 run_state/notes/investigate.
#   split/per_assay_deviance_split.tsv: per-assay macro of scores.json["deviance"], mean of seeds 0-2,
#     read from both models p16_dev folders on the Fir login node (no job).
# ssh exit 255 means the tunnel is down: run `hpc up fir`, then rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

H=fir
SRC=/project/def-maxwl/mforooz/t392/investigate
OUTPUTS=(conv_sparsity/summary_all/table_a.tsv conv_sparsity/summary_all/table_b.tsv conv_sparsity/summary_all/table_link.tsv conv_sparsity/summary_all/per_assay_scores_vs_measures.tsv conv_sparsity/summary_all/per_measurement.tsv conv_sparsity/summary_pair/table_a.tsv conv_sparsity/summary_pair/table_b.tsv conv_sparsity/summary_pair/per_measurement.tsv conv_sparsity/per_assay_scores.json split/per_assay_deviance_split.tsv)

sha() {  # sha <file> -> hex digest (macOS shasum or GNU sha256sum)
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# 1. pull
for f in "${OUTPUTS[@]}"; do
  [ -s "$f" ] && continue
  mkdir -p "$(dirname "$f")"
  rc=0; rsync -a -e 'ssh -o BatchMode=yes' "$H:$SRC/$f" "$f" || rc=$?
  if [ "$rc" = 255 ]; then echo "the tunnel to $H is down: run 'hpc up $H', then rerun" >&2; exit 255; fi
  [ "$rc" = 0 ] || { echo "rsync $SRC/$f failed" >&2; exit 1; }
  echo "1 pull: $f"
done

# 2. manifest
if [ ! -s MANIFEST.tsv ]; then
  {
    printf 'path\tbytes\tsha256\twhere\n'
    for f in "${OUTPUTS[@]}"; do
      printf '%s\t%s\t%s\tgit\n' "$f" "$(wc -c < "$f" | tr -d ' ')" "$(sha "$f")"
    done
  } > MANIFEST.tsv.tmp
  mv MANIFEST.tsv.tmp MANIFEST.tsv
  echo "2 manifest: wrote MANIFEST.tsv"
fi

# 3. check
bad=0; n=0
while IFS=$'\t' read -r path bytes want where; do
  [ "$path" = path ] && continue
  n=$((n + 1))
  if [ "$(sha "$path")" != "$want" ]; then echo "MISMATCH $path ($where)"; bad=$((bad + 1)); fi
done < MANIFEST.tsv
echo "3 check: $((n - bad)) of $n outputs match MANIFEST.tsv"
[ "$bad" = 0 ]
