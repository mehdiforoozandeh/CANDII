#!/bin/bash
# t395: smoothing test of the read-start miniCANDI (seed 0) -- start-space means box-smoothed at
# W = 1, 9, 41 before conversion; residual correlation length; calibration ratio after n -> n / c.
# Rebuilds this folder:
#
#   bash runall.sh
#
# No arguments, no environment variables. Every step skips when its output exists, so a rerun of
# the finished folder only checks MANIFEST.tsv. Frozen once a finding cites it. A redo is a new
# task folder. SMOOTHING.md is hand-written and is not an output.
#
# Steps (cluster: Fir; this script writes only in this folder):
#   1. pull       each missing out/ file -> rsync from $SRC on Fir
#   2. table      summary/table_skill.tsv missing -> tools/t392/smooth_summary.py skill (laptop, seconds)
#   3. manifest   MANIFEST.tsv missing -> hash every output
#   4. check      always: re-hash every MANIFEST.tsv row, print MISMATCH, exit 1 on any
# Producers (rerun t395 if the Fir files are gone; never rebuilt here):
#   out/live_*: tools/t392/smooth_skill.py on kit RS7 (git fe138144), jobs 6033442 (probe), 6034117-19.
#   out/store_all: the same script on the stored seed-0 predictions, job 6034120.
#   out/calib_seed0*: smooth_summary.py calib + table, job 6040094. Total 12,704 CPU-s = 3.53 CPU-h.
# Inputs: ../t394/conv_sparsity/per_assay_scores.json (tracked, frozen).
# ssh exit 255 means the tunnel is down: run `hpc up fir`, then rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

H=fir
SRC=/project/def-maxwl/mforooz/t392/investigate/smooth
REPO=$(cd ../../.. && pwd)
PY=/Users/mforooz/miniforge3/envs/candii/bin/python
ASSAYS=(H3K27ac H3K36me3 H3K4me3)
PULL=(out/store_all/records.jsonl out/calib_seed0.jsonl out/calib_seed0_table.json)
for a in "${ASSAYS[@]}"; do PULL+=("out/live_$a/records.jsonl" "out/live_$a/resid.jsonl"); done
OUTPUTS=("${PULL[@]}" summary/table_skill.tsv)

sha() {  # sha <file> -> hex digest (macOS shasum or GNU sha256sum)
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# 1. pull
for f in "${PULL[@]}"; do
  [ -s "$f" ] && continue
  mkdir -p "$(dirname "$f")"
  rc=0; rsync -a -e 'ssh -o BatchMode=yes' "$H:$SRC/$f" "$f" || rc=$?
  if [ "$rc" = 255 ]; then echo "the tunnel to $H is down: run 'hpc up $H', then rerun" >&2; exit 255; fi
  [ "$rc" = 0 ] || { echo "rsync $SRC/$f failed" >&2; exit 1; }
  echo "1 pull: $f"
done

# 2. table
if [ ! -s summary/table_skill.tsv ]; then
  recs=(); res=()
  for a in "${ASSAYS[@]}"; do recs+=("out/live_$a/records.jsonl"); res+=("out/live_$a/resid.jsonl"); done
  PYTHONPATH="$REPO/src" "$PY" "$REPO/tools/t392/smooth_summary.py" skill \
    --records "${recs[@]}" out/store_all/records.jsonl --resid "${res[@]}" \
    --scores ../t394/conv_sparsity/per_assay_scores.json --out summary >/dev/null
  echo "2 table: summary/table_skill.tsv"
fi

# 3. manifest
if [ ! -s MANIFEST.tsv ]; then
  {
    printf 'path\tbytes\tsha256\twhere\n'
    for f in "${OUTPUTS[@]}"; do
      printf '%s\t%s\t%s\tgit\n' "$f" "$(wc -c < "$f" | tr -d ' ')" "$(sha "$f")"
    done
  } > MANIFEST.tsv.tmp
  mv MANIFEST.tsv.tmp MANIFEST.tsv
  echo "3 manifest: wrote MANIFEST.tsv"
fi

# 4. check
bad=0; n=0
while IFS=$'\t' read -r path bytes want where; do
  [ "$path" = path ] && continue
  n=$((n + 1))
  if [ "$(sha "$path")" != "$want" ]; then echo "MISMATCH $path ($where)"; bad=$((bad + 1)); fi
done < MANIFEST.tsv
echo "4 check: $((n - bad)) of $n outputs match MANIFEST.tsv"
[ "$bad" = 0 ]
