#!/bin/bash
# Close-out D2: add-on tables of eic_v1_2 with the second experiments (prep). Rebuilds this folder:
#
#   bash runall.sh
#
# No arguments, no environment variables. Every step skips when its output exists, so a rerun of
# the finished folder only checks MANIFEST.tsv. Frozen: a finding cites this folder. A redo is a
# new task folder.
#
# Steps (cluster: Nibi; this script writes only in this folder):
#   1. pull       each missing output -> rsync from $SRC on Nibi
#   2. manifest   MANIFEST.tsv        missing -> hash every output
#   3. check      always: re-hash every MANIFEST.tsv row, print MISMATCH, exit 1 on any
# The output is the log of Nibi job 27601259 (kit COD2, out /project/def-maxwl/mforooz/ONION/CO-D2, per the log header). The log's own Nibi path is not recorded; $SRC is the item root.
# Hand-written files (D2_PREP.md) are not outputs.
# ssh exit 69 or 255 means the tunnel is down: run `hpc up nibi`, then rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

H=nibi
SRC=/project/def-maxwl/mforooz/ONION/CO-D2
OUTPUTS=(co_d2_prep_27601259.log)

sha() {  # sha <file> -> hex digest (macOS shasum or GNU sha256sum)
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# 1. pull
for f in "${OUTPUTS[@]}"; do
  [ -s "$f" ] && continue
  rc=0; rsync -a -e 'ssh -o BatchMode=yes' "$H:$SRC/$f" . || rc=$?
  if [ "$rc" = 255 ]; then echo "the tunnel to $H is down: run 'hpc up $H', then rerun" >&2; exit 255; fi
  [ "$rc" = 0 ] || { echo "rsync $SRC/$f failed; rerun the job named in D2_PREP.md" >&2; exit 1; }
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
