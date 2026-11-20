#!/bin/bash
# Read of the 16 miniCANDI ablations (close-out): the two arm-summary tables behind
# ABLATIONS_READ.md. Rebuilds this folder:
#
#   bash runall.sh
#
# No arguments, no environment variables. Every step skips when its output exists, so a rerun of
# the finished folder only checks MANIFEST.tsv. Frozen: findings cite ABLATIONS_READ.md. A redo is
# a new task folder.
#
# Steps (cluster: Fir; this script writes only in this folder):
#   1. sv table       arm_summary_p16_sv_2026-11-17.tsv       missing -> rsync the dated Fir copy
#   2. foreign table  arm_summary_p16_foreign_2026-11-17.tsv  missing -> print the command, stop
#   3. manifest       MANIFEST.tsv                            missing -> hash every output
#   4. check          always: re-hash every MANIFEST.tsv row, print MISMATCH, exit 1 on any
# Both tables are snapshots of `select_rule.py arm-summary` over runs_fir on 2026-11-17. A rerun
# today reads the arms scored since then, so it is not the same table; the dated copy is the
# output. The foreign table has no recorded Fir copy (ABLATIONS_READ.md, "Sources"), so a missing
# copy stops the script and names the command (kit CO6).
# Hand-written files (ABLATIONS_READ.md) are not outputs.
# ssh exit 69 or 255 means the tunnel is down: run `hpc up fir`, then rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

H=fir
RUNS=/project/def-maxwl/mforooz/ONION/runs_fir
K=/project/def-maxwl/mforooz/onion_kit/CO6
SV=arm_summary_p16_sv_2026-11-17.tsv
SV_FIR=$RUNS/ARM_SUMMARY_p16_sv_2026-11-17.txt
FOREIGN=arm_summary_p16_foreign_2026-11-17.tsv

remote() {  # remote <command>; stops the script on a dead tunnel
  local rc=0
  ssh -o BatchMode=yes "$H" "$1" || rc=$?
  if [ "$rc" = 69 ] || [ "$rc" = 255 ]; then
    echo "ssh $H exited $rc: the tunnel is down. Run 'hpc up $H', then rerun." >&2
    exit "$rc"
  fi
  return "$rc"
}

sha() {  # sha <file> -> hex digest (macOS shasum or GNU sha256sum)
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# 1. sv table
if [ -s "$SV" ]; then echo "1 sv table: have $SV"
else
  remote "test -s $SV_FIR" || { echo "$SV_FIR is missing; the 2026-11-17 snapshot is lost" >&2; exit 1; }
  rsync -a -e 'ssh -o BatchMode=yes' "$H:$SV_FIR" "$SV" || { echo "rsync $SV_FIR failed" >&2; exit 1; }
  echo "1 sv table: pulled $SV"
fi

# 2. foreign table
if [ -s "$FOREIGN" ]; then echo "2 foreign table: have $FOREIGN"
else
  echo "2 foreign table: $FOREIGN is missing; on Fir run" \
       "PYTHONPATH=$K/src:$K/tools python $K/tools/onion_eval/select_rule.py arm-summary" \
       "--runs $RUNS --sub p16_foreign_<mode> for each foreign mode (it reads today's runs)" >&2
  exit 1
fi

# 3. manifest
if [ ! -s MANIFEST.tsv ]; then
  {
    printf 'path\tbytes\tsha256\twhere\n'
    for f in "$SV" "$FOREIGN"; do
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
  if [ "$where" = git ]; then got=$(sha "$path")
  else got=$(remote "sha256sum ${where#*:} | cut -d' ' -f1"); fi
  if [ "$got" != "$want" ]; then echo "MISMATCH $path ($where)"; bad=$((bad + 1)); fi
done < MANIFEST.tsv
echo "4 check: $((n - bad)) of $n outputs match MANIFEST.tsv"
[ "$bad" = 0 ]
