#!/bin/bash
# t392 read-start miniCANDI: side-by-side with the read-count reference on 16 regions (eic_v1_2).
# Rebuilds this folder:
#
#   bash runall.sh
#
# No arguments, no environment variables. Every step skips when its output exists, so a rerun of
# the finished folder only checks MANIFEST.tsv. Frozen once a finding cites it. A redo is a new
# task folder. SIDE_BY_SIDE.md is hand-written and is not an output.
#
# Steps (cluster: Fir; writes only in this folder and in $OUT on Fir):
#   1. build      $OUT/side_by_side.png missing -> copy plot.py to $OUT, run it on Fir (login node,
#                 reads score JSONs only, about 10 s)
#   2. pull       each missing output -> rsync from $OUT on Fir
#   3. manifest   MANIFEST.tsv missing -> hash plot.py and every output
#   4. check      always: re-hash every MANIFEST.tsv row, print MISMATCH, exit 1 on any
# Inputs read, owned by these jobs (rerun t392 if missing; never rebuilt here):
#   read-start: $RUNS_BOTH/rs_halves_mix_avg_pearson_unet/seed{0,1,2}: train 5691700/01/02 (kit RS4),
#     p16_sv predict 5874564/65/67/68, 5874996-99, 5875002-05 and metrics 5874569, 5875000, 5875006,
#     p16_dev 5875186/87/88, sanity_p16 5808635, 5851878, 5851884 (scoring kit RS7, git fe138144).
#   reference: $RUNS_BOTH/rcp_halves_mix_avg_pearson_unet -> ONION/runs_rcp (task t370; p16_sv and
#     p16_dev reused, never rerun here).
# ssh exit 255 means the tunnel is down: run `hpc up fir`, then rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

H=fir
PY=/project/def-maxwl/mforooz/candi_venv/bin/python          # matplotlib 3.10.0
KIT=/project/def-maxwl/mforooz/onion_kit/RS7                   # git fe138144
RUNS_BOTH=/project/def-maxwl/mforooz/t392/runs_both
OUT=/project/def-maxwl/mforooz/t392/side_by_side
OUTPUTS=(per_seed.tsv table.tsv sanity.tsv open_edges.tsv side_by_side.png)

sha() {  # sha <file> -> hex digest (macOS shasum or GNU sha256sum)
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
remote() {  # remote <cmd>: run on Fir; exit 255 = tunnel down
  local rc=0; ssh -o BatchMode=yes "$H" "$1" || rc=$?
  if [ "$rc" = 255 ]; then echo "the tunnel to $H is down: run 'hpc up $H', then rerun" >&2; exit 255; fi
  return "$rc"
}

# 1. build
need=0; for f in "${OUTPUTS[@]}"; do [ -s "$f" ] || need=1; done
if [ "$need" = 1 ] && ! remote "test -s $OUT/side_by_side.png"; then
  remote "mkdir -p $OUT"
  rsync -a -e 'ssh -o BatchMode=yes' plot.py "$H:$OUT/plot.py"
  remote "cd $OUT && $PY plot.py $KIT $RUNS_BOTH $OUT"
  echo "1 build: ran plot.py on $H"
fi

# 2. pull
for f in "${OUTPUTS[@]}"; do
  [ -s "$f" ] && continue
  rc=0; rsync -a -e 'ssh -o BatchMode=yes' "$H:$OUT/$f" "$f" || rc=$?
  if [ "$rc" = 255 ]; then echo "the tunnel to $H is down: run 'hpc up $H', then rerun" >&2; exit 255; fi
  [ "$rc" = 0 ] || { echo "rsync $OUT/$f failed" >&2; exit 1; }
  echo "2 pull: $f"
done

# 3. manifest
if [ ! -s MANIFEST.tsv ]; then
  {
    printf 'path\tbytes\tsha256\twhere\n'
    for f in plot.py "${OUTPUTS[@]}"; do
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
