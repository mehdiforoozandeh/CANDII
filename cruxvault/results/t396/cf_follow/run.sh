#!/bin/bash
# t396 it2 B6: follow test of the existing models on the 14 controlled pairs x 2 directions.
# No arguments. Every step is skipped when its output exists.
#   1. pull: fir/{table,per_track}.tsv from Fir. The Fir side (kit SC2B4, job 6939785) ran
#        cf_follow.sh <kit> table --pred <6 prediction dirs> --caps-table cf_caps_table.tsv
#      on the predictions of arrays 6931724-29 (jobs.tsv) under /scratch/mforooz/sc396/it2/cf_follow/
#      {rcp_combo_w2x_unet,read-start_80000_windows_unequal}/seed{0,1,2}; cf_caps_table.tsv is
#      ../cf_caps/table.tsv copied to Fir (sha256 checked below).
#   2. table.tsv, per_track.tsv: the Fir files without their leading "# stated limit" comment line,
#      so that line 1 is the column header. The limit itself is stated in CF_FOLLOW.md.
# ssh exit 255 means the tunnel is down: run `hpc up fir`, then rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
FIR_OUT=/project/def-maxwl/mforooz/t396/cf_follow

if [ ! -f fir/.pulled ]; then
  mkdir -p fir
  rc=0
  rsync -a -e 'ssh -o BatchMode=yes' "fir:$FIR_OUT/table/table.tsv" "fir:$FIR_OUT/table/per_track.tsv" \
    fir/ || rc=$?
  if [ "$rc" = 255 ]; then echo "the tunnel to fir is down: run 'hpc up fir', then rerun" >&2; exit 255; fi
  [ "$rc" = 0 ] || { echo "rsync failed ($rc)" >&2; exit 1; }
  rc=0
  remote=$(ssh -n -o BatchMode=yes fir "sha256sum '$FIR_OUT/cf_caps_table.tsv'" 2>/dev/null) || rc=$?
  if [ "$rc" = 255 ]; then echo "the tunnel to fir is down: run 'hpc up fir', then rerun" >&2; exit 255; fi
  if command -v sha256sum >/dev/null; then local_sha=$(sha256sum ../cf_caps/table.tsv | cut -d' ' -f1)
  else local_sha=$(shasum -a 256 ../cf_caps/table.tsv | cut -d' ' -f1); fi
  [ "${remote%% *}" = "$local_sha" ] || { echo "Fir caps table differs from ../cf_caps/table.tsv" >&2; exit 1; }
  touch fir/.pulled
fi

for f in table.tsv per_track.tsv; do
  [ -s "$f" ] || grep -v '^#' "fir/$f" > "$f"
done
echo "cf_follow: $(($(wc -l < table.tsv) - 1)) table rows, $(($(wc -l < per_track.tsv) - 1)) per-track rows"
