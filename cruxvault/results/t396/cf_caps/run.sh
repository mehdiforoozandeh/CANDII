#!/bin/bash
# t396 it2 B5: cap tests on the 14 controlled pairs (7 dedup, 7 run type). No arguments.
# Every step is skipped when its output exists.
#   1. pull: fir/<tool>/<scenario>/{rows,per_assay,pairs,table}.tsv from Fir. The Fir side is
#      fir_job.sh (kit SC2B5): fit -> floor-count -> score per tool and rate scenario, on the
#      statistics files of B3 (/scratch/mforooz/sc396/it2/cf_caps/{dedup,sepe}). Job ids: jobs.tsv.
#   2. assemble.py -> table.tsv, per_assay.tsv, per_track.tsv, q_sensitivity.tsv.
# ssh exit 255 means the tunnel is down: run `hpc up fir`, then rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
FIR_OUT=/project/def-maxwl/mforooz/t396/cf_caps
N_SCEN=7                                   # scenarios per tool (fitted + 6 fixed rho)
PY="$(command -v python3 || command -v python)"

if [ ! -f fir/.pulled ]; then
  mkdir -p fir
  rc=0
  rsync -a -e 'ssh -o BatchMode=yes' --include='*/' --include='rows.tsv' --include='per_assay.tsv' \
    --include='pairs.tsv' --include='table.tsv' --exclude='*' "fir:$FIR_OUT/" fir/ || rc=$?
  if [ "$rc" = 255 ]; then echo "the tunnel to fir is down: run 'hpc up fir', then rerun" >&2; exit 255; fi
  [ "$rc" = 0 ] || { echo "rsync failed ($rc)" >&2; exit 1; }
  find fir -type d -empty -delete
  for t in dedup sepe; do
    n=$(ls fir/$t/*/table.tsv 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" = "$N_SCEN" ] || { echo "fir/$t: $n of $N_SCEN scenario tables" >&2; exit 1; }
  done
  touch fir/.pulled
fi

need=0
for f in table.tsv per_assay.tsv per_track.tsv q_sensitivity.tsv; do [ -s "$f" ] || need=1; done
[ "$need" = 0 ] || "$PY" assemble.py
echo "cf_caps: $(($(wc -l < table.tsv) - 1)) table rows, $(($(wc -l < per_track.tsv) - 1)) per-track rows"
