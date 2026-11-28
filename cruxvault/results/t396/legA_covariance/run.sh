#!/bin/bash
# t396 C15: leg A, collect and report. No arguments.
# Every step is skipped when its output exists. The records come from the 3 fit -> score chains
# of jobs.tsv (C7b; tools/t396/calib_cov.py, kit SC7b). `calib_cov.py table` runs on Fir (seconds,
# login node) and only its two small tables come down.
set -euo pipefail

KIT=/project/def-maxwl/mforooz/onion_kit/SC7b                       # calib_cov.py as on d03fff03
FIR_REC=/project/def-maxwl/mforooz/t396/legA_covariance/records       # records_seed{0,1,2}.jsonl
FIR_TAB=/project/def-maxwl/mforooz/t396/legA_covariance/table         # table output dir
SCORES=/project/def-maxwl/mforooz/t392/runs_both/rs_halves_mix_avg_pearson_unet/seed0/p16_sv/scores.json  # scorable assays
FIR_PY=/project/def-maxwl/mforooz/candi_venv/bin/python

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 1. calib_cov.py table on Fir (all_assays and scorable macros; rho.tsv)
if [ ! -f table.tsv ] || [ ! -f rho.tsv ]; then
  ssh -o BatchMode=yes fir "test -f $FIR_TAB/table.tsv || (cd $KIT && $FIR_PY tools/t396/calib_cov.py table \
    --records $FIR_REC/records_seed0.jsonl $FIR_REC/records_seed1.jsonl $FIR_REC/records_seed2.jsonl \
    --scores-json $SCORES --out $FIR_TAB)" > /dev/null
fi

# 2. pull the two tables
[ -f table.tsv ] || rsync -a -e 'ssh -o BatchMode=yes' "fir:$FIR_TAB/table.tsv" table.tsv
[ -f rho.tsv ] || rsync -a -e 'ssh -o BatchMode=yes' "fir:$FIR_TAB/rho.tsv" rho.tsv

echo "legA_covariance: $(($(wc -l < table.tsv) - 1)) table rows, $(($(wc -l < rho.tsv) - 1)) rho rows"
