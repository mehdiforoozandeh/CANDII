#!/bin/bash
# t396 C14: leg B (h57), both models in three score spaces. No arguments.
# Every step is skipped when its output exists.
# The scoring records (24 jsonl, 3.8 MB in all) stay on Fir under $FIR_REC; they were written by
# the arrays of jobs.tsv (C7b, kit SC7b). Both steps below run tools/t396/score_spaces.py of
# kit SC7b on the Fir login node (numpy on small json lines, seconds) and pull the small outputs.
#   1. check  gate (b): overlap_count, regions all, scorable macro vs scores.json, bar 0.002;
#             fewer than 6 pass rows -> exit 1, no table is built
#   2. table  table.tsv, per_assay.tsv, per_read_length.tsv, truth_validation.tsv (gate c)
set -euo pipefail

KIT=/project/def-maxwl/mforooz/onion_kit/SC7b            # same score_spaces.py as d03fff03
FIR_PY=/project/def-maxwl/mforooz/candi_venv/bin/python   # numpy only
FIR_REC=/project/def-maxwl/mforooz/t396/legB_spaces/records
FIR_OUT=/project/def-maxwl/mforooz/t396/legB_spaces/report
N_RUNS=6                                                   # 2 models x 3 seeds

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

fir() {  # fir <command>: run on the Fir login node; ssh exit 255 = tunnel down
  local rc=0
  ssh -o BatchMode=yes fir "$1" || rc=$?
  if [ "$rc" = 255 ]; then echo "the tunnel to fir is down: run 'hpc up fir', then rerun" >&2; fi
  return "$rc"
}
REC="$FIR_REC/reference_seed[012]_[0-3].jsonl $FIR_REC/read_start_seed[012]_[0-3].jsonl"

# 1. gate (b)
if [ ! -f check.tsv ]; then
  rc=0
  fir "mkdir -p $FIR_OUT && cd $KIT && $FIR_PY tools/t396/score_spaces.py check \
       --records $REC --out $FIR_OUT/check.tsv 2>&1 | tail -1" || rc=$?
  [ "$rc" = 255 ] && exit 255
  rsync -a -e 'ssh -o BatchMode=yes' "fir:$FIR_OUT/check.tsv" check.tsv
fi
n_pass=$(awk -F'\t' 'NR > 1 && $NF == "pass"' check.tsv | wc -l | tr -d ' ')
if [ "$n_pass" -lt "$N_RUNS" ]; then
  echo "gate (b) failed: $n_pass of $N_RUNS runs pass; no table is built" >&2
  exit 1
fi

# 2. tables
if [ ! -f table.tsv ] || [ ! -f per_assay.tsv ] || [ ! -f per_read_length.tsv ] \
   || [ ! -f truth_validation.tsv ]; then
  fir "cd $KIT && $FIR_PY tools/t396/score_spaces.py table --records $REC --out $FIR_OUT \
       2>&1 | tail -1"
  rsync -a -e 'ssh -o BatchMode=yes' --include='table.tsv' --include='per_assay.tsv' \
    --include='per_read_length.tsv' --include='truth_validation.tsv' --exclude='*' \
    "fir:$FIR_OUT/" ./
fi

echo "legB_spaces: gate (b) $n_pass of $N_RUNS pass; $(($(wc -l < table.tsv) - 1)) table rows"
