#!/bin/bash
# t396 C10c: dedup cap test, collect, fit, score (h58). No arguments.
# Every step is skipped when its output exists. Pulls the small C10c outputs from Fir.
# The Fir side (kit SC10c): fir_job.sh fit (array) -> merge-fit -> slurm/t396/dedup_cap.sh
# floor-count (val pairs) -> fir_job.sh score (array). Job ids: jobs_c10c.tsv.
set -euo pipefail

FIR_OUT=/project/def-maxwl/mforooz/t396/dedup_cap   # fit_*.tsv, per_assay.tsv, pairs_*.tsv, totals_*.tsv
N_BOOT=1000                                          # bootstrap over experiments per table row

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
cd "$HERE"
PY="$(command -v python3 || command -v python)"
export PYTHONPATH="$REPO/src:$REPO"

# 1. pull the small outputs (per-pair deviances, totals, fits) from Fir
if [ ! -f fir/.pulled ]; then
  mkdir -p fir
  rsync -a -e 'ssh -o BatchMode=yes' --include='per_assay.tsv' --include='pairs_*.tsv' \
    --include='totals_*.tsv' --exclude='*' "fir:$FIR_OUT/" fir/
  touch fir/.pulled
fi

# 2. per_assay.tsv: one a (independent slots), one a_shared (shared weights) and one kappa per
#    assay and run type, train pairs, 2.5-97.5% bootstrap over train experiments
[ -f per_assay.tsv ] || cp fir/per_assay.tsv per_assay.tsv

# 3. totals.tsv: per pair, reads off / on and their ratio (tables), starts and counts (stores)
[ -f totals.tsv ] || "$PY" fit_score.py totals --totals fir/totals_*.tsv --out totals.tsv

# 4. table.tsv: gaps closed per law, space, layer, run type, assay class, in_eic, scope; gate (d)
[ -f table.tsv ] || "$PY" fit_score.py aggregate --pairs fir/pairs_*.tsv --n-boot "$N_BOOT" \
  --out table.tsv > /dev/null

echo "dedup_cap: $(($(wc -l < table.tsv) - 1)) table rows, $(($(wc -l < totals.tsv) - 1)) pairs"
