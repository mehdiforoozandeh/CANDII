#!/bin/bash
# Read-depth rule (fix b): observed reads thinned to fractions 1..0.125 through the mean-mode conversion, nMSE vs the
# challenge truth on chr9 + chr21 (tools/eic_entry/diag_depth.py; Fir 5965800-5965801, 5966954-5967209, 242 jobs, 2.98 CPU-h).
# The cap is chosen on the 212 TRAINING tracks only; the 30 validation tracks are a check. tools/eic_entry/depth_agg.py -> DEPTH_RULE.md.
# Source /scratch/mforooz/eicx/val/depth_rule on Fir. No arguments; skips files that exist.
set -euo pipefail
cd "$(dirname "$0")"
for f in DEPTH_RULE.md JOBS.txt; do
  [ -s "$f" ] || hpc run fir "cat /scratch/mforooz/eicx/val/depth_rule/$f" | grep -v -iE "module\(s\)|nodejs|deprecated|recommend|at your own risk|^-{20,}$" | sed '/./,$!d' > "$f"
done
shasum -a 256 -c MANIFEST.tsv
