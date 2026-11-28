#!/bin/bash
# Expected-p validation board of the wide-grid recalibrated copy (rcp_combo_w2x_unet seed 0, 8 tracks).
# Fir jobs 5763739-5763746, collect 5763747; out /scratch/mforooz/eicx/val/rcp_combo_w2x_unet_s0_recalwide_exp.
# Its raw partner (same mode) is ../val_scores_rcp_s0_8/raw_exp_*. No arguments; skips files that exist.
set -euo pipefail
cd "$(dirname "$0")"
for f in SUMMARY.md board_validation.tsv JOBS.txt; do
  [ -s "recalwide_exp_$f" ] || hpc run fir "cat /scratch/mforooz/eicx/val/rcp_combo_w2x_unet_s0_recalwide_exp/$f" | grep -v -iE "module\(s\)|nodejs|deprecated|recommend|at your own risk|^-{20,}$" | sed '/./,$!d' > "recalwide_exp_$f"
done
shasum -a 256 -c MANIFEST.tsv
