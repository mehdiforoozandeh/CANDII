#!/bin/bash
# Validation boards for miniCANDI rcp_combo_w2x_unet seed 0 (not recalibrated), all 8 predicted tracks.
# raw_*: mean mode (Fir 5706614-5706621, 5733708-5733709). raw_exp_*: expected-p mode (Fir 5733710-5733849, collect 5733850).
# Source: /scratch/mforooz/eicx/val/rcp_combo_w2x_unet_s0_{raw,raw_exp} on Fir. No arguments; skips files that exist.
set -euo pipefail
cd "$(dirname "$0")"
for m in raw raw_exp; do for f in SUMMARY.md board_validation.tsv JOBS.txt; do
  [ -s "${m}_$f" ] || hpc run fir "cat /scratch/mforooz/eicx/val/rcp_combo_w2x_unet_s0_$m/$f" | grep -v -iE "module\(s\)|nodejs|deprecated|recommend|at your own risk|^-{20,}$" | sed '/./,$!d' > "${m}_$f"
done; done
shasum -a 256 -c MANIFEST.tsv
