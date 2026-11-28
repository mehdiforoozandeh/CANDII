#!/bin/bash
# Pull the validation board for miniCANDI rcp_combo_w2x_unet seed 0 (uncalibrated, mean mode) from Fir.
# Produced by: BOARD=validation run_entry.sh submit .../runs_rcp/eic_val/rcp_combo_w2x_unet_s0 /scratch/mforooz/eicx/val/cov.tsv /scratch/mforooz/eicx/val/rcp_combo_w2x_unet_s0_raw <7 tracks>
# Fir jobs 5706614-5706620 (score), 5706621 (collect). No arguments; skips files that exist.
set -euo pipefail
cd "$(dirname "$0")"
for f in SUMMARY.md board_validation.tsv JOBS.txt; do
  [ -s "$f" ] || hpc run fir "cat /scratch/mforooz/eicx/val/rcp_combo_w2x_unet_s0_raw/$f" | sed -n '/^[#a-zA-Z|C0-9]/,$p' > "$f"
done
shasum -a 256 -c MANIFEST.tsv
