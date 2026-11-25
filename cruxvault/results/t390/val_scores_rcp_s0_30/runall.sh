#!/bin/bash
# INTERIM validation board for miniCANDI rcp_combo_w2x_unet seed 0 (not recalibrated), mean mode: 30 of 42 predicted tracks.
# Fir 5706614-5706621, 5733708-5733709 (8 tracks), 5942490-5942550 + collect 5942551 (22 tracks), collect done 2026-11-25 02:29 UTC.
# Source: /scratch/mforooz/eicx/val/rcp_combo_w2x_unet_s0_raw on Fir (later collects overwrite it; this copy is the 30-track state).
# No arguments; skips files that exist.
set -euo pipefail
cd "$(dirname "$0")"
for f in SUMMARY.md board_validation.tsv JOBS.txt; do
  [ -s "interim30_$f" ] || hpc run fir "cat /scratch/mforooz/eicx/val/rcp_combo_w2x_unet_s0_raw/$f" | grep -v -iE "module\(s\)|nodejs|deprecated|recommend|at your own risk|^-{20,}$" | sed '/./,$!d' > "interim30_$f"
done
shasum -a 256 -c MANIFEST.tsv
