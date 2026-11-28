#!/bin/bash
# Scale-error diagnosis of the 30 scored validation tracks (miniCANDI rcp_combo_w2x_unet seed 0, mean mode),
# chr1 + chr9 + chr21: tools/eic_entry/diag_scale.py per track (Fir 5964408-5964409, 5964853-5964880),
# tools/eic_entry/diag_agg.py -> DIAG_TABLE.tsv, DIAG_BY_ASSAY.tsv. Source /scratch/mforooz/eicx/val/rcp_combo_w2x_unet_s0_raw on Fir.
# No arguments; skips files that exist.
set -euo pipefail
cd "$(dirname "$0")"
for f in DIAG_TABLE.tsv DIAG_BY_ASSAY.tsv DIAG_JOBS.txt; do
  [ -s "$f" ] || hpc run fir "cat /scratch/mforooz/eicx/val/rcp_combo_w2x_unet_s0_raw/$f" | grep -v -iE "module\(s\)|nodejs|deprecated|recommend|at your own risk|^-{20,}$" | sed '/./,$!d' > "$f"
done
shasum -a 256 -c MANIFEST.tsv
