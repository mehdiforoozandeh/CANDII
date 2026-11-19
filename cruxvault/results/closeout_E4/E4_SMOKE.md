# Close-out E4: smoke run of the replicate ladder (2026-11-17)

Nibi, kit COE4 (git a8d6e13), assay H3K36me3 only, tables `CO-D2/tables/eic_v1_2x`. Outputs: Nibi `/project/def-maxwl/mforooz/ONION/CO-E4/smoke/` (9 files) and `logs/` (4 files).

| job | id | elapsed | CPUs | MaxRSS |
|---|---|---|---|---|
| pair list | 27607315 | 0:47 | 1 | 2.9 GB |
| layers --pairs-out | 27607317 | 7:52 | 4 | 4.6 GB |
| mixed model, band fine, --second-experiments, lab, 20 draws | 27607318 | 1:57 | 2 | 3.2 GB |
| report | 27607319 | 0:52 | 1 | 2.9 GB |

Allocated CPU-h: 1x0.013 + 4x0.131 + 2x0.033 + 1x0.014 = 0.62.

## What the smoke shows

- The chain runs end to end. Pair list: 383 H3K36me3 pairs (59 pseudoreplicate added, 1 anisogenic dropped).
- Checks against t305 (`results/t305/LAYERS_REAL.md`, H3K36me3): sampling 0.04092 [0.0366, 0.0452] here against 0.0409 [0.0366, 0.0452] there; biological replicate (assay-matched) 0.000369 [0.0000069, 0.00073] against 0.0004 [-0.0000004, 0.0007]. They agree.
- One assay cannot fill the other-experiment rows (1 to 4 pairs each): they are "not estimable", as designed. In `layers.tsv` (not assay-matched, 4 pairs, smoke only) the lab-with-donor increment is -0.0027: no sign of a lab layer at 25 bp in this one assay. Not a result.
- Mixed model (H3K36me3, fine band): residual (pseudoreplicate halves) 0.994 [0.946, 1]; replicate (donor) 0.0044 [0.0031, 0.0058]; lab 0.0019 [0, 0.050]; cell 1e-9; experiment 2e-8. The cell fraction near 0 at the fine band, below the donor fraction, is not checked (the cell-type layer of t305 sits at the broad and domain scales); read the other bands before quoting.
- Telescoping (`total_variance.telescoping_check`): 0 of 75 groups pass. Every rung above sampling holds other cells than the rung below (flag "cells differ from lower rung"), so the increments use common cells only and the sum is not expected to telescope (the same cause as t305's failed groups). The acceptance check "telescoping passes" cannot hold on this pair set as the check is written.
