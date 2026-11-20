# Replicate ladder (close-out item E4)

Stratum all, band all. Increments are assay-matched pair values (0.5 E[(Y - Y')^2] of log1p counts at each assay's common depth); intervals are 95% cluster-bootstrap intervals over cells. No row is pooled across assay classes. Full table: ladder.tsv.

## broad_histone

| rung | layer | increment [95% interval] | pairs (cells) matched | pairs (cells) all | share of cell-type pair value | flag |
|---|---|---|---|---|---|---|
| pseudoreplicate | sampling | 0.04092 [0.0366, 0.0452] | 59 (25) | 59 (25) | 0.928 | ok |
| sequencing | sequencing |  | 0 (0) | 0 (0) |  | no pair: the per-library and per-lane units are build priority 4 (PI approved all 722 rows on 2026-11-16; not built: Nibi /scratch over quota, Fir outage) |
| technical | library |  | 0 (0) | 0 (0) |  | no pair: the per-library and per-lane units are build priority 4 (PI approved all 722 rows on 2026-11-16; not built: Nibi /scratch over quota, Fir outage) |
| isogenic | biological replicate | 0.0003688 [6.94e-06, 0.000731] | 16 (16) | 16 (16) | 0.00836 | ok |
| anisogenic | donor |  | 0 (0) | 0 (0) |  | no pair in EIC for this assay class |
| other_experiment (experiment) | experiment repeat |  | 1 (1) | 1 (1) |  | not estimable: 1 pairs from 1 cells with a isogenic pair in the same cell and assay (rule: 5 pairs from 3 cells); all pairs of the row: 1 from 1 cells |
| other_experiment (experiment;lab) | lab |  | 2 (2) | 2 (2) |  | not estimable: 2 pairs from 2 cells with a isogenic pair in the same cell and assay (rule: 5 pairs from 3 cells); all pairs of the row: 2 from 2 cells |
| other_experiment (experiment;lab;donor) | lab with donor |  | 4 (4) | 4 (4) |  | not estimable: 4 pairs from 4 cells with a isogenic pair in the same cell and assay (rule: 5 pairs from 3 cells); all pairs of the row: 4 from 4 cells |
| other_experiment (experiment;donor) | experiment repeat with donor |  | 0 (0) | 1 (1) |  | not estimable: 0 pairs from 0 cells with a isogenic pair in the same cell and assay (rule: 5 pairs from 3 cells); all pairs of the row: 1 from 1 cells |
| other_cell_type | cell type (with donor and experiment) | 0.004489 [0.0016, 0.00738] | 188 (16) | 300 (24) | 0.102 | ok |

## Telescoping check of layers.tsv

0 of 75 groups telescope exactly (total_variance.telescoping_check); 0 have the ruling-30 chain. The rest hold rungs with different cells (telescoping.tsv).

## Crossed mixed model (order-free cross-check)

| assay | band | factor | variance fraction [95% interval] | levels | estimable | note |
|---|---|---|---|---|---|---|
| H3K36me3 | fine | cell | 1.01e-09 [7.45e-10, 1.28e-09] | 19 | 1 |  |
| H3K36me3 | fine | experiment | 1.55e-08 [4.73e-09, 2.62e-08] | 26 | 1 |  |
| H3K36me3 | fine | replicate | 0.00445 [0.00313, 0.00576] | 23 | 1 |  |
| H3K36me3 | fine | library | 2.62e-12 [2.36e-12, 2.89e-12] | 42 | 1 |  |
| H3K36me3 | fine | lab | 0.00194 [0, 0.0496] | 4 | 1 |  |
| H3K36me3 | fine | residual | 0.994 [0.946, 1] | 84 | 1 | REML (scipy L-BFGS-B, log variance components; mu and beta profiled); 20 cluster-bootstrap draws over 19 cells; layer uniform; 84 of 84 observations are pseudoreplicate halves |
| H3K36me3 | fine | log2_depth |  | 84 | 0 | every observation thinned to 3.84304e+06 reads before log1p; no depth slope fitted |

