# Replicate ladder (close-out item E4)

Stratum all, band all. Increments are assay-matched pair values (0.5 E[(Y - Y')^2] of log1p counts at each assay's common depth); intervals are 95% cluster-bootstrap intervals over cells. No row is pooled across assay classes. Full table: ladder.tsv.

## broad_histone

| rung | layer | increment [95% interval] | pairs (cells) matched | pairs (cells) all | share of the cell-type pair value (same cells and assays) | flag |
|---|---|---|---|---|---|---|
| pseudoreplicate | sampling | 0.04092 [0.0366, 0.0452] | 59 (25) | 59 (25) | 0.927 | ok |
| sequencing | sequencing |  | 0 (0) | 0 (0) |  | no pair: the per-library and per-lane units are build priority 4 (PI approved all 722 rows on 2026-11-16; not built: Nibi /scratch over quota, Fir outage) |
| technical | library |  | 0 (0) | 0 (0) |  | no pair: the per-library and per-lane units are build priority 4 (PI approved all 722 rows on 2026-11-16; not built: Nibi /scratch over quota, Fir outage) |
| isogenic | biological replicate | 0.0003688 [6.94e-06, 0.000731] | 16 (16) | 16 (16) | 0.00856 | ok |
| anisogenic | donor |  | 0 (0) | 0 (0) |  | no pair in EIC for this assay class |
| other_experiment (experiment) | experiment repeat |  | 1 (1) | 1 (1) |  | not estimable: 1 pairs from 1 cells with a isogenic pair in the same cell and assay (rule: 5 pairs from 3 cells); all pairs of the row: 1 from 1 cells |
| other_experiment (experiment;lab) | lab |  | 2 (2) | 2 (2) |  | not estimable: 2 pairs from 2 cells with a isogenic pair in the same cell and assay (rule: 5 pairs from 3 cells); all pairs of the row: 2 from 2 cells |
| other_experiment (experiment;lab;donor) | lab with donor |  | 4 (4) | 4 (4) |  | not estimable: 4 pairs from 4 cells with a isogenic pair in the same cell and assay (rule: 5 pairs from 3 cells); all pairs of the row: 4 from 4 cells |
| other_experiment (experiment;donor) | experiment repeat with donor |  | 0 (0) | 1 (1) |  | not estimable: 0 pairs from 0 cells with a isogenic pair in the same cell and assay (rule: 5 pairs from 3 cells); all pairs of the row: 1 from 1 cells |
| other_cell_type | cell type (with donor and experiment) | 0.004489 [0.0016, 0.00738] | 188 (16) | 300 (24) | 0.104 | ok |

## Old telescoping check of layers.tsv (known to fail)

0 of 75 groups telescope exactly (total_variance.telescoping_check on the pooled rows of layers.tsv, which the ladder does not use). Known cause of the failures: the rungs hold different cells, so each increment uses the cells it shares with the rung below and the sum does not telescope (telescoping.tsv).

## Telescoping check of the ladder (chain_check.tsv)

On the (cell, assay) units that hold pseudoreplicate, isogenic and other_cell_type pairs: 65 of 65 (class, stratum, band) groups telescope exactly.

| assay class | units | cells | sampling | replicate | cell type | sum | other_cell_type | ok |
|---|---|---|---|---|---|---|---|---|
| broad_histone | 16 | 16 | 0.03821 | 0.0003688 | 0.004489 | 0.04307 | 0.04307 | True |

## Crossed mixed model (order-free cross-check)

| assay | band | factor | variance fraction [95% interval] | levels | estimable | note |
|---|---|---|---|---|---|---|
| H3K36me3 | fine | cell | 1.01e-09 [7.45e-10, 1.28e-09] | 19 | 1 |  |
| H3K36me3 | fine | experiment |  | 26 | 1 | not identified inside cells: varies inside 0 cells < 3 (fitted, not quoted) |
| H3K36me3 | fine | replicate |  | 23 | 1 | not identified inside cells: varies inside 0 cells < 3 (fitted, not quoted) |
| H3K36me3 | fine | library | 2.62e-12 [2.36e-12, 2.89e-12] | 42 | 1 |  |
| H3K36me3 | fine | lab |  | 4 | 1 | not identified inside cells: varies inside 0 cells < 3 (fitted, not quoted) |
| H3K36me3 | fine | residual | 0.994 [0.946, 1] | 84 | 1 | REML (scipy L-BFGS-B, log variance components; mu and beta profiled); 20 cluster-bootstrap draws over 19 cells; layer uniform; 84 of 84 observations are pseudoreplicate halves |
| H3K36me3 | fine | log2_depth |  | 84 | 0 | every observation thinned to 3.84304e+06 reads before log1p; no depth slope fitted |

## Identifiability of the mixed model (design.tsv)

| assay | second experiments | obs | cells | experiment | replicate (donor) | library | lab |
|---|---|---|---|---|---|---|---|
| ATAC-seq | 0 | 8 | 4 | no (0) | no (0) | no (0) | no (0) |
| ATAC-seq | 1 | 12 | 4 | yes (2) | no (2) | no (2) | no (0) |
| DNase-seq | 0 | 50 | 17 | no (0) | no (0) | yes (8) | no (0) |
| DNase-seq | 1 | 66 | 17 | yes (8) | yes (2) | yes (12) | yes (2) |
| H2AFZ | 0 | 34 | 10 | no (0) | no (0) | yes (7) | yes (0) |
| H2AFZ | 1 | 36 | 10 | yes (1) | no (0) | yes (8) | yes (1) |
| H2AK5ac | 0 | 20 | 5 | no (0) | no (0) | yes (5) | no (0) |
| H2AK5ac | 1 | 20 | 5 | no (0) | no (0) | yes (5) | no (0) |
| H2AK9ac | 0 | 4 | 1 |  |  |  |  |
| H2AK9ac | 1 | 4 | 1 |  |  |  |  |
| H2BK120ac | 0 | 18 | 5 | no (0) | no (0) | yes (4) | no (0) |
| H2BK120ac | 1 | 18 | 5 | no (0) | no (0) | yes (4) | no (0) |
| H2BK12ac | 0 | 16 | 5 | no (0) | no (0) | yes (3) | no (0) |
| H2BK12ac | 1 | 18 | 5 | yes (1) | no (0) | yes (4) | no (0) |
| H2BK15ac | 0 | 16 | 4 | no (0) | no (0) | yes (3) | no (0) |
| H2BK15ac | 1 | 16 | 4 | no (0) | no (0) | yes (3) | no (0) |
| H2BK20ac | 0 | 8 | 2 | no (0) | no (0) | yes (2) | no (0) |
| H2BK20ac | 1 | 8 | 2 | no (0) | no (0) | yes (2) | no (0) |
| H2BK5ac | 0 | 18 | 5 | no (0) | no (0) | yes (4) | no (0) |
| H2BK5ac | 1 | 18 | 5 | no (0) | no (0) | yes (4) | no (0) |
| H3F3A | 0 | 4 | 1 |  |  |  |  |
| H3F3A | 1 | 4 | 1 |  |  |  |  |
| H3K14ac | 0 | 16 | 5 | no (0) | no (0) | yes (3) | no (0) |
| H3K14ac | 1 | 18 | 5 | yes (1) | no (0) | yes (4) | no (0) |
| H3K18ac | 0 | 22 | 6 | no (0) | no (0) | yes (5) | no (0) |
| H3K18ac | 1 | 24 | 6 | yes (1) | no (0) | yes (6) | no (0) |
| H3K23ac | 0 | 10 | 3 | no (0) | no (0) | yes (2) | no (0) |
| H3K23ac | 1 | 12 | 3 | yes (1) | no (0) | yes (3) | no (0) |
| H3K23me2 | 0 | 8 | 2 | no (0) | no (0) | yes (2) | no (0) |
| H3K23me2 | 1 | 8 | 2 | no (0) | no (0) | yes (2) | no (0) |
| H3K27ac | 0 | 74 | 20 | no (0) | no (0) | yes (15) | yes (0) |
| H3K27ac | 1 | 88 | 20 | yes (7) | yes (6) | yes (17) | yes (2) |
| H3K27me3 | 0 | 48 | 13 | no (0) | no (0) | yes (11) | yes (0) |
| H3K27me3 | 1 | 58 | 14 | yes (4) | yes (2) | yes (12) | yes (3) |
| H3K36me3 | 0 | 70 | 19 | no (0) | no (0) | yes (16) | yes (0) |
| H3K36me3 | 1 | 84 | 19 | yes (7) | yes (4) | yes (17) | yes (5) |
| H3K4ac | 0 | 22 | 6 | no (0) | no (0) | yes (5) | no (0) |
| H3K4ac | 1 | 24 | 6 | yes (1) | no (0) | yes (6) | no (0) |
| H3K4me1 | 0 | 56 | 15 | no (0) | no (0) | yes (13) | yes (0) |
| H3K4me1 | 1 | 68 | 15 | yes (6) | yes (3) | yes (14) | yes (4) |
| H3K4me2 | 0 | 42 | 11 | no (0) | no (0) | yes (10) | yes (0) |
| H3K4me2 | 1 | 44 | 11 | yes (1) | no (0) | yes (10) | yes (1) |
| H3K4me3 | 0 | 92 | 27 | no (0) | no (0) | yes (19) | yes (0) |
| H3K4me3 | 1 | 112 | 27 | yes (10) | yes (8) | yes (24) | yes (4) |
| H3K56ac | 0 | 12 | 3 | no (0) | no (0) | yes (3) | no (0) |
| H3K56ac | 1 | 12 | 3 | no (0) | no (0) | yes (3) | no (0) |
| H3K79me1 | 0 | 20 | 6 | no (0) | no (0) | yes (4) | no (0) |
| H3K79me1 | 1 | 22 | 6 | yes (1) | no (0) | yes (4) | no (0) |
| H3K79me2 | 0 | 38 | 9 | no (0) | no (0) | yes (9) | yes (0) |
| H3K79me2 | 1 | 40 | 9 | yes (1) | no (0) | yes (9) | yes (1) |
| H3K9ac | 0 | 42 | 11 | no (0) | no (0) | yes (10) | yes (0) |
| H3K9ac | 1 | 46 | 11 | yes (2) | no (0) | yes (10) | yes (2) |
| H3K9me1 | 0 | 6 | 2 | no (0) | no (0) | yes (1) | no (0) |
| H3K9me1 | 1 | 6 | 2 | no (0) | no (0) | yes (1) | no (0) |
| H3K9me2 | 0 | 4 | 1 |  |  |  |  |
| H3K9me2 | 1 | 4 | 1 |  |  |  |  |
| H3K9me3 | 0 | 52 | 15 | no (0) | no (0) | yes (10) | yes (0) |
| H3K9me3 | 1 | 60 | 15 | yes (4) | yes (3) | yes (12) | yes (3) |
| H3T11ph | 0 | 2 | 1 |  |  |  |  |
| H3T11ph | 1 | 2 | 1 |  |  |  |  |
| H4K12ac | 0 | 4 | 1 |  |  |  |  |
| H4K12ac | 1 | 4 | 1 |  |  |  |  |
| H4K20me1 | 0 | 24 | 6 | no (0) | no (0) | yes (6) | yes (0) |
| H4K20me1 | 1 | 26 | 6 | yes (1) | no (0) | yes (6) | yes (1) |
| H4K5ac | 0 | 12 | 3 | no (0) | no (0) | yes (3) | no (0) |
| H4K5ac | 1 | 12 | 3 | no (0) | no (0) | yes (3) | no (0) |
| H4K8ac | 0 | 20 | 6 | no (0) | no (0) | yes (4) | no (0) |
| H4K8ac | 1 | 24 | 6 | yes (2) | no (0) | yes (5) | no (0) |
| H4K91ac | 0 | 14 | 5 | no (0) | no (0) | yes (2) | no (0) |
| H4K91ac | 1 | 14 | 5 | no (0) | no (0) | yes (2) | no (0) |

Cell entries: estimable by mixed_model.estimability (cells in which the factor varies).

