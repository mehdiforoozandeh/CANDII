# Replicate ladder (close-out item E4)

Stratum all, band all. Increments are assay-matched pair values (0.5 E[(Y - Y')^2] of log1p counts at each assay's common depth); intervals are 95% cluster-bootstrap intervals over cells. No row is pooled across assay classes. Full table: ladder.tsv.

Form: one-sided (every rung subtracts the lower rung of the a-side unit only). Kept for comparison; for the other_cell_type and other_experiment rows it is not the ladder definition since 2026-11-17.

## accessibility_atac

| rung | layer | increment [95% interval] | pairs (cells) matched | pairs (cells) all | share of the cell-type pair value (same cells and assays) | flag |
|---|---|---|---|---|---|---|
| pseudoreplicate | sampling |  | 4 (4) | 4 (4) |  | not estimable: 4 pairs from 4 cells (rule: 5 pairs from 3 cells); all pairs of the row: 4 from 4 cells |
| sequencing | sequencing |  | 0 (0) | 0 (0) |  | no pair: the per-library and per-lane units are build priority 4 (PI approved all 722 rows on 2026-11-16; not built: Nibi /scratch over quota, Fir outage) |
| technical | library |  | 0 (0) | 0 (0) |  | no pair: the per-library and per-lane units are build priority 4 (PI approved all 722 rows on 2026-11-16; not built: Nibi /scratch over quota, Fir outage) |
| isogenic | biological replicate |  | 0 (0) | 0 (0) |  | no pair in EIC for this assay class |
| anisogenic | donor |  | 0 (0) | 0 (0) |  | no pair in EIC for this assay class |
| other_experiment (experiment) | experiment repeat |  | 0 (0) | 0 (0) |  | no pair in EIC for this assay class |
| other_experiment (experiment;lab) | lab |  | 0 (0) | 0 (0) |  | no pair in EIC for this assay class |
| other_experiment (experiment;lab;donor) | lab with donor |  | 0 (0) | 0 (0) |  | no pair in EIC for this assay class |
| other_experiment (experiment;donor) | experiment repeat with donor |  | 0 (0) | 2 (2) |  | not estimable: 0 pairs from 0 cells with a isogenic pair in the same cell and assay (rule: 5 pairs from 3 cells); all pairs of the row: 2 from 2 cells |
| other_cell_type | cell type (with donor and experiment) |  | 0 (0) | 6 (3) |  | not estimable: 0 pairs from 0 cells with a isogenic pair in the same cell and assay (rule: 5 pairs from 3 cells); all pairs of the row: 6 from 3 cells |

## accessibility_dnase

| rung | layer | increment [95% interval] | pairs (cells) matched | pairs (cells) all | share of the cell-type pair value (same cells and assays) | flag |
|---|---|---|---|---|---|---|
| pseudoreplicate | sampling | 0.0001564 [0.000153, 0.000159] | 33 (17) | 33 (17) | 1 | ok |
| sequencing | sequencing |  | 0 (0) | 0 (0) |  | no pair: the per-library and per-lane units are build priority 4 (PI approved all 722 rows on 2026-11-16; not built: Nibi /scratch over quota, Fir outage) |
| technical | library |  | 0 (0) | 0 (0) |  | no pair: the per-library and per-lane units are build priority 4 (PI approved all 722 rows on 2026-11-16; not built: Nibi /scratch over quota, Fir outage) |
| isogenic | biological replicate |  | 8 (8) | 8 (8) | -0.0121 | ok; negative increment (not clipped) |
| anisogenic | donor |  | 0 (0) | 0 (0) |  | no pair in EIC for this assay class |
| other_experiment (experiment) | experiment repeat |  | 4 (4) | 4 (4) |  | not estimable: 4 pairs from 4 cells with a isogenic pair in the same cell and assay (rule: 5 pairs from 3 cells); all pairs of the row: 4 from 4 cells |
| other_experiment (experiment;lab) | lab |  | 0 (0) | 2 (2) |  | not estimable: 0 pairs from 0 cells with a isogenic pair in the same cell and assay (rule: 5 pairs from 3 cells); all pairs of the row: 2 from 2 cells |
| other_experiment (experiment;lab;donor) | lab with donor |  | 0 (0) | 0 (0) |  | no pair in EIC for this assay class |
| other_experiment (experiment;donor) | experiment repeat with donor |  | 0 (0) | 2 (2) |  | not estimable: 0 pairs from 0 cells with a isogenic pair in the same cell and assay (rule: 5 pairs from 3 cells); all pairs of the row: 2 from 2 cells |
| other_cell_type | cell type (with donor and experiment) | 2.425e-06 [-1.25e-06, 6.1e-06] | 79 (7) | 136 (16) | 0.0156 | ok |

## broad_histone

| rung | layer | increment [95% interval] | pairs (cells) matched | pairs (cells) all | share of the cell-type pair value (same cells and assays) | flag |
|---|---|---|---|---|---|---|
| pseudoreplicate | sampling | 0.04627 [0.0413, 0.0512] | 262 (27) | 262 (27) | 0.951 | ok |
| sequencing | sequencing |  | 0 (0) | 0 (0) |  | no pair: the per-library and per-lane units are build priority 4 (PI approved all 722 rows on 2026-11-16; not built: Nibi /scratch over quota, Fir outage) |
| technical | library |  | 0 (0) | 0 (0) |  | no pair: the per-library and per-lane units are build priority 4 (PI approved all 722 rows on 2026-11-16; not built: Nibi /scratch over quota, Fir outage) |
| isogenic | biological replicate | 0.0005323 [0.000249, 0.000815] | 76 (17) | 76 (17) | 0.0127 | ok |
| anisogenic | donor |  | 0 (0) | 0 (0) |  | no pair in EIC for this assay class |
| other_experiment (experiment) | experiment repeat |  | 4 (3) | 4 (3) |  | not estimable: 4 pairs from 3 cells with a isogenic pair in the same cell and assay (rule: 5 pairs from 3 cells); all pairs of the row: 4 from 3 cells |
| other_experiment (experiment;lab) | lab |  | 8 (2) | 8 (2) |  | not estimable: 8 pairs from 2 cells with a isogenic pair in the same cell and assay (rule: 5 pairs from 3 cells); all pairs of the row: 8 from 2 cells |
| other_experiment (experiment;lab;donor) | lab with donor |  | 10 (4) | 13 (4) | -0.0694 | ok; negative increment (not clipped) |
| other_experiment (experiment;donor) | experiment repeat with donor |  | 1 (1) | 5 (4) |  | not estimable: 1 pairs from 1 cells with a isogenic pair in the same cell and assay (rule: 5 pairs from 3 cells); all pairs of the row: 5 from 4 cells |
| other_cell_type | cell type (with donor and experiment) | 0.003597 [0.000676, 0.00652] | 522 (17) | 938 (26) | 0.0792 | ok |

## narrow_histone

| rung | layer | increment [95% interval] | pairs (cells) matched | pairs (cells) all | share of the cell-type pair value (same cells and assays) | flag |
|---|---|---|---|---|---|---|
| pseudoreplicate | sampling | 0.0262 [0.0228, 0.0296] | 391 (37) | 391 (37) | 0.875 | ok |
| sequencing | sequencing |  | 0 (0) | 0 (0) |  | no pair: the per-library and per-lane units are build priority 4 (PI approved all 722 rows on 2026-11-16; not built: Nibi /scratch over quota, Fir outage) |
| technical | library |  | 0 (0) | 0 (0) |  | no pair: the per-library and per-lane units are build priority 4 (PI approved all 722 rows on 2026-11-16; not built: Nibi /scratch over quota, Fir outage) |
| isogenic | biological replicate | 0.0005815 [0.000294, 0.000869] | 119 (24) | 119 (24) | 0.018 | ok |
| anisogenic | donor |  | 0 (0) | 0 (0) |  | no pair in EIC for this assay class |
| other_experiment (experiment) | experiment repeat |  | 2 (2) | 8 (3) |  | not estimable: 2 pairs from 2 cells with a isogenic pair in the same cell and assay (rule: 5 pairs from 3 cells); all pairs of the row: 8 from 3 cells |
| other_experiment (experiment;lab) | lab |  | 5 (2) | 6 (2) |  | not estimable: 5 pairs from 2 cells with a isogenic pair in the same cell and assay (rule: 5 pairs from 3 cells); all pairs of the row: 6 from 2 cells |
| other_experiment (experiment;lab;donor) | lab with donor |  | 4 (3) | 6 (4) |  | not estimable: 4 pairs from 3 cells with a isogenic pair in the same cell and assay (rule: 5 pairs from 3 cells); all pairs of the row: 6 from 4 cells |
| other_experiment (experiment;donor) | experiment repeat with donor | 0.000262 [-0.0016, 0.00212] | 5 (3) | 11 (8) | -0.0115 | ok |
| other_cell_type | cell type (with donor and experiment) | 0.004129 [0.00118, 0.00708] | 670 (23) | 1128 (36) | 0.136 | ok |

## Old telescoping check of layers.tsv (known to fail)

327 of 2925 groups telescope exactly (total_variance.telescoping_check on the pooled rows of layers.tsv, which the ladder does not use). Known cause of the failures: the rungs hold different cells, so each increment uses the cells it shares with the rung below and the sum does not telescope (telescoping.tsv).

## Telescoping check of the ladder (chain_check.tsv)

On the (cell, assay) units that hold pseudoreplicate, isogenic and other_cell_type pairs: 195 of 195 (class, stratum, band) groups telescope exactly.

| assay class | units | cells | sampling | replicate | cell type | sum | other_cell_type | ok |
|---|---|---|---|---|---|---|---|---|
| accessibility_atac | 0 | 0 |  |  |  |  |  |  |
| accessibility_dnase | 7 | 7 | 0.0001545 | -1.871e-06 | 2.425e-06 | 0.000155 | 0.000155 | True |
| broad_histone | 65 | 17 | 0.04123 | 0.0005784 | 0.003597 | 0.0454 | 0.0454 | True |
| narrow_histone | 91 | 23 | 0.02558 | 0.0005459 | 0.004129 | 0.03025 | 0.03025 | True |

## Common-cell check (common_cells.tsv; replaces the telescoping check)

Pseudoreplicate, isogenic and other_cell_type recomputed on the a-side cells that all three rows hold. Pass: at least 3 common cells and each common-cell increment inside its row's interval. 167 of 195 (class, stratum, band) groups pass.

| assay class | rung | common cells | increment on common cells [95% interval] | row increment [95% interval] (cells) | inside | ok |
|---|---|---|---|---|---|---|
| accessibility_dnase | pseudoreplicate | 7 | 0.0001545 [0.000148, 0.000161] | 0.0001564 [0.000153, 0.000159] (17) | 1 | 1 |
| accessibility_dnase | isogenic | 7 | -1.871e-06 [-4.95e-06, 1.21e-06] | -1.716e-06 [-4.25e-06, 8.18e-07] (8) | 1 | 1 |
| accessibility_dnase | other_cell_type | 7 | 2.425e-06 [-1.25e-06, 6.1e-06] | 2.425e-06 [-1.25e-06, 6.1e-06] (7) | 1 | 1 |
| broad_histone | pseudoreplicate | 17 | 0.04481 [0.0377, 0.0519] | 0.04627 [0.0413, 0.0512] (27) | 1 | 1 |
| broad_histone | isogenic | 17 | 0.0005323 [0.000249, 0.000815] | 0.0005323 [0.000249, 0.000815] (17) | 1 | 1 |
| broad_histone | other_cell_type | 17 | 0.003597 [0.000676, 0.00652] | 0.003597 [0.000676, 0.00652] (17) | 1 | 1 |
| narrow_histone | pseudoreplicate | 23 | 0.0262 [0.0214, 0.031] | 0.0262 [0.0228, 0.0296] (37) | 1 | 1 |
| narrow_histone | isogenic | 23 | 0.0006082 [0.000308, 0.000908] | 0.0005815 [0.000294, 0.000869] (24) | 1 | 1 |
| narrow_histone | other_cell_type | 23 | 0.004129 [0.00118, 0.00708] | 0.004129 [0.00118, 0.00708] (23) | 1 | 1 |

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

