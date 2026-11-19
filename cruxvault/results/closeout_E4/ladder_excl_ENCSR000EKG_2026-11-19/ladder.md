# Replicate ladder (close-out item E4)

Stratum all, band all. Increments are assay-matched pair values (0.5 E[(Y - Y')^2] of log1p counts at each assay's common depth); intervals are 95% cluster-bootstrap intervals over cells. No row is pooled across assay classes. Full table: ladder.tsv.

Excluded experiments (layers.py --exclude-experiments): ENCSR000EKG. Pairs dropped before the common depth: 1 ({'DNase-seq': 1}).

Form: two-sided (PI decision of 2026-11-17, row 5). The other_cell_type and other_experiment rows subtract, pair by pair, 0.5 x (the isogenic pair value of the a-side experiment + that of the b-side experiment). A pair whose b-side experiment has no isogenic pair leaves the matched set (ladder.tsv column n_pairs_left_b). The bootstrap resamples all cells jointly: a drawn cell counts as a-side and as b-side, and the isogenic values are recomputed in each draw. Sides from: /project/def-maxwl/mforooz/ONION/CO-D2/tables/eic_v1_2x (pairs.tsv sha256 03594f8d6227, measurements.tsv sha256 301114715bb4).

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
| other_experiment (experiment;donor) | experiment repeat with donor |  | 0 (0) | 2 (2) |  | not estimable: 0 pairs from 0 cells with a isogenic pair in the experiments of both sides (rule: 5 pairs from 3 cells); left: 0 pairs whose b-side experiment has no isogenic pair, 2 whose a-side has none; all pairs of the row: 2 from 2 cells |
| other_cell_type | cell type (with donor and experiment) |  | 0 (0) | 6 (3) |  | not estimable: 0 pairs from 0 cells with a isogenic pair in the experiments of both sides (rule: 5 pairs from 3 cells); left: 0 pairs whose b-side experiment has no isogenic pair, 6 whose a-side has none; all pairs of the row: 6 from 3 cells |

## accessibility_dnase

| rung | layer | increment [95% interval] | pairs (cells) matched | pairs (cells) all | share of the cell-type pair value (same cells and assays) | flag |
|---|---|---|---|---|---|---|
| pseudoreplicate | sampling | 0.04833 [0.045, 0.0517] | 33 (17) | 33 (17) | 0.839 | ok |
| sequencing | sequencing |  | 0 (0) | 0 (0) |  | no pair: the per-library and per-lane units are build priority 4 (PI approved all 722 rows on 2026-11-16; not built: Nibi /scratch over quota, Fir outage) |
| technical | library |  | 0 (0) | 0 (0) |  | no pair: the per-library and per-lane units are build priority 4 (PI approved all 722 rows on 2026-11-16; not built: Nibi /scratch over quota, Fir outage) |
| isogenic | biological replicate | 0.0005948 [-0.000232, 0.00142] | 8 (8) | 8 (8) | 0.00907 | ok |
| anisogenic | donor |  | 0 (0) | 0 (0) |  | no pair in EIC for this assay class |
| other_experiment (experiment) | experiment repeat |  | 0 (0) | 4 (4) |  | not estimable: 0 pairs from 0 cells with a isogenic pair in the experiments of both sides (rule: 5 pairs from 3 cells); left: 4 pairs whose b-side experiment has no isogenic pair, 0 whose a-side has none; all pairs of the row: 4 from 4 cells |
| other_experiment (experiment;lab) | lab |  | 0 (0) | 1 (1) |  | not estimable: 0 pairs from 0 cells with a isogenic pair in the experiments of both sides (rule: 5 pairs from 3 cells); left: 0 pairs whose b-side experiment has no isogenic pair, 1 whose a-side has none; all pairs of the row: 1 from 1 cells |
| other_experiment (experiment;lab;donor) | lab with donor |  | 0 (0) | 0 (0) |  | no pair in EIC for this assay class |
| other_experiment (experiment;donor) | experiment repeat with donor |  | 0 (0) | 2 (2) |  | not estimable: 0 pairs from 0 cells with a isogenic pair in the experiments of both sides (rule: 5 pairs from 3 cells); left: 0 pairs whose b-side experiment has no isogenic pair, 2 whose a-side has none; all pairs of the row: 2 from 2 cells |
| other_cell_type | cell type (with donor and experiment) | 0.01271 [0.00841, 0.017] | 28 (7) | 136 (16) | 0.229 | ok |

## broad_histone

| rung | layer | increment [95% interval] | pairs (cells) matched | pairs (cells) all | share of the cell-type pair value (same cells and assays) | flag |
|---|---|---|---|---|---|---|
| pseudoreplicate | sampling | 0.04627 [0.0413, 0.0512] | 262 (27) | 262 (27) | 0.951 | ok |
| sequencing | sequencing |  | 0 (0) | 0 (0) |  | no pair: the per-library and per-lane units are build priority 4 (PI approved all 722 rows on 2026-11-16; not built: Nibi /scratch over quota, Fir outage) |
| technical | library |  | 0 (0) | 0 (0) |  | no pair: the per-library and per-lane units are build priority 4 (PI approved all 722 rows on 2026-11-16; not built: Nibi /scratch over quota, Fir outage) |
| isogenic | biological replicate | 0.0005323 [0.000249, 0.000815] | 76 (17) | 76 (17) | 0.0127 | ok |
| anisogenic | donor |  | 0 (0) | 0 (0) |  | no pair in EIC for this assay class |
| other_experiment (experiment) | experiment repeat |  | 0 (0) | 4 (3) |  | not estimable: 0 pairs from 0 cells with a isogenic pair in the experiments of both sides (rule: 5 pairs from 3 cells); left: 4 pairs whose b-side experiment has no isogenic pair, 0 whose a-side has none; all pairs of the row: 4 from 3 cells |
| other_experiment (experiment;lab) | lab |  | 0 (0) | 8 (2) |  | not estimable: 0 pairs from 0 cells with a isogenic pair in the experiments of both sides (rule: 5 pairs from 3 cells); left: 8 pairs whose b-side experiment has no isogenic pair, 0 whose a-side has none; all pairs of the row: 8 from 2 cells |
| other_experiment (experiment;lab;donor) | lab with donor |  | 0 (0) | 13 (4) |  | not estimable: 0 pairs from 0 cells with a isogenic pair in the experiments of both sides (rule: 5 pairs from 3 cells); left: 10 pairs whose b-side experiment has no isogenic pair, 3 whose a-side has none; all pairs of the row: 13 from 4 cells |
| other_experiment (experiment;donor) | experiment repeat with donor |  | 0 (0) | 5 (4) |  | not estimable: 0 pairs from 0 cells with a isogenic pair in the experiments of both sides (rule: 5 pairs from 3 cells); left: 1 pairs whose b-side experiment has no isogenic pair, 4 whose a-side has none; all pairs of the row: 5 from 4 cells |
| other_cell_type | cell type (with donor and experiment) | 0.00273 [0.0015, 0.00396] | 355 (16) | 938 (26) | 0.0633 | ok |

## narrow_histone

| rung | layer | increment [95% interval] | pairs (cells) matched | pairs (cells) all | share of the cell-type pair value (same cells and assays) | flag |
|---|---|---|---|---|---|---|
| pseudoreplicate | sampling | 0.0262 [0.0228, 0.0296] | 391 (37) | 391 (37) | 0.875 | ok |
| sequencing | sequencing |  | 0 (0) | 0 (0) |  | no pair: the per-library and per-lane units are build priority 4 (PI approved all 722 rows on 2026-11-16; not built: Nibi /scratch over quota, Fir outage) |
| technical | library |  | 0 (0) | 0 (0) |  | no pair: the per-library and per-lane units are build priority 4 (PI approved all 722 rows on 2026-11-16; not built: Nibi /scratch over quota, Fir outage) |
| isogenic | biological replicate | 0.0005815 [0.000294, 0.000869] | 119 (24) | 119 (24) | 0.018 | ok |
| anisogenic | donor |  | 0 (0) | 0 (0) |  | no pair in EIC for this assay class |
| other_experiment (experiment) | experiment repeat |  | 0 (0) | 8 (3) |  | not estimable: 0 pairs from 0 cells with a isogenic pair in the experiments of both sides (rule: 5 pairs from 3 cells); left: 2 pairs whose b-side experiment has no isogenic pair, 6 whose a-side has none; all pairs of the row: 8 from 3 cells |
| other_experiment (experiment;lab) | lab |  | 0 (0) | 6 (2) |  | not estimable: 0 pairs from 0 cells with a isogenic pair in the experiments of both sides (rule: 5 pairs from 3 cells); left: 5 pairs whose b-side experiment has no isogenic pair, 1 whose a-side has none; all pairs of the row: 6 from 2 cells |
| other_experiment (experiment;lab;donor) | lab with donor |  | 0 (0) | 6 (4) |  | not estimable: 0 pairs from 0 cells with a isogenic pair in the experiments of both sides (rule: 5 pairs from 3 cells); left: 4 pairs whose b-side experiment has no isogenic pair, 2 whose a-side has none; all pairs of the row: 6 from 4 cells |
| other_experiment (experiment;donor) | experiment repeat with donor |  | 0 (0) | 11 (8) |  | not estimable: 0 pairs from 0 cells with a isogenic pair in the experiments of both sides (rule: 5 pairs from 3 cells); left: 5 pairs whose b-side experiment has no isogenic pair, 6 whose a-side has none; all pairs of the row: 11 from 8 cells |
| other_cell_type | cell type (with donor and experiment) | 0.003547 [0.00198, 0.00511] | 445 (23) | 1128 (36) | 0.12 | ok |

## Common-cell check (common_cells.tsv; replaces the telescoping check)

Pseudoreplicate, isogenic and other_cell_type recomputed on the a-side cells that all three rows hold. Pass: at least 3 common cells and each common-cell increment inside its row's interval. 165 of 195 (class, stratum, band) groups pass.

| assay class | rung | common cells | increment on common cells [95% interval] | row increment [95% interval] (cells) | inside | ok |
|---|---|---|---|---|---|---|
| accessibility_dnase | pseudoreplicate | 7 | 0.0415 [0.0394, 0.0436] | 0.04833 [0.045, 0.0517] (17) | 0 | 0 |
| accessibility_dnase | isogenic | 7 | 0.000514 [-0.000462, 0.00149] | 0.0005948 [-0.000232, 0.00142] (8) | 1 | 0 |
| accessibility_dnase | other_cell_type | 7 | 0.01271 [0.0109, 0.0145] | 0.01271 [0.00841, 0.017] (7) | 1 | 0 |
| broad_histone | pseudoreplicate | 16 | 0.04543 [0.038, 0.0529] | 0.04627 [0.0413, 0.0512] (27) | 1 | 1 |
| broad_histone | isogenic | 16 | 0.0004872 [0.00018, 0.000795] | 0.0005323 [0.000249, 0.000815] (17) | 1 | 1 |
| broad_histone | other_cell_type | 16 | 0.00273 [0.00214, 0.00332] | 0.00273 [0.0015, 0.00396] (16) | 1 | 1 |
| narrow_histone | pseudoreplicate | 23 | 0.0262 [0.0214, 0.031] | 0.0262 [0.0228, 0.0296] (37) | 1 | 1 |
| narrow_histone | isogenic | 23 | 0.0006082 [0.000308, 0.000908] | 0.0005815 [0.000294, 0.000869] (24) | 1 | 1 |
| narrow_histone | other_cell_type | 23 | 0.003547 [0.00253, 0.00457] | 0.003547 [0.00198, 0.00511] (23) | 1 | 1 |

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

