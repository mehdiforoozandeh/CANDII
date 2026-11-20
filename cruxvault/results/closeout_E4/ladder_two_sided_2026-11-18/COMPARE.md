# Replicate ladder: two-sided against one-sided (close-out item E4, follow-up, 2026-11-18)

## What ran

- Input: the per-pair values of tonight's layers stage, `../ladder_2026-11-18/pair_values.tsv.gz` (Nibi job 27692829 made the one-sided report from them; sha256 16c10099...). The experiment of each side of a pair comes from `tables_eic_v1_2x/` (byte copies of Nibi `/project/def-maxwl/mforooz/ONION/CO-D2/tables/eic_v1_2x/pairs.tsv` and `measurements.tsv`; sha256 03594f8d..., 30111471...).
- Code: `tools/onion_eval/ladder.py report` on branch `chunk/onion-CO-E4b`, on the laptop, 1,000 bootstrap draws, seed 0, about 5 s per form.
  - `two_sided/`: `--form two_sided` (the default) `--tables tables_eic_v1_2x --design ../ladder_2026-11-18/design.tsv`.
  - `one_sided/`: `--form one_sided --layers ../ladder_2026-11-18/layers.tsv --design ../ladder_2026-11-18/design.tsv`.
- Reproduction check: `one_sided/ladder.tsv` is byte-identical to tonight's `../ladder_2026-11-18/ladder.tsv` (sha256 a39e9b99... for both). `one_sided/chain_check.tsv` and `one_sided/telescoping.tsv` are byte-identical to tonight's files too.
- `make_compare.py` (this folder) prints the tables below.

## The two-sided form (the specification is in the item file, section 10)

- The rungs other_cell_type and other_experiment have two sides from two different experiments. Each such pair subtracts 0.5 x (L(a) + L(b)). L(x) is the mean isogenic pair value of the experiment of side x, in the same assay, stratum and band.
- A pair leaves the matched set when the experiment of a side has no isogenic pair. The column `n_pairs_left_b` of `two_sided/ladder.tsv` counts the pairs that left because of the b-side.
- The nested rungs (pseudoreplicate, isogenic) are the same in both forms (Table 1: relative change 0.0%).
- Interval (revised after review, 2026-11-18): 95% cluster bootstrap that resamples all cells jointly. A drawn cell counts wherever it appears, as a-side or as b-side, and the isogenic values L are recomputed inside each draw (`ladder._joint_boot`). The first version resampled a-side cells only and held each L(b) fixed; its intervals were too narrow (broad cell type 0.00214 to 0.00332, width 0.00118). On synthetic data with shared b-side cells, the joint interval covered the true increment in 281 of 300 data sets (93.7%; test `test_joint_bootstrap_covers_truth_at_about_nominal_rate` uses 120 data sets).

## Results

1. Cell-type rung, all regions and bands (Table 1). Broad histone marks: one-sided 0.003597 [0.000676, 0.00652] (522 pairs, 17 cells) -> two-sided 0.00273 [0.0015, 0.00396] (355 pairs, 16 cells), -24.1%. Narrow histone marks: 0.004129 [0.00118, 0.00708] (670 pairs, 23 cells) -> 0.003547 [0.00198, 0.00511] (445 pairs, 23 cells), -14.1%. DNase-seq: 2.425e-06 [-1.25e-06, 6.1e-06] (79 pairs, 7 cells) -> 1.971e-06 [-3.32e-06, 7.27e-06] (28 pairs, 7 cells), -18.7% (the DNase-seq values are not interpretable; see the anomaly below).
2. Most of the drop comes from the pairs that leave, not from the subtraction (Table 2). 167 broad, 225 narrow and 51 DNase-seq cell-type pairs leave, because their b-side experiment has no isogenic pair. On the pairs that the two-sided form keeps, the one-sided subtraction gives 0.001172 for broad marks against two-sided 0.00273: there the b-sides are on average quieter than the a-sides (mean 0.5 (L(b) - L(a)) = -0.00156). For narrow marks the two agree on the kept pairs (0.003521 against 0.003547). Therefore the one-sided form is not an upper value in general. The earlier (one-sided) cell-type numbers are higher than the two-sided numbers at all regions and bands in all three classes, and in 56 of 65 broad and 49 of 65 narrow (stratum, band) groups (Table 3).
3. Every other_experiment rung is not estimable in the two-sided form. No second experiment has an isogenic pair of its own in the layers pair list (3,979 pairs); the b-side of every other_experiment pair is such a second experiment. The two rows that were estimable one-sided are lab with donor in broad marks, -0.003265 [-0.0118, 0.00526] (10 pairs, 4 cells), and experiment repeat with another donor in narrow marks, 0.000262 [-0.0016, 0.00212] (5 pairs, 3 cells). Both intervals included zero. In the two-sided form all 10 and all 5 pairs leave.
4. Common-cell check (replaces the telescoping check): 167 of 195 (class, stratum, band) groups pass in both forms. All 28 failures are DNase-seq strata, where the read-sampling value on the 7 common cells lies outside the interval of the 17-cell row (Table 4). All histone groups pass. At all regions and bands, each class passes (`two_sided/ladder.md`).

## Anomaly: the DNase-seq increment "3.40" in the second-experiment layers

- The rows: `../../closeout_D2/layers_2026-11-18/layers.tsv`, stratum all, band all, rung other_experiment. The DNase-seq assay row (8 pairs, 8 cells, variance 0.0001727) and its pair class "experiment" (4 pairs, 4 cells, variance 0.0001878; also pooled as `accessibility_dnase`) hold layer_increment 3.3982866565707295e-05, interval [1.787e-05, 5.009e-05].
- Verdict, part 1 (reading error): the value is 3.40e-05, not 3.40. The exponent was lost when it was read. The 4-pair and 8-pair rows share it because only 4 of the 8 cells (C05, C06, C07, C51) hold DNase-seq isogenic pairs, and all 4 hold pairs of class "experiment". The 4-pair row is flagged "few pairs" (rule: 5 pairs).
- Verdict, part 2 (a depth defect, real and not fixed): every DNase-seq pair value in the second-experiment layers and in the replicate ladder (both forms) is computed at a common depth of 15,201 reads (column `depth` of `layers.tsv` and of `pair_values.tsv.gz`). `layers.py` sets the common depth of an assay to the shallowest side of any of its pairs. The shallowest side is the second experiment ENCSR000EKG, pooled track, cell C23, lab Duke (measurements.tsv: 15,201 reads), in pair pd4626cb394db (class experiment with lab, split val). The next shallowest DNase-seq side has 13,300,584 reads, 875 times deeper (13,300,584 / 15,201 = 875; the side of a pair is what sets the common depth). The shallowest DNase-seq measurement in the tables that is in no pair of the list is a depth-reduced copy (arm depth0.125) of ENCSR328UMC, cell C21, with 3,434,289 reads (226 times 15,201); it does not enter the common depth. At 15,201 reads a genome of about 3.1 Gb in 25 bp bins (about 124 million bins) gets about 0.0001 reads per bin on average (arithmetic, not measured), so the DNase-seq pair values (read sampling 0.0001564 against 0.0262 to 0.0463 for the histone classes) and their increments carry no usable information. The ratio 3.40e-05 over 0.00019 must not be read as a layer.
- What depends on it. The value 3.40e-05 enters no number of the deck or the report. The deck and the report read the pair count of these rows only ("the accessibility classes have at most 4 pairs"); the count does not depend on depth. The DNase-seq rows of the replicate ladder (deck slide 12, panel of DNase-seq; report table of the ladder and the DNase-seq shares) are at the same 15,201-read depth. Version 3 of the deck and the report marks them as not interpretable for this reason.
- Fix (not built; a layers rerun on the cluster): leave the 15,201-read track out of the DNase-seq pairs (or set a minimum depth for the common depth), and rerun the layers stage for DNase-seq. The foreman or the PI decides.

## Table 1. Stratum all, band all: every rung with a pair, both forms

| assay class | rung | one-sided increment [95% interval] | pairs (cells) | two-sided increment [95% interval] | pairs (cells) | pairs left: b-side has no isogenic pair | relative change |
|---|---|---|---|---|---|---|---|
| accessibility_atac | other_cell_type | not estimable | 0 (0) | not estimable | 0 (0) | 0 |  |
| accessibility_atac | other_experiment (experiment;donor) | not estimable | 0 (0) | not estimable | 0 (0) | 0 |  |
| accessibility_atac | pseudoreplicate | not estimable | 4 (4) | not estimable | 4 (4) |  |  |
| accessibility_dnase | isogenic | -1.716e-06 [-4.25e-06, 8.18e-07] | 8 (8) | -1.716e-06 [-4.25e-06, 8.18e-07] | 8 (8) |  | +0.0% |
| accessibility_dnase | other_cell_type | 2.425e-06 [-1.25e-06, 6.1e-06] | 79 (7) | 1.971e-06 [-3.32e-06, 7.27e-06] | 28 (7) | 51 | -18.7% |
| accessibility_dnase | other_experiment (experiment) | not estimable | 4 (4) | not estimable | 0 (0) | 4 |  |
| accessibility_dnase | other_experiment (experiment;donor) | not estimable | 0 (0) | not estimable | 0 (0) | 0 |  |
| accessibility_dnase | other_experiment (experiment;lab) | not estimable | 0 (0) | not estimable | 0 (0) | 0 |  |
| accessibility_dnase | pseudoreplicate | 0.0001564 [0.000153, 0.000159] | 33 (17) | 0.0001564 [0.000153, 0.000159] | 33 (17) |  | +0.0% |
| broad_histone | isogenic | 0.0005323 [0.000249, 0.000815] | 76 (17) | 0.0005323 [0.000249, 0.000815] | 76 (17) |  | +0.0% |
| broad_histone | other_cell_type | 0.003597 [0.000676, 0.00652] | 522 (17) | 0.00273 [0.0015, 0.00396] | 355 (16) | 167 | -24.1% |
| broad_histone | other_experiment (experiment) | not estimable | 4 (3) | not estimable | 0 (0) | 4 |  |
| broad_histone | other_experiment (experiment;donor) | not estimable | 1 (1) | not estimable | 0 (0) | 1 |  |
| broad_histone | other_experiment (experiment;lab) | not estimable | 8 (2) | not estimable | 0 (0) | 8 |  |
| broad_histone | other_experiment (experiment;lab;donor) | -0.003265 [-0.0118, 0.00526] | 10 (4) | not estimable | 0 (0) | 10 |  |
| broad_histone | pseudoreplicate | 0.04627 [0.0413, 0.0512] | 262 (27) | 0.04627 [0.0413, 0.0512] | 262 (27) |  | +0.0% |
| narrow_histone | isogenic | 0.0005815 [0.000294, 0.000869] | 119 (24) | 0.0005815 [0.000294, 0.000869] | 119 (24) |  | +0.0% |
| narrow_histone | other_cell_type | 0.004129 [0.00118, 0.00708] | 670 (23) | 0.003547 [0.00198, 0.00511] | 445 (23) | 225 | -14.1% |
| narrow_histone | other_experiment (experiment) | not estimable | 2 (2) | not estimable | 0 (0) | 2 |  |
| narrow_histone | other_experiment (experiment;donor) | 0.000262 [-0.0016, 0.00212] | 5 (3) | not estimable | 0 (0) | 5 |  |
| narrow_histone | other_experiment (experiment;lab) | not estimable | 5 (2) | not estimable | 0 (0) | 5 |  |
| narrow_histone | other_experiment (experiment;lab;donor) | not estimable | 4 (3) | not estimable | 0 (0) | 4 |  |
| narrow_histone | pseudoreplicate | 0.0262 [0.0228, 0.0296] | 391 (37) | 0.0262 [0.0228, 0.0296] | 391 (37) |  | +0.0% |

## Table 2. Cell-type rung, stratum all, band all: the one-sided subtraction on the pairs the two-sided form keeps

| assay class | pairs (cells) | one-sided, all its matched pairs | one-sided, two-sided pairs | two-sided | mean 0.5 (L(b) - L(a)) over the kept pairs |
|---|---|---|---|---|---|
| accessibility_dnase | 28 (7) | 2.425e-06 | 2.238e-06 | 1.971e-06 | 2.67e-07 |
| broad_histone | 355 (16) | 0.003597 | 0.001172 | 0.00273 | -0.00156 |
| narrow_histone | 445 (23) | 0.004129 | 0.003521 | 0.003547 | -2.52e-05 |

## Table 3. Cell-type rung over all strata and signal bands: relative change two-sided vs one-sided

| assay class | groups with both values | median relative change | min | max | groups where two-sided < one-sided |
|---|---|---|---|---|---|
| accessibility_dnase | 65 | +9.8% | -480.3% | +1843.6% | 29 |
| broad_histone | 65 | -21.5% | -119.7% | +14.8% | 56 |
| narrow_histone | 65 | -8.8% | -115.8% | +6.7% | 49 |

## Table 4. Common-cell check, two-sided: 167 of 195 groups pass; the failing groups

| assay class | stratum | band | common cells | rungs outside their row interval |
|---|---|---|---|---|
| accessibility_dnase | enhancers | all | 7 | pseudoreplicate |
| accessibility_dnase | enhancers | broad | 7 | pseudoreplicate |
| accessibility_dnase | enhancers | domain | 7 | pseudoreplicate |
| accessibility_dnase | enhancers | fine | 7 | pseudoreplicate |
| accessibility_dnase | enhancers | peak | 7 | pseudoreplicate |
| accessibility_dnase | heterochromatin | all | 7 | pseudoreplicate |
| accessibility_dnase | heterochromatin | broad | 7 | pseudoreplicate |
| accessibility_dnase | heterochromatin | domain | 7 | pseudoreplicate |
| accessibility_dnase | heterochromatin | fine | 7 | pseudoreplicate |
| accessibility_dnase | heterochromatin | peak | 7 | pseudoreplicate |
| accessibility_dnase | mappability_d0 | all | 7 | pseudoreplicate |
| accessibility_dnase | mappability_d0 | broad | 7 | pseudoreplicate |
| accessibility_dnase | mappability_d0 | domain | 7 | pseudoreplicate |
| accessibility_dnase | mappability_d0 | fine | 7 | pseudoreplicate |
| accessibility_dnase | mappability_d0 | peak | 7 | pseudoreplicate |
| accessibility_dnase | mappability_d2 | all | 7 | pseudoreplicate |
| accessibility_dnase | mappability_d2 | fine | 7 | pseudoreplicate |
| accessibility_dnase | mappability_d2 | peak | 7 | pseudoreplicate |
| accessibility_dnase | mappability_d4 | all | 7 | pseudoreplicate |
| accessibility_dnase | mappability_d4 | domain | 7 | pseudoreplicate |
| accessibility_dnase | mappability_d4 | fine | 7 | pseudoreplicate |
| accessibility_dnase | mappability_d4 | peak | 7 | pseudoreplicate |
| accessibility_dnase | mappability_d5 | peak | 7 | pseudoreplicate |
| accessibility_dnase | promoters | all | 7 | pseudoreplicate |
| accessibility_dnase | promoters | broad | 7 | pseudoreplicate |
| accessibility_dnase | promoters | domain | 7 | pseudoreplicate |
| accessibility_dnase | promoters | fine | 7 | pseudoreplicate |
| accessibility_dnase | promoters | peak | 7 | pseudoreplicate |
