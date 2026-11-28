# t396 iteration 1: read start vs read count, and the dedup and SE/PE caps

Hand-written by chunk C16 on 2026-11-28. Not an output of `runall.sh`. Every number below was re-read
from the subfolder file that the row names. No model was trained in this iteration.

==Iteration 1 gave three clear results and one void test. (1) The covariance term repaired the
calibration of the start-to-count conversion. (2) `log2_depth` counts both mates of a PE fragment, and
the stores count the same mates. (3) Deduplication is per replicate and keeps exactly the distinct
5' positions. The start-space comparison of leg B is void, because the Richardson-Lucy deconvolution
failed gate (c). No closed-form cap passed gate (d), on the dedup pairs or on the SE/PE pairs.==

## Gates

| gate | bar | outcome | source |
|---|---|---|---|
| (a) dump reproduces stored `mu` | `max_abs_rel_diff` <= 1e-4 | 129 of 129 pass; max 2.29e-07 | `dump/gate.tsv`, `dump/DUMP.md` |
| (b) `score_spaces.py` reproduces `scores.json` | within 0.002 | 6 of 6 pass; max abs diff 1.1e-16 | `legB_spaces/check.tsv` |
| (c) deconvolution round trip and conversion floor | round trip <= 0.05 per scorable assay; floor median <= 0.10 | **failed**: H3K36me3 0.055, H3K9me3 0.144; floor median 0.133 | `legB_spaces/LEG_B.md`, `truth_validation.tsv` |
| (d) a cap law explains a pair set | gap closed vs depth scaling >= 0.8, top 1% | **failed** for every law on dedup `val` pairs and on PE parents of SE views | `dedup_cap/table.tsv`, `sepe_cap/table.tsv` |
| golden (bit-exact vs `golden_base_2c51e792.pt`) | 0 ULP | OK, 0 ULP, params 2,353,634 | `python tools/golden.py check`, run by C16 |

## 1. Leg B: both models in three score spaces (h57)

Deviance skill against the average reference, macro over the 9 scorable assays, 256 regions, seeds
0-2. Higher is better. Compare within one space only. Source: `legB_spaces/table.tsv`, `LEG_B.md`.

| space | model | skill | floor | seed range | per seed (0, 1, 2) | ratio_to_halves | gate outcome |
|---|---|---|---|---|---|---|---|
| overlap_count | reference | 0.543 | 0.516 | 0.065 | 0.543, 0.510, 0.575 | 0.304 | (b) pass |
| overlap_count | read-start | 0.375 | 0.525 | 0.087 | 0.425, 0.338, 0.362 | 0.322 | (b) pass |
| start_sum | reference, n_iter 50 | 0.588 | 0.711 | 0.067 | 0.592, 0.552, 0.620 | 0.357 | (c) failed: `not_interpreted` |
| start_sum | reference, n_iter 10 (best over grid) | 0.650 | 0.706 | 0.070 | 0.650, 0.615, 0.685 | 0.352 | (c) failed: `not_interpreted` |
| start_sum | read-start | 0.578 | 0.713 | 0.072 | 0.615, 0.543, 0.577 | 0.358 | -- |
| sum_1kb_count | reference | 0.394 | 0.038 | 0.065 | 0.405, 0.357, 0.421 | 0.690 | -- |
| sum_1kb_count | read-start | 0.184 | 0.038 | 0.118 | 0.252, 0.134, 0.164 | 0.877 | -- |

- ==In `overlap_count` the reference led by 0.168. In `sum_1kb_count` it led by 0.211. Both leads
  exceed both seed ranges, and every reference seed was above every read-start seed.==
- ==H3K36me3 carried 0.093 of the 0.168 `overlap_count` gap. Without it, the 8-assay gap was 0.084,
  about equal to the macro seed ranges (`LEG_B.md`, `per_assay.tsv`).==
- ==The 16 pair regions gave the same order in every space (`LEG_B.md`).==
- ==Richardson-Lucy of the half-1 counts against the half-2 starts had a median normalized MSE of
  1.59 at 50 iterations, against 1.05 without deconvolution. Deconvolution amplified the count
  noise (`LEG_B.md`).==
- ==Verdict: h57 has no answer yet. The read-count model leads in the two count spaces; the start
  space is void.==

## 2. Leg A: covariance term in the start-to-count conversion

Calibration ratio (`deviance.calibration_ratio`; 1 = calibrated, above 1 = NB too narrow), 128 even
regions, mean over seeds 0-2. `rho` and `c` were fitted on the 128 odd regions. Source:
`legA_covariance/table.tsv`, `LEG_A.md`.

| variant | all_assays | seed range | scorable | seed range | floor | gate outcome |
|---|---|---|---|---|---|---|
| read-start, independent conversion as stored | 1.258 | 0.050 | 1.303 | 0.098 | 1 (calibrated) | -- |
| read-start, covariance term, `rho` leave-one-target-out | 1.032 | 0.025 | 1.044 | 0.060 | 1 | -- |
| read-start, covariance term, `rho` in-sample | 1.055 | 0.015 | 1.052 | 0.041 | 1 | -- |
| read-start, independent, `n / c` leave-one-target-out | 1.050 | 0.031 | 0.975 | 0.024 | 1 | -- |
| reference as stored | 1.064 | 0.032 | 1.064 | 0.038 | 1 | -- |
| reference, `n / c` leave-one-target-out | 1.062 | 0.048 | 0.996 | 0.023 | 1 | -- |

- ==The covariance term closed 1.16 (all_assays) and 1.09 (scorable) of the gap between the
  independent conversion and the stored reference. The gap is 4 to 8 times the seed range of the
  covariance variant.==
- ==Under the same leave-one-target-out recalibration, the converted read-start model and the
  reference were not separable at 3 seeds: the difference was -0.029 (all_assays) and +0.047
  (scorable), inside the seed ranges.==
- ==No gate was set for leg A before the run. The reproduction check of the independent variant
  matched the stored `n` on 129 of 129 target-seed pairs (max relative difference 8.2e-7).==

## 3. Dedup cap: start space vs count space (h58)

Data only, no seeds. Gap closed against depth scaling on the 43 `val` pairs, pooled over layers, run
types and assay classes. Interval = 95% bootstrap over experiments (1000 draws). Source:
`dedup_cap/table.tsv`, `DEDUP_CAP.md`.

| law | space | scope | gap closed | interval | D_law | D_depth_scaling | attainable floor | gate (d) |
|---|---|---|---|---|---|---|---|---|
| shared-weight cap | start | top 1% | -0.56 | -2.68 .. 0.33 | 7.74e7 | 5.76e7 | 2.20e7 | no |
| count saturation | count | top 1% | -1.91 | -4.97 .. -0.38 | 1.77e8 | 6.71e7 | 9.48e6 | no |
| shared-weight cap | start | genome | -0.68 | -2.97 .. 0.47 | 2.19e8 | 1.65e8 | 8.63e7 | no |
| count saturation | count | genome | -2.75 | -7.17 .. -0.22 | 7.15e8 | 2.66e8 | 1.03e8 | no |

- ==The shared-weight cap beat the count law. The gap difference was 1.35 in the top 1% (interval
  0.29 .. 3.41) and 2.07 genome-wide (0.59 .. 4.47).==
- ==Per pair, the cap had a median top-1% gap of 0.69 and reached 0.8 on 19 of 43 pairs. A few deep
  multi-unit pairs made the pooled gap negative.==
- ==DNase was the only class that the cap explained (one `val` pair at 0.97; `train` 0.96, interval
  0.88 .. 0.98). ATAC-seq and PE DNase-seq fits sat at the upper shape bound, i.e. no shape.==
- ==Verdict proposed by C10c: mixed. The comparative claim held; the cap did not explain the pairs.==

## 4. SE/PE cap: PE parent thinned to one mate (h60, SE views)

Data only, no seeds. 12 PE parents (all `train`, leave-one-experiment-out fits). Interval = 95%
bootstrap over experiments. Source: `sepe_cap/table.tsv`, `SEPE_CAP.md`.

| law | space | scope | gap closed vs depth scaling | interval | D_law | D_depth_scaling | attainable floor | gate (d) | beats other law |
|---|---|---|---|---|---|---|---|---|---|
| `thinned_dm_distinct`, q = 0.5 | start | top 1% | 0.365 | -0.073 .. 0.548 | 2.08e7 | 2.29e7 | 1.72e7 | no | no |
| count saturation, 0.5 x parent | count | top 1% | -0.117 | -0.443 .. 0.482 | 2.30e7 | 2.13e7 | 6.83e6 | no | no |
| `thinned_dm_distinct`, q = 0.5 | start | genome | -0.003 | -0.054 .. 0.030 | 6.24e8 | 6.23e8 | 4.50e8 | no | no |
| count saturation, 0.5 x parent | count | genome | -0.051 | -0.139 .. 0.015 | 7.60e8 | 7.52e8 | 5.81e8 | no | no |

- ==Per pair, the start-space law was below depth scaling in 11 of 12 pairs, with a median ratio
  law / depth scaling of 0.957. Depth scaling already sits near the floor (median 0.890 of its
  deviance).==
- ==The 9 mixed parents are not interpreted: the floor was above depth scaling in 7 of 9, because
  q = 0.5 is wrong for the SE replicates of a mixed experiment.==
- ==The pooled shape `a` was unstable under leave-one-experiment-out (2.34 to 998.8 over 21 fits).==
- ==The PE dedup removed a median of about 20% (PE parents) to 23% (mixed parents) of the reads in
  the top 1% of bins (`pe_cap.tsv`).==

## 5. PE depth: what `log2_depth` counts (h61)

Data only, exact counts, so there is no floor and no seed range. Source: `pe_depth/summary.tsv`,
`PE_DEPTH.md`.

| run type | rows read (not blind) | table depth_reads / depth_fragments | store lines = depth_reads | lines_per_fragment = table ratio | plus strand = minus strand | gate outcome |
|---|---|---|---|---|---|---|
| single-ended | 2,469 | 1.0 on 2,540 of 2,540 | 2,469 of 2,469 | 2,469 of 2,469 | 0 | no gate set |
| paired-ended | 121 | 2.0 on 115, 1.0 on 13 | 121 of 121 | 121 of 121 | 111 | no gate set |
| mixed | 72 | 2.0 on 58, 1.0 on 14 | 72 of 72 | 72 of 72 | 0 | no gate set |

- ==For every PE row, `depth_reads` counts tagAlign lines, i.e. both mates. The count and start
  stores count the same lines. The 13 PE rows at ratio 1.0 are DNase-seq rows that also write one
  line per mate (moderate evidence; no read names).==
- ==Verdict: the h61 premise that our counts see half the recorded depth does not hold. What MACS2
  received for PE was not determined.==

## 6. Dedup at 1 bp: mechanism and dedup unit

Data only, exact counts on 6 `train` SE experiments (9 replicate units). No bootstrap was computed.
Source: `dedup_1bp/UNIT_RULE.json`, `mechanism.tsv`, `shape.tsv`, `DEDUP_1BP.md`.

| quantity | value | floor or spread | outcome | source |
|---|---|---|---|---|
| dedup-on = distinct dedup-off positions, per replicate | 9 of 9 units; 0 positions absent either way | exact | `dedup_unit` = `replicate` (tolerance 1e-3) | `UNIT_RULE.json`, `mechanism.tsv` |
| repeated-position share of pooled 2-replicate dedup-on | 0.0021, 0.0037, 0.0263 | exact | dedup ran per replicate | `mechanism.tsv` |
| 25 bp `dm_distinct` error on top-1% dedup-on sum, replicate units | -0.004 .. +0.027 | not set next to a floor | 25 bp store suffices for the top-1% mean | `shape.tsv` |
| same, pooled views with K = 50 | -0.006 .. +0.016 | not set next to a floor | -- | `shape.tsv` |
| same, pooled view treated as one unit | 0.023 .. 0.115 | not set next to a floor | -- | `shape.tsv` |
| occupancy (uniform positions) error | 0.023 .. 2.75 | not set next to a floor | a positional shape is necessary | `shape.tsv` |

## 7. Pair census for the widened correction headline (R39)

Counts from the tables and `evalset_p16.json`. Exact counts, so there is no floor and no bootstrap.
Source: `pair_census/by_covariate.tsv`, `PAIR_CENSUS.md`.

| covariate | candidate `val` pairs | of which headline now | experiments | outcome |
|---|---|---|---|---|
| dedup (`nodedup` in knobs) | 458 | 84 | 43 | usable |
| SE (`se` in knobs) | 20 | 4 | 2 | no usable spread over experiments |
| union | 476 | 88 | 44 | -- |

- ==The headline now uses 102 pairs. The widened set adds 388 pairs. Every candidate has both halves
  in the read-count store and both members in the read-start store.==

## Compute spent

CPU-h on Fir. Two measures occur: allocated (Elapsed x AllocCPUS, or `CPUTimeRAW`) and used
(`TotalCPU`). Decision D16 counts used CPU time for the plan's bars.

| work | jobs | measure | CPU-h | source |
|---|---|---|---|---|
| dump arrays (truth + 3 model seeds) | 6731315-18 | used / allocated | 13.5 / 88.2 | `dump/DUMP.md` |
| dump probes | 6727795, 6727796 | allocated | 2.72 | `sacct -X -o CPUTimeRAW`, run by C16 (2.47 + 0.25) |
| leg B scoring (2 probes, 6 arrays, 26 tasks) | `legB_spaces/jobs.tsv` | allocated | 16.11 | `legB_spaces/LEG_B.md` |
| leg A fit and score (2 probes, 3 chains) | `legA_covariance/jobs.tsv` | allocated | 3.31 | `sacct -X -o CPUTimeRAW`, run by C16 (0.27 + 0.11 + 0.58 + 0.35 + 0.72 + 0.30 + 0.68 + 0.30) |
| cap statistics arrays (C10b) | 6736415, 6737535, 6736416, 6737536 | allocated | 4.05 | `sacct -X -o CPUTimeRAW`, run by C16 (0.07 + 3.53 + 0.08 + 0.37) |
| dedup cap fit, floor, score, probes | `dedup_cap/jobs_c10c.tsv` | used | 3.96 | `dedup_cap/DEDUP_CAP.md` |
| SE/PE cap fit, floor, score | `sepe_cap/jobs_c11.tsv` | allocated | 2.24 | `sepe_cap/SEPE_CAP.md` |
| dedup at 1 bp (7 jobs, 1 OOM) | `dedup_1bp/jobs.tsv` | used | 0.55 | `dedup_1bp/cost.tsv` (0.0690 + 0.3065 + 0.0516 + 0.0492 + 0.0265 + 0.0192 + 0.0283) |
| table audits (pe_depth, pair_census) | 6727207, 6731380 | elapsed x 1 CPU | 0.56 | `pe_depth/PE_DEPTH.md` |
| tar extraction | 6726827 | allocated | 0.01 | `sacct`, run by C16 |

Sum of the column as written: 88.2 + 2.72 + 16.11 + 3.31 + 4.05 + 3.96 + 2.24 + 0.55 + 0.56 + 0.01
= 121.7 CPU-h. ==This mixes allocated and used time. The dump used about 5% of its allocated cores,
so the used total is far lower (dump used time 13.5 against 88.2 allocated).==

## What iterations 2 and 3 take from these results

Iteration 2 is the 2 x 2 factorial: input start or count x target start or count, 160,000 windows,
3 seeds, 256 regions. Iteration 3 is a cap output layer.

- ==**Foreman decision D17.** The leg B start-space test by deconvolution is void, because gate (c)
  failed and the deconvolution amplifies the count noise. The fair start-space test of the
  read-count model is the factorial arm "count input, start target". Iteration 2 does not tune the
  deconvolution.==
- ==**Count spaces.** The read-count reference leads in `overlap_count` and `sum_1kb_count` beyond
  seed noise. The factorial must show which factor (input or target) carries this lead. Score
  H3K36me3 as its own row, because it carries about half of the `overlap_count` gap.==
- ==**Conversion.** Leg A showed that the covariance term calibrates (1.258 to 1.032, all_assays).
  Iteration 2 uses it in the start-to-count conversion of every start-output arm (plan "Iteration 2
  prep", item 6).==
- ==**Depth covariate.** h61 did not hold: `log2_depth` and the stores count the same mates. The PE
  depth covariate needs no change for the factorial.==
- ==**Dump.** The model dump peaked at 19.5 GiB against a 20000M limit. The iteration-2 dump of every
  arm needs more memory, and `--cpus-per-task` sized to measured use (D16).==
- ==**Correction headline.** The widened set adds 388 dedup and SE pairs. The SE part rests on 2
  experiments, so an SE number has no spread over experiments.==
- ==**Iteration 3 (cap layer).** Use the start-space cap, not the count saturation law: it beat the
  count law on the dedup pairs. Set the dedup unit to the replicate (K = 25 U). One shape per assay
  did not explain the `val` pairs, and the failures sit on deep multi-unit pairs. A per-library
  shape, or a learned shape, is the next candidate. For SE views of mixed experiments, q = 0.5 is
  wrong; the layer needs the SE / parent read share per replicate. A full 1 bp store would cost
  109.0 to 225.5 CPU-h (`dedup_1bp/cost.tsv`), and iteration 1 showed that the 25 bp store suffices
  for the top-1% mean.==

## Open defect

==Read-start conversion stitches each forward batch separately (`tools/minicandi/rs_adapter.py`).
The converted output thus depends on how the regions are batched. It touches <= 0.1% of bins and has
no fixed sign (foreman review of C6, plan Log). C8 and C9 converted per forward batch, so the
iteration-1 numbers are consistent with the stored predictions. The adapter fix is a separate task.==

## C13 to-do items for iteration 2 (plan Log, C13 entry)

1. ==Route `kernel_head` in `train.py` without `read_start`.==
2. ==Stage `counts.h5` under `--stage-dir`.==
3. ==Check parameter-count parity across the 4 arms.==
4. ==Replace the `count_in_start_out` config, which is a `_todo` placeholder.==
5. ==Run a real-scale probe per arm.==

## Not checked by C16

- ==The sacct figures that C16 measured (dump probes, leg A, cap statistics arrays, tar extraction)
  are allocated time only; their used `TotalCPU` was not read.==
- ==The Fir manifest rows cover only the 24 leg B records under `/project`. The dump arrays and the
  cap statistics files stay on `/scratch` (purgeable) as intermediates and have no manifest row.==
