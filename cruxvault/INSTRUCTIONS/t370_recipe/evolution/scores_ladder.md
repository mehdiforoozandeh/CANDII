# Screening scores: the rung ladder (base arms) and the baselines

Label: **screening score: 256 regions for imputation items, 16 regions for pair tests** (evaluation set `/project/def-maxwl/mforooz/ONION/eval/eic_v1/evalset_p16.json`, sha256 a82cdfd1252d42ef673501eb155a9fb3140d3dce1c1d6f2b7f5a024b298b4c41, read from every scores.json; dataset eic_v1; regime leave_one_assay_out).

- Built 2026-11-15T07:18:36Z (UTC, `date -u` on Nibi at the start of the run) by builder SH2: a rerun of `tables_tb2.sh` (kit TB2), then `build_tables_tb2.py` and `notes_tb2.py` run on a copy of `out/` with one change: the rung-4 arm list includes `rung4_hard_offset` (the patched copy and the outputs are in `sh2_refresh/` beside this note). Scored runs at that time: 58 `SCORED` files under `$O/runs/*/seed*/p16/` and `$O/runs_region/*/seed*/p16/` (list: `out/scored_list.txt`).
- Kit: `/project/def-maxwl/mforooz/onion_kit/TB2` (git 141a0fd, dirty=0, pushed 2026-11-15T02:17:05Z).
- Composite version: 2 for every score in this note (checked: `out/checks_tb2.tsv`, "composite versions present" = 2 over 62 scores). Composite version 2 = 0.5 x specific variance + 0.25 x gap closed (oracle floor, all headline pairs) + 0.25 x uncertainty term, each term clipped to [0, 1] (`select_rule.py`, PI ruling 47; `cruxvault/results/oracle_floor/ORACLE_FLOOR.md`).
- All files named `out/...` are in `/scratch/mforooz/onion/tb2/out/` on Nibi. `$O` = `/project/def-maxwl/mforooz/ONION`.

Rung 1 was scored from GPU predictions with batch 32; rungs 2 to 6 and the baselines from CPU predictions with batch 8. The tools compare them with `--allow-mix` on device and batch; GPU and CPU predictions of one model differ by up to about 1e-5 in a score (`slurm/onion_eval/score.sh` header). Parameters and training FLOPs per rung are in the previous version of this note and in `/scratch/mforooz/onion/mtb/out/runjson.tsv`; they did not change. The rung-4 controls moved to `scores_rung4_arms.md`.

## Missing runs

- None for the ladder: rungs 1 to 6 base have 3 scored seeds each. Each baseline has one seed (seed 0), as designed. Depth-only scaling and the pseudoreplicate floor have no imputation term and so no composite.

## Reading

- ==Under composite version 2, rung 2 has the highest seed mean (0.5021, spread 0.0309), then rung 3 (0.4927), rung 5 (0.4690), rung 4 (0.4444), rung 1 (0.4439) and rung 6 (0.2605).== Under the superseded composite (version 1), rung 4 had the highest mean (0.6198; previous version of this note); version 2 no longer clips the correction term of every run to 1.
- ==Rung 2 is higher than rung 1 on every column by more than the larger seed spread== (composite +0.0581 against 0.0309; gap closed, oracle +0.1018 against 0.0230).
- ==Rung 3 differs from rung 2 by less than the seed spread on every column except the no-nodedup gap closed== (-0.0398 against 0.0286).
- ==Rung 4 trades correction for imputation:== specific variance +0.0511 over rung 3 (spread 0.0134), gap closed (oracle) -0.2734 (spread 0.0519), composite -0.0483 (spread 0.0326). All three exceed the seed spread.
- Rung 5 against rung 4: composite +0.0246 against a spread of 0.0297 (does not exceed); specific variance -0.0495 (0.0134, exceeds); gap closed (oracle) +0.1491 (0.0519, exceeds).
- ==Rung 6 is lower than rung 5 on every column by more than the seed spread== (specific variance -0.2360 against 0.0641; composite -0.2085 against 0.0516).
- ==Every rung is above the position average on composite version 2 (0.1601) and on gap closed (oracle) (0.2349).== The position average scores 0.2349 on all pairs; marginal NB scores -0.2651.

## Tables

### Seed mean (seed spread = max - min)

| row | n seeds | specific variance recovered above noise (counts_log1p) | correction CRPS | correction gap closed, oracle floor (all 102 pairs) | correction gap closed, oracle floor (18 pairs, no nodedup side) | correction gap closed, half-read floor (superseded) | composite, version 2 | composite_version |
|---|---|---|---|---|---|---|---|---|
| rung1_base | 3 | 0.3431 (0.0029) | 0.4971 (0.0034) | 0.4743 (0.0230) | 0.5930 (0.0309) | 5.3386 (0.4062) | 0.4439 (0.0086) | 2 |
| rung2_base | 3 | 0.3850 (0.0297) | 0.4754 (0.0027) | 0.5761 (0.0125) | 0.7187 (0.0123) | 6.5296 (0.2156) | 0.5021 (0.0309) | 2 |
| rung3_base | 3 | 0.3885 (0.0128) | 0.4772 (0.0026) | 0.5718 (0.0105) | 0.6789 (0.0286) | 6.4174 (0.1930) | 0.4927 (0.0326) | 2 |
| rung4_base | 3 | 0.4397 (0.0134) | 0.5309 (0.0114) | 0.2984 (0.0519) | 0.5185 (0.0530) | 2.9784 (0.5925) | 0.4444 (0.0297) | 2 |
| rung5_base | 3 | 0.3901 (0.0094) | 0.5010 (0.0082) | 0.4474 (0.0490) | 0.5793 (0.0146) | 4.6741 (0.5848) | 0.4690 (0.0297) | 2 |
| rung6_base | 3 | 0.1541 (0.0641) | 0.5421 (0.0107) | 0.2907 (0.0439) | 0.4445 (0.0427) | 4.0279 (0.4294) | 0.2605 (0.0516) | 2 |
| baseline position_average | 1 | 0.1190 | 0.5563 | 0.2349 | 0.3835 | 3.3519 | 0.1601 | 2 |
| baseline depth_only_scaling | 1 |  | 0.6014 | 0.0000 | 0.0000 | 0.0000 |  | 2 |
| baseline marginal_nb | 1 | 0.0000 | 0.6559 | -0.2651 | 0.2069 | -2.3528 | 0.0000 | 2 |
| baseline pseudoreplicate_floor | 1 |  | 0.5626 | 0.1508 | 0.0766 | 1.0000 |  | 2 |

### Per seed

| row | seed | specific variance recovered above noise (counts_log1p) | correction CRPS | correction gap closed, oracle floor (all 102 pairs) | correction gap closed, oracle floor (18 pairs, no nodedup side) | correction gap closed, half-read floor (superseded) | composite, version 2 | composite_version |
|---|---|---|---|---|---|---|---|---|
| rung1_base | 0 | 0.3416 | 0.4956 | 0.4844 | 0.5907 | 5.5290 | 0.4490 | 2 |
| rung1_base | 1 | 0.3445 | 0.4968 | 0.4771 | 0.6096 | 5.3639 | 0.4404 | 2 |
| rung1_base | 2 | 0.3432 | 0.4990 | 0.4614 | 0.5787 | 5.1229 | 0.4423 | 2 |
| rung2_base | 0 | 0.3653 | 0.4756 | 0.5717 | 0.7230 | 6.4435 | 0.4881 | 2 |
| rung2_base | 1 | 0.3948 | 0.4766 | 0.5723 | 0.7106 | 6.4861 | 0.5189 | 2 |
| rung2_base | 2 | 0.3950 | 0.4739 | 0.5842 | 0.7224 | 6.6592 | 0.4992 | 2 |
| rung3_base | 0 | 0.3934 | 0.4783 | 0.5667 | 0.6635 | 6.2990 | 0.5056 | 2 |
| rung3_base | 1 | 0.3806 | 0.4758 | 0.5773 | 0.6920 | 6.4920 | 0.4730 | 2 |
| rung3_base | 2 | 0.3916 | 0.4774 | 0.5714 | 0.6812 | 6.4611 | 0.4993 | 2 |
| rung4_base | 0 | 0.4480 | 0.5338 | 0.2855 | 0.5191 | 2.7748 | 0.4306 | 2 |
| rung4_base | 1 | 0.4347 | 0.5238 | 0.3308 | 0.5447 | 3.3673 | 0.4603 | 2 |
| rung4_base | 2 | 0.4363 | 0.5352 | 0.2789 | 0.4917 | 2.7930 | 0.4423 | 2 |
| rung5_base | 0 | 0.3839 | 0.5061 | 0.4186 | 0.5875 | 4.3839 | 0.4675 | 2 |
| rung5_base | 1 | 0.3933 | 0.4989 | 0.4560 | 0.5729 | 4.6695 | 0.4549 | 2 |
| rung5_base | 2 | 0.3931 | 0.4980 | 0.4677 | 0.5774 | 4.9688 | 0.4847 | 2 |
| rung6_base | 0 | 0.1764 | 0.5351 | 0.3192 | 0.4698 | 4.2995 | 0.2927 | 2 |
| rung6_base | 1 | 0.1737 | 0.5452 | 0.2775 | 0.4364 | 3.9141 | 0.2477 | 2 |
| rung6_base | 2 | 0.1123 | 0.5459 | 0.2753 | 0.4271 | 3.8701 | 0.2410 | 2 |
| baseline position_average | 0 | 0.1190 | 0.5563 | 0.2349 | 0.3835 | 3.3519 | 0.1601 | 2 |
| baseline depth_only_scaling | 0 |  | 0.6014 | 0.0000 | 0.0000 | 0.0000 |  | 2 |
| baseline marginal_nb | 0 | 0.0000 | 0.6559 | -0.2651 | 0.2069 | -2.3528 | 0.0000 | 2 |
| baseline pseudoreplicate_floor | 0 |  | 0.5626 | 0.1508 | 0.0766 | 1.0000 |  | 2 |

## Adjacent rungs (rung k minus rung k-1)

### Summary: mean paired difference (arm minus reference); [larger seed spread]; exceeds?

| arm | n seeds | specific variance recovered above noise (counts_log1p) | correction CRPS | correction gap closed, oracle floor (all 102 pairs) | correction gap closed, oracle floor (18 pairs, no nodedup side) | correction gap closed, half-read floor (superseded) | composite, version 2 | composite_version |
|---|---|---|---|---|---|---|---|---|
| rung2_base | 3 | +0.0419 [0.0297] yes | -0.0218 [0.0034] yes | +0.1018 [0.0230] yes | +0.1256 [0.0309] yes | +1.1910 [0.4062] yes | +0.0581 [0.0309] yes | 2/2 |
| rung3_base | 3 | +0.0035 [0.0297] no | +0.0018 [0.0027] no | -0.0043 [0.0125] no | -0.0398 [0.0286] yes | -0.1122 [0.2156] no | -0.0094 [0.0326] no | 2/2 |
| rung4_base | 3 | +0.0511 [0.0134] yes | +0.0538 [0.0114] yes | -0.2734 [0.0519] yes | -0.1604 [0.0530] yes | -3.4390 [0.5925] yes | -0.0483 [0.0326] yes | 2/2 |
| rung5_base | 3 | -0.0495 [0.0134] yes | -0.0299 [0.0114] yes | +0.1491 [0.0519] yes | +0.0608 [0.0530] yes | +1.6957 [0.5925] yes | +0.0246 [0.0297] no | 2/2 |
| rung6_base | 3 | -0.2360 [0.0641] yes | +0.0411 [0.0107] yes | -0.1568 [0.0490] yes | -0.1348 [0.0427] yes | -0.6462 [0.5848] yes | -0.2085 [0.0516] yes | 2/2 |

### rung2_base minus rung1_base (paired seeds: 0,1,2; seeds: rung1_base 3, rung2_base 3)

| metric | rung1_base mean (spread) | rung2_base mean (spread) | diff s0 | diff s1 | diff s2 | mean diff | spread of diff | larger seed spread | exceeds seed spread | composite_version (arm/ref) |
|---|---|---|---|---|---|---|---|---|---|---|
| specific variance recovered above noise (counts_log1p) | 0.3431 (0.0029) | 0.3850 (0.0297) | +0.0237 | +0.0503 | +0.0518 | +0.0419 | 0.0281 | 0.0297 | yes | 2/2 |
| correction CRPS | 0.4971 (0.0034) | 0.4754 (0.0027) | -0.0200 | -0.0202 | -0.0251 | -0.0218 | 0.0051 | 0.0034 | yes | 2/2 |
| correction gap closed, oracle floor (all 102 pairs) | 0.4743 (0.0230) | 0.5761 (0.0125) | +0.0873 | +0.0952 | +0.1228 | +0.1018 | 0.0355 | 0.0230 | yes | 2/2 |
| correction gap closed, oracle floor (18 pairs, no nodedup side) | 0.5930 (0.0309) | 0.7187 (0.0123) | +0.1322 | +0.1010 | +0.1437 | +0.1256 | 0.0427 | 0.0309 | yes | 2/2 |
| correction gap closed, half-read floor (superseded) | 5.3386 (0.4062) | 6.5296 (0.2156) | +0.9145 | +1.1222 | +1.5363 | +1.1910 | 0.6218 | 0.4062 | yes | 2/2 |
| composite, version 2 | 0.4439 (0.0086) | 0.5021 (0.0309) | +0.0390 | +0.0785 | +0.0569 | +0.0581 | 0.0394 | 0.0309 | yes | 2/2 |

### rung3_base minus rung2_base (paired seeds: 0,1,2; seeds: rung2_base 3, rung3_base 3)

| metric | rung2_base mean (spread) | rung3_base mean (spread) | diff s0 | diff s1 | diff s2 | mean diff | spread of diff | larger seed spread | exceeds seed spread | composite_version (arm/ref) |
|---|---|---|---|---|---|---|---|---|---|---|
| specific variance recovered above noise (counts_log1p) | 0.3850 (0.0297) | 0.3885 (0.0128) | +0.0281 | -0.0142 | -0.0034 | +0.0035 | 0.0424 | 0.0297 | no | 2/2 |
| correction CRPS | 0.4754 (0.0027) | 0.4772 (0.0026) | +0.0027 | -0.0008 | +0.0035 | +0.0018 | 0.0043 | 0.0027 | no | 2/2 |
| correction gap closed, oracle floor (all 102 pairs) | 0.5761 (0.0125) | 0.5718 (0.0105) | -0.0050 | +0.0049 | -0.0128 | -0.0043 | 0.0178 | 0.0125 | no | 2/2 |
| correction gap closed, oracle floor (18 pairs, no nodedup side) | 0.7187 (0.0123) | 0.6789 (0.0286) | -0.0595 | -0.0186 | -0.0412 | -0.0398 | 0.0409 | 0.0286 | yes | 2/2 |
| correction gap closed, half-read floor (superseded) | 6.5296 (0.2156) | 6.4174 (0.1930) | -0.1445 | +0.0059 | -0.1980 | -0.1122 | 0.2040 | 0.2156 | no | 2/2 |
| composite, version 2 | 0.5021 (0.0309) | 0.4927 (0.0326) | +0.0176 | -0.0459 | +0.0001 | -0.0094 | 0.0635 | 0.0326 | no | 2/2 |

### rung4_base minus rung3_base (paired seeds: 0,1,2; seeds: rung3_base 3, rung4_base 3)

| metric | rung3_base mean (spread) | rung4_base mean (spread) | diff s0 | diff s1 | diff s2 | mean diff | spread of diff | larger seed spread | exceeds seed spread | composite_version (arm/ref) |
|---|---|---|---|---|---|---|---|---|---|---|
| specific variance recovered above noise (counts_log1p) | 0.3885 (0.0128) | 0.4397 (0.0134) | +0.0546 | +0.0541 | +0.0447 | +0.0511 | 0.0100 | 0.0134 | yes | 2/2 |
| correction CRPS | 0.4772 (0.0026) | 0.5309 (0.0114) | +0.0554 | +0.0480 | +0.0578 | +0.0538 | 0.0098 | 0.0114 | yes | 2/2 |
| correction gap closed, oracle floor (all 102 pairs) | 0.5718 (0.0105) | 0.2984 (0.0519) | -0.2813 | -0.2465 | -0.2925 | -0.2734 | 0.0460 | 0.0519 | yes | 2/2 |
| correction gap closed, oracle floor (18 pairs, no nodedup side) | 0.6789 (0.0286) | 0.5185 (0.0530) | -0.1444 | -0.1473 | -0.1895 | -0.1604 | 0.0451 | 0.0530 | yes | 2/2 |
| correction gap closed, half-read floor (superseded) | 6.4174 (0.1930) | 2.9784 (0.5925) | -3.5242 | -3.1247 | -3.6681 | -3.4390 | 0.5435 | 0.5925 | yes | 2/2 |
| composite, version 2 | 0.4927 (0.0326) | 0.4444 (0.0297) | -0.0751 | -0.0128 | -0.0570 | -0.0483 | 0.0623 | 0.0326 | yes | 2/2 |

### rung5_base minus rung4_base (paired seeds: 0,1,2; seeds: rung4_base 3, rung5_base 3)

| metric | rung4_base mean (spread) | rung5_base mean (spread) | diff s0 | diff s1 | diff s2 | mean diff | spread of diff | larger seed spread | exceeds seed spread | composite_version (arm/ref) |
|---|---|---|---|---|---|---|---|---|---|---|
| specific variance recovered above noise (counts_log1p) | 0.4397 (0.0134) | 0.3901 (0.0094) | -0.0641 | -0.0413 | -0.0432 | -0.0495 | 0.0228 | 0.0134 | yes | 2/2 |
| correction CRPS | 0.5309 (0.0114) | 0.5010 (0.0082) | -0.0277 | -0.0249 | -0.0373 | -0.0299 | 0.0124 | 0.0114 | yes | 2/2 |
| correction gap closed, oracle floor (all 102 pairs) | 0.2984 (0.0519) | 0.4474 (0.0490) | +0.1332 | +0.1253 | +0.1888 | +0.1491 | 0.0635 | 0.0519 | yes | 2/2 |
| correction gap closed, oracle floor (18 pairs, no nodedup side) | 0.5185 (0.0530) | 0.5793 (0.0146) | +0.0684 | +0.0282 | +0.0857 | +0.0608 | 0.0575 | 0.0530 | yes | 2/2 |
| correction gap closed, half-read floor (superseded) | 2.9784 (0.5925) | 4.6741 (0.5848) | +1.6092 | +1.3022 | +2.1758 | +1.6957 | 0.8735 | 0.5925 | yes | 2/2 |
| composite, version 2 | 0.4444 (0.0297) | 0.4690 (0.0297) | +0.0369 | -0.0053 | +0.0423 | +0.0246 | 0.0477 | 0.0297 | no | 2/2 |

### rung6_base minus rung5_base (paired seeds: 0,1,2; seeds: rung5_base 3, rung6_base 3)

| metric | rung5_base mean (spread) | rung6_base mean (spread) | diff s0 | diff s1 | diff s2 | mean diff | spread of diff | larger seed spread | exceeds seed spread | composite_version (arm/ref) |
|---|---|---|---|---|---|---|---|---|---|---|
| specific variance recovered above noise (counts_log1p) | 0.3901 (0.0094) | 0.1541 (0.0641) | -0.2075 | -0.2197 | -0.2808 | -0.2360 | 0.0733 | 0.0641 | yes | 2/2 |
| correction CRPS | 0.5010 (0.0082) | 0.5421 (0.0107) | +0.0290 | +0.0462 | +0.0479 | +0.0411 | 0.0189 | 0.0107 | yes | 2/2 |
| correction gap closed, oracle floor (all 102 pairs) | 0.4474 (0.0490) | 0.2907 (0.0439) | -0.0995 | -0.1785 | -0.1924 | -0.1568 | 0.0929 | 0.0490 | yes | 2/2 |
| correction gap closed, oracle floor (18 pairs, no nodedup side) | 0.5793 (0.0146) | 0.4445 (0.0427) | -0.1177 | -0.1365 | -0.1502 | -0.1348 | 0.0325 | 0.0427 | yes | 2/2 |
| correction gap closed, half-read floor (superseded) | 4.6741 (0.5848) | 4.0279 (0.4294) | -0.0844 | -0.7555 | -1.0987 | -0.6462 | 1.0142 | 0.5848 | yes | 2/2 |
| composite, version 2 | 0.4690 (0.0297) | 0.2605 (0.0516) | -0.1748 | -0.2072 | -0.2436 | -0.2085 | 0.0688 | 0.0516 | yes | 2/2 |

## Definitions

- Seed mean = the mean over the scored seeds. Seed spread = maximum minus minimum over the scored seeds (the select_rule definition). A paired difference = (arm value) minus (reference value) for the same training seed.
- "Exceeds seed spread" = yes when |mean paired difference| is greater than the larger of the two seed spreads (arm and reference). An arm with one seed is not judged. A spread over two seeds is a weak estimate of seed noise.
- Specific variance recovered above noise = the fraction of the cell-type-specific variance that the prediction recovers above the pseudoreplicate-halves noise floor (counts_log1p, macro over the 9 imputation-scorable assays). Higher is better.
- Correction CRPS = the continuous ranked probability score of the predicted count distribution on the correction pairs (macro over assays). Lower is better. Its `oracle_scaled` / `scale_error` split is in each scores.json (`correction_crps_model_oracle_scaled`) and in the previous notes; this note does not repeat it.
- Gap closed (oracle floor) = (CRPS depth-only - CRPS method) / (CRPS depth-only - oracle floor), macro over the 11 correction-scorable assays, on all 102 headline pairs. 0 = depth-only scaling; 1 = the oracle floor. The second column uses only the 18 headline pairs where no side keeps duplicates (6 assays), where the oracle floor is a strict bound.
- Gap closed, half-read floor (superseded) = the old correction term (PI ruling 47 replaced it). It is shown once, for comparison with earlier notes.
- The noise floor: imputation numbers use the pseudoreplicate-halves floor (`noise_floor: pseudoreplicate_halves`); the correction numbers use the oracle floor above.

## Second method

- Every value in the tables was compared with a second path (`out/checks_tb2.tsv`). Specific variance and the half-read value from `select_rule.py table` agreed with a plain `json` read of each scores.json within 5e-7 and 5e-6 (58 runs; print rounding). Gap closed (oracle, both subsets) and composite version 2 agreed exactly (difference 0) between `select_rule.py table` and `oracle_floor.py report`. Correction CRPS from the plain read agreed with `oracle_floor.py report` within 5e-7 (62 scores). Seed means and spreads from `build_tables_tb2.py` agreed with `arm_report.py headline` within 8e-6 (21 arms).
- Row `rung2_base` seed 1 read with a separate `json.load` of `$O/runs/rung2_base/seed1/p16/scores.json` on 2026-11-15: specific variance 0.3948, correction CRPS 0.4766, half-read 6.4861; the table has 0.3948, 0.4766, 6.4861. Composite version 2 recomputed by hand from the `select_rule.py table` terms: 0.5 x 0.394801 + 0.25 x 0.572309 + 0.25 x 0.713821 = 0.518933, equal to the table (0.5189).

## Sources

Per seed: `$O/runs/<run>/seed<k>/p16/scores.json` and `pairs_per_region.npz`; per region-set run: `$O/runs_region/<run>/seed<k>/p16/`. Baselines: `$O/eval/eic_v1/baselines_p16/<name>/seed0/`. Oracle reference: `$O/eval/eic_v1/baselines_p16/oracle_floor/seed0/`. Joined per-seed values: `out/perseed_tb2.tsv` (with each scores.json path and the first 16 hex of its sha256).

## Refresh

```
hpc run nibi 'bash /scratch/mforooz/onion/tb2/tables_tb2.sh > /scratch/mforooz/onion/tb2/out/last_run.log 2>&1; cat /scratch/mforooz/onion/tb2/out/run_time.txt /scratch/mforooz/onion/tb2/out/checks_tb2.tsv'
hpc run nibi 'cat /scratch/mforooz/onion/tb2/out/notes/ladder_tables.md'   # the new tables; paste over the "Tables" section
```
The script is read-only on `$O`. It runs on the Nibi login node (as `tables_mtb.sh` did) in about 1 minute and writes only under `/scratch/mforooz/onion/tb2/out/`. It runs `select_rule.py table`, `arm_report.py headline`, `oracle_floor.py report` (kit TB2), a plain `json` read of every scores.json, then `build_tables_tb2.py` and `notes_tb2.py` (both in `/scratch/mforooz/onion/tb2/`). Update the header (time, count, missing runs) and the highlighted reading by hand.
