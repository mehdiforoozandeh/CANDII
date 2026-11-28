# t396 leg B: both models in three score spaces (h57)

Hand-written. Not an output of `run.sh`. Built 2026-11-28 by chunk C14 from branch tip `d03fff03`.

## What ran

- Models. Reference = the read-count model `rcp_halves_mix_avg_pearson_unet`. Read-start = `rs_halves_mix_avg_pearson_unet`. Both ran with seeds 0, 1, 2, on `evalset_p16.json` (43 targets, 256 regions, 16 pair regions).
- Spaces. `overlap_count` = 25 bp overlap counts. `start_sum` = strand-summed read starts per 25 bp bin. `sum_1kb_count` = sums of 40 overlap-count bins. The reference reaches `start_sum` by Richardson-Lucy deconvolution, i.e. an iterative non-negative inverse of the span kernel, at `n_iter` in {10, 25, 50, 100, 200}. The primary `n_iter` is 50.
- Jobs. `jobs.tsv` lists 2 probes and 6 arrays of 4 tasks. `sacct` showed all 26 tasks COMPLETED on 2026-11-28. The allocated cost was 16.11 CPU-h (`sacct -X -o CPUTimeRAW`, summed). The largest MaxRSS was 2.03 GB.
- Records. 24 jsonl files (3.8 MB) stay on Fir under `/project/def-maxwl/mforooz/t396/legB_spaces/records/`. Their manifest rows are at the end of this file. `run.sh` runs `score_spaces.py check` and `table` of kit `SC7b` on the Fir login node, and pulls the 5 outputs. The kit's `score_spaces.py`, `spaces.py`, `deconv.py` and `deviance.py` have the same sha256 as `d03fff03`.

## Rules for reading the numbers

- Compare models within one space only. A skill or a `ratio_to_halves` is not comparable across spaces.
- `skill` = deviance skill against the average reference (`deviance.gap`, `D_ref`), the macro over the 9 scorable assays. Higher is better.
- `ratio_to_halves` = mean(D_model_pr2 / D0_pr2) / mean(D_halves_pr2 / D0_pr2), i.e. the model's deviance on half 2 relative to half 1 predicting half 2. Lower is better; below 1 the model predicts half 2 better than half 1 does.
- `floor` = mean over assays of the Poisson plug-in floor divided by D0.
- Seed range = max - min of `skill` over seeds 0-2. The table holds no seed range for `ratio_to_halves`.
- Earlier context (t392, leg A): read-count won every term; deviance skill was 0.3746 for read-start and 0.5426 for read-count. This leg is not a re-derivation of that result.

## Gate (b): reproduction of scores.json

`check.tsv`: 6 of 6 runs pass. The largest absolute difference from `scores.json["deviance"]["dev_imputation_skill"]["macro"]` was 1.1e-16 (bar 0.002). The `overlap_count` / `all` rows therefore reproduce t392: 0.5426 for the reference and 0.3746 for read-start.

## Gate (c): deconvolution validation (`truth_validation.tsv`, n_iter 50)

Gate (c) **failed**. Two conditions failed.

1. Round trip `richardson_lucy(forward(pa_starts))` vs `pa_starts`, bar 0.05 per scorable assay. H3K36me3 had a median normalized MSE of 0.055 (3 targets). H3K9me3 had 0.144 (3 targets). The other 7 scorable assays passed (0.004 to 0.017).
2. Conversion floor with the strand-mixed kernel, bar 0.10 over all targets. The median was 0.133 (43 targets). 9 of the 22 assays were below 0.10.

Consequence: every reference row in `start_sum` carries `not_interpreted`. As the plan sets, h57 is read from `overlap_count` and `sum_1kb_count` only. The start-space numbers below are reported as uninterpreted.

Also measured, not a bar: Richardson-Lucy of the half-1 counts against the half-2 starts had a median normalized MSE of 1.59 at 50 iterations, against 1.05 for the undeconvolved `y_pr1 / w.sum()`. The error grew with `n_iter` (1.30 at 10, 1.70 at 200). Deconvolution amplified the count noise.

## Main table (`table.tsv`, macro = scorable, 256 regions)

| space | model | skill | seed range | per seed (0, 1, 2) | ratio_to_halves | floor | every reference seed above every read-start seed |
|---|---|---|---|---|---|---|---|
| overlap_count | reference | 0.543 | 0.065 | 0.543, 0.510, 0.575 | 0.304 | 0.516 | yes (0.510 > 0.425) |
| overlap_count | read-start | 0.375 | 0.087 | 0.425, 0.338, 0.362 | 0.322 | 0.525 | |
| start_sum | reference, n_iter 50 (primary, not_interpreted) | 0.588 | 0.067 | 0.592, 0.552, 0.620 | 0.357 | 0.711 | no (0.552 < 0.615) |
| start_sum | reference, n_iter 10 (best_over_grid, not_interpreted) | 0.650 | 0.070 | 0.650, 0.615, 0.685 | 0.352 | 0.706 | no (0.61473 < 0.61477) |
| start_sum | read-start | 0.578 | 0.072 | 0.615, 0.543, 0.577 | 0.358 | 0.713 | |
| sum_1kb_count | reference | 0.394 | 0.065 | 0.405, 0.357, 0.421 | 0.690 | 0.038 | yes (0.357 > 0.252) |
| sum_1kb_count | read-start | 0.184 | 0.118 | 0.252, 0.134, 0.164 | 0.877 | 0.038 | |

- `overlap_count`: the reference led by 0.168. This is above both seed ranges (0.065, 0.087). Its `ratio_to_halves` was lower (0.304 against 0.322).
- `sum_1kb_count`: the reference led by 0.211. This is above both seed ranges (0.065, 0.118). Its `ratio_to_halves` was lower (0.690 against 0.877).
- `start_sum` (not interpreted): at the primary `n_iter` the reference led by 0.010, far below both seed ranges. The `ratio_to_halves` values were equal to 0.001 (0.357, 0.358). At the best grid point (n_iter 10) the reference led by 0.072, which equals the seed ranges (0.070, 0.072); paired by seed it led on all 3 seeds, but its worst seed tied the best read-start seed.

## Sensitivity to n_iter (reference, start_sum, all regions)

Skill fell monotonically with `n_iter`: 0.650 (10), 0.623 (25), 0.588 (50, primary), 0.528 (100), 0.423 (200). The seed range was 0.051 to 0.070. `best_over_grid` is n_iter 10, at the edge of the grid. It is chosen after the fact, so it is an upper bound for the grid, and a smaller `n_iter` was not tested. The trend agrees with the noise amplification above.

## The 16 pair regions (`regions = pair`)

| space | reference skill (range) | read-start skill (range) | every reference seed above every read-start seed |
|---|---|---|---|
| overlap_count | 0.542 (0.036) | 0.409 (0.099) | yes (0.519 > 0.463) |
| start_sum, n_iter 50 (not_interpreted) | 0.614 (0.035) | 0.621 (0.070) | no |
| start_sum, n_iter 10 (not_interpreted) | 0.662 (0.039) | 0.621 (0.070) | no (0.639 < 0.656) |
| sum_1kb_count | 0.390 (0.049) | 0.201 (0.148) | yes (0.360 > 0.280) |

The pair regions gave the same order as the 256 regions in every space. The `even` region set agreed with `all` to within 0.014 in every row.

## Read length (`per_read_length.tsv`, 28 targets at 36 bp, 15 longer)

| space | group | reference skill (range) | read-start skill (range) |
|---|---|---|---|
| overlap_count | 36 bp | 0.407 (0.122) | 0.235 (0.173) |
| overlap_count | longer | 0.623 (0.019) | 0.399 (0.039) |
| start_sum, n_iter 50 (not_interpreted) | 36 bp | 0.434 (0.137) | 0.428 (0.124) |
| start_sum, n_iter 50 (not_interpreted) | longer | 0.714 (0.017) | 0.657 (0.035) |
| sum_1kb_count | 36 bp | 0.272 (0.122) | 0.086 (0.255) |
| sum_1kb_count | longer | 0.454 (0.018) | 0.226 (0.066) |

On the longer reads the seed ranges were small, and the reference led in every space by more than the seed range. In start space the lead (0.057) is uninterpreted. On 36 bp reads, the start-space difference (0.006) was far below the seed ranges.

## H3K36me3 and H3K27ac (`per_assay.tsv`, 3 targets each, all regions)

| space | assay | reference skill (range) | read-start skill (range) |
|---|---|---|---|
| overlap_count | H3K36me3 | 0.409 (0.055) | -0.428 (0.137) |
| overlap_count | H3K27ac | 0.317 (0.362) | 0.516 (0.189) |
| start_sum, n_iter 50 (not_interpreted) | H3K36me3 | 0.335 (0.051) | -0.109 (0.141) |
| start_sum, n_iter 50 (not_interpreted) | H3K27ac | 0.245 (0.482) | 0.610 (0.211) |
| sum_1kb_count | H3K36me3 | 0.191 (0.077) | -0.713 (0.106) |
| sum_1kb_count | H3K27ac | 0.302 (0.212) | 0.439 (0.102) |

- H3K36me3 gave read-start a negative skill in every space. In `overlap_count` it accounted for 0.093 of the 0.168 macro gap (0.837 / 9). Without H3K36me3, the 8-assay mean was 0.559 for the reference and 0.475 for read-start; the gap of 0.084 is about equal to the macro seed ranges.
- H3K27ac favored read-start in every space. The reference's seed range on H3K27ac (0.36 to 0.48) covers the difference.
- In `start_sum` at n_iter 50 without H3K36me3, read-start led (0.664 against 0.620). This row is uninterpreted.

## Calibration ratios

- The reference's `start_sum` calibration ratio (0.91 to 0.99) carries the flag `conversion_artifact`. Its NB size comes from `match_size` under an independence assumption, so the ratio describes the conversion, not the model.
- In `sum_1kb_count` the calibration ratio was 8.7 (reference) and 13.4 (read-start). The 40-bin pooled size is moment-matched under independence of bins. Neighboring bins are correlated, so this ratio is not interpretable. This was not investigated further.

## Not checked

- The seed range of `ratio_to_halves`. The table holds only the skill's seed range.
- The `odd` region set (not scored by C7b).
- A smaller `n_iter` than 10.
- Why H3K36me3 and H3K9me3 fail the round trip (broad domains are a candidate; not tested).

## Records on Fir (manifest rows for C16; columns path, bytes, sha256, where)

```
path	bytes	sha256	where
legB_spaces/records/read_start_seed0_0.jsonl	88178	e198520362f3c886db07c3e8d53f6f65c49ccc2137acf5af835422ef432b7b62	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/read_start_seed0_0.jsonl
legB_spaces/records/read_start_seed0_1.jsonl	88177	2c6a921f9342ce8831a1a09eb60d2203a05d370f96602722c991365bb3ba83f9	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/read_start_seed0_1.jsonl
legB_spaces/records/read_start_seed0_2.jsonl	88198	751b116f2a4bc2796affd57279a49e1939c4d2510df47d8f02b8ff9324b338e9	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/read_start_seed0_2.jsonl
legB_spaces/records/read_start_seed0_3.jsonl	80172	e7daf1afea0c278fee5874dfbcc21663f0616296275e069f2e0bcd59b2a22480	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/read_start_seed0_3.jsonl
legB_spaces/records/read_start_seed1_0.jsonl	88162	59ac2a8fda91381ee35fc4884824e1ad9d1190da5df928a9e4f79152cc97b9be	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/read_start_seed1_0.jsonl
legB_spaces/records/read_start_seed1_1.jsonl	88170	971436cdcc8c45cd7468fdc7d42035e62204baddaa6ffdfec81a38fc7af090b8	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/read_start_seed1_1.jsonl
legB_spaces/records/read_start_seed1_2.jsonl	88185	d1db08596bca559474d22650a1f3081476d628743d3579aaad808cac47936806	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/read_start_seed1_2.jsonl
legB_spaces/records/read_start_seed1_3.jsonl	80176	e9d75a2e066ec98d7f6c8b641fb25e3358e2c5850f6440b12973b9a5389c147d	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/read_start_seed1_3.jsonl
legB_spaces/records/read_start_seed2_0.jsonl	88065	9d542a399f5b3d0055fa65c351dcbc787f1d7f6befbcf567cf899e4341c78b04	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/read_start_seed2_0.jsonl
legB_spaces/records/read_start_seed2_1.jsonl	88096	cebaf5b0f5a6df23fe932e339ff117e42491e91debbb2f8d29263b2a741ae262	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/read_start_seed2_1.jsonl
legB_spaces/records/read_start_seed2_2.jsonl	88088	b8d9ae3451e8ad41242cb3f131c28338ef3ecf899a0aab094de440659503d283	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/read_start_seed2_2.jsonl
legB_spaces/records/read_start_seed2_3.jsonl	80066	7f508cd268d818136ab103e9a17d8d5bfbbc2928cbce47c70461b6b36eaab6ff	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/read_start_seed2_3.jsonl
legB_spaces/records/reference_seed0_0.jsonl	323014	ea8a114266a5f248864632a1cdf59f186ea1bf4b3679237aaf5a8cc3a2683dac	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/reference_seed0_0.jsonl
legB_spaces/records/reference_seed0_1.jsonl	323077	3225d00c27ae1917f3873c471fe73eb2dd77905f98a806c4382cbc4a6e73d4fe	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/reference_seed0_1.jsonl
legB_spaces/records/reference_seed0_2.jsonl	322970	5ebb6cc90bd66d64d7ccdd03c5c6c60e52d71f1c002b39e8a934574c922d8c0b	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/reference_seed0_2.jsonl
legB_spaces/records/reference_seed0_3.jsonl	293706	3d846a360001de3516ec34a42237c25c3f252a35d8f92539ae9bca92d992f772	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/reference_seed0_3.jsonl
legB_spaces/records/reference_seed1_0.jsonl	204827	ec3ee9afb0a0493713cac2c2a48806c8bdf01c5f64ee463ded6343195739f73b	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/reference_seed1_0.jsonl
legB_spaces/records/reference_seed1_1.jsonl	204814	792fd20f21ccc707071a05700d87c8ca50afebd0ba945c9ced40390d3c4bde1d	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/reference_seed1_1.jsonl
legB_spaces/records/reference_seed1_2.jsonl	204843	db50f64deb8bb184429f264aeff862b7ce7d737f5090f2bb6c53d22623a3db0e	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/reference_seed1_2.jsonl
legB_spaces/records/reference_seed1_3.jsonl	186277	02224a5fe2d587635ccf1d298207784dc033019a25ff5749eee9aca43e807d39	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/reference_seed1_3.jsonl
legB_spaces/records/reference_seed2_0.jsonl	204883	3a7e4679836ab81835528811ce6eff17282c06c21ea5bc01e91478b56c310f95	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/reference_seed2_0.jsonl
legB_spaces/records/reference_seed2_1.jsonl	204838	e2a012efc1e93c147863d73cf031cdb68a5d13941a50d9c6e7e81b66becfa7d5	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/reference_seed2_1.jsonl
legB_spaces/records/reference_seed2_2.jsonl	204857	5fe39cadcf5f8df404f24f913053cabd5b211850e088bb76644c0c498452ed05	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/reference_seed2_2.jsonl
legB_spaces/records/reference_seed2_3.jsonl	186266	5c2921d6bcaf1440db622365ad9f62812449b840b3d45d3663c860d49dea4186	fir:/project/def-maxwl/mforooz/t396/legB_spaces/records/reference_seed2_3.jsonl
```
