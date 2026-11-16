---
title: "Computed metrics report: the metrics the scorer computes and no report showed"
parent: "[[closeout E3 report the computed metrics]]"
created: 2026-11-16
tier: screening
---

# Computed metrics report

This report shows the metrics that the scorer already writes to `scores.json` and that no earlier report showed. No model was trained or scored for it. All values were read from stored files with read-only commands.

## 1. Source, scope and how to read the tables

- Evaluation set: the screening set of 16 regions (scorer sub-directory `p16_sv`), `/project/def-maxwl/mforooz/ONION/eval/eic_v1/evalset_p16.json`, `evalset_sha256` = `a82cdfd1252d42ef673501eb155a9fb3140d3dce1c1d6f2b7f5a024b298b4c41`. Every one of the 49 files read carries this hash. Tier: screening (not settled). Composite version: 3 (the correction term is the gap closed on crop pairs with the oracle reference). Scorer version 2 for every model file and version 1 for the four baseline files (the table tools accept a version-1 baseline as a reference).
- Models: 15 runs on Fir, 3 seeds each, 80,000 training windows: `runs_fir/<run>/seed{0,1,2}/p16_sv/scores.json`. The five base recipes are `rung2_base_heldout` (conv autoencoder), `rung3_base_heldout` (conv + attention), `rung4_base_heldout` (U-Net), `rung5_base_heldout` (axial transformer) and `rung6_base_heldout` (Perceiver IO). The other 10 runs are the final-round arms with equal windows (`final_*`).
- Baselines: `eval/eic_v1/baselines_p16/<name>/seed0/scores.json`, one seed each, so they have no range. Marginal NB is one negative binomial per assay with no position information. Position average is the training average at each position. Depth-only scaling is the source track scaled by the depth ratio (correction only). Half-read copy is the target's own other read half (`pseudoreplicate_floor`, correction only).
- Each model cell is "mean (range)": the mean over the 3 seeds and the range (maximum minus minimum) over the 3 seeds. "undefined" means the scorer wrote null for at least one seed. "absent" means the key does not exist in that file.
- Regime: every table is the leave-one-assay-out regime (43 imputation items, 9 scorable assays), except section 9 (accessibility-only).
- H3K9ac flag: the validation target of H3K9ac is the replicate view of ENCSR521AIV, which has counts and no signal products. H3K9ac is therefore scored in count space only (long-horizon notepad, 2026-11-13). This applies to every H3K9ac column below.
- Rebuild: `cruxvault/results/closeout_E3/` holds `extract_e3.py` (reads the 49 files, writes `tables/seed_values.tsv` with the source path of every value and `tables/summary.tsv` with the seed means and ranges) and `make_tables.py` (prints the tables). The same `extract_e3.py` run on Fir wrote `/project/def-maxwl/mforooz/ONION/closeout_E3/tables/`; its two outputs are byte-identical to the laptop run (sha256 `73420dbf...` for `seed_values.tsv`, `4a1815b9...` for `summary.tsv`). The per-knob table (section 8) and the headline table (section 2) come from `arm_report.py per-knob` and `select_rule.py arm-summary` of kit `AB`, run on Fir into the same folder.

## 2. The four headline metrics, for reference

Source: `/project/def-maxwl/mforooz/ONION/closeout_E3/arm_summary_p16_sv.txt` and `arm_summary_baselines_p16.txt` (`select_rule.py arm-summary`). Variance recovered is metric 1, the recovered share of specific variance above the noise floor. Gap closed is the correction term of composite version 3. Calibration is the gap closed in PIT deviation (probability integral transform, i.e. where the truth falls in the predicted distribution) between the marginal NB and a perfect forecast.

| Model | Variance recovered | Gap closed, crop pairs | Calibration | Composite |
|---|---|---|---|---|
| Conv autoencoder, base | 0.382 (0.028) | 0.718 (0.012) | 0.631 (0.178) | 0.528 (0.045) |
| Conv autoencoder, 3 levels | 0.378 (0.008) | 0.644 (0.019) | 0.584 (0.100) | 0.496 (0.028) |
| Conv + attention, base | 0.387 (0.012) | 0.673 (0.031) | 0.652 (0.157) | 0.525 (0.048) |
| Conv + attention, attention at 128 positions | 0.369 (0.002) | 0.643 (0.039) | 0.661 (0.142) | 0.510 (0.044) |
| U-Net, base | 0.432 (0.023) | 0.507 (0.071) | 0.610 (0.108) | 0.495 (0.010) |
| U-Net, lr 3e-3 | 0.445 (0.017) | 0.534 (0.063) | 0.619 (0.063) | 0.511 (0.028) |
| U-Net, skips carry DNA only | 0.380 (0.022) | 0.734 (0.063) | 0.640 (0.065) | 0.533 (0.029) |
| U-Net, skips carry counts only | 0.432 (0.006) | 0.518 (0.037) | 0.563 (0.145) | 0.486 (0.033) |
| Axial transformer, base | 0.378 (0.041) | 0.579 (0.009) | 0.657 (0.081) | 0.498 (0.031) |
| Axial, attention along positions only | 0.371 (0.013) | 0.668 (0.065) | 0.651 (0.153) | 0.515 (0.026) |
| Axial, attention along assays only | 0.357 (0.005) | 0.592 (0.009) | 0.589 (0.082) | 0.474 (0.023) |
| Axial, one shared attention | 0.385 (0.019) | 0.605 (0.098) | 0.609 (0.186) | 0.496 (0.062) |
| Axial, FiLM covariate injection | 0.393 (0.029) | 0.645 (0.017) | 0.655 (0.031) | 0.521 (0.026) |
| Perceiver IO, base | 0.179 (0.034) | 0.495 (0.094) | 0.487 (0.190) | 0.335 (0.088) |
| Perceiver IO, FiLM covariate injection | 0.203 (0.012) | 0.546 (0.039) | 0.626 (0.120) | 0.395 (0.042) |
| Marginal NB (baseline) | 0.000 | 0.207 | 0.000 | 0.052 |
| Position average (baseline) | 0.119 | 0.383 | 0.168 | 0.197 |
| Depth-only scaling (baseline) | undefined | 0.000 | undefined | undefined |
| Half-read copy (baseline) | undefined | 0.077 | undefined | undefined |

## 3. Metric 2: r_spec, the specific correlation

r_spec is the correlation between the truth and the prediction after each has the part removed that a smooth monotone function of the training average explains, i.e. it measures how well the model predicts what is specific to this cell (`metrics.r_spec`). Reference points: maximum 1; 0 for any prediction that is a function of the training average, so the position average and the marginal NB score 0 by construction (the no-skill baseline); minimum -1. Space: log1p counts (`r_spec.counts_log1p`).

| Model | r_spec, macro over 9 assays |
|---|---|
| Conv autoencoder, base | 0.285 (0.007) |
| Conv autoencoder, 3 levels | 0.291 (0.004) |
| Conv + attention, base | 0.296 (0.003) |
| Conv + attention, attention at 128 positions | 0.290 (0.003) |
| U-Net, base | 0.286 (0.003) |
| U-Net, lr 3e-3 | 0.300 (0.008) |
| U-Net, skips carry DNA only | 0.275 (0.014) |
| U-Net, skips carry counts only | 0.285 (0.007) |
| Axial transformer, base | 0.282 (0.010) |
| Axial, attention along positions only | 0.284 (0.002) |
| Axial, attention along assays only | 0.262 (0.002) |
| Axial, one shared attention | 0.285 (0.006) |
| Axial, FiLM covariate injection | 0.288 (0.006) |
| Perceiver IO, base | 0.145 (0.017) |
| Perceiver IO, FiLM covariate injection | 0.161 (0.020) |
| Marginal NB (baseline) | -0.000 |
| Position average (baseline) | 0.000 |
| Depth-only scaling (baseline) | undefined |
| Half-read copy (baseline) | undefined |

Per assay, base recipes (H3K9ac: count space from a replicate view):

| Model | H3K27ac | H3K27me3 | H3K36me3 | H3K4me1 | H3K4me2 | H3K4me3 | H3K79me2 | H3K9ac | H3K9me3 |
|---|---|---|---|---|---|---|---|---|---|
| Conv autoencoder, base | 0.25 (0.02) | 0.13 (0.02) | 0.22 (0.02) | 0.39 (0.01) | 0.49 (0.01) | 0.11 (0.01) | 0.40 (0.01) | 0.42 (0.00) | 0.15 (0.03) |
| Conv + attention, base | 0.26 (0.01) | 0.16 (0.01) | 0.23 (0.00) | 0.41 (0.01) | 0.50 (0.01) | 0.09 (0.01) | 0.41 (0.01) | 0.43 (0.01) | 0.17 (0.02) |
| U-Net, base | 0.29 (0.02) | 0.11 (0.02) | 0.24 (0.00) | 0.36 (0.01) | 0.49 (0.02) | 0.13 (0.01) | 0.39 (0.03) | 0.43 (0.01) | 0.14 (0.02) |
| Axial transformer, base | 0.27 (0.02) | 0.14 (0.04) | 0.23 (0.02) | 0.37 (0.02) | 0.47 (0.02) | 0.10 (0.04) | 0.39 (0.01) | 0.41 (0.04) | 0.17 (0.01) |
| Perceiver IO, base | 0.16 (0.01) | 0.03 (0.04) | 0.14 (0.02) | 0.13 (0.05) | 0.23 (0.01) | 0.05 (0.03) | 0.23 (0.04) | 0.24 (0.01) | 0.09 (0.01) |

How to read it. Every model except the Perceiver IO sits between 0.26 and 0.30, so r_spec separates the Perceiver IO and little else. The order disagrees with variance recovered at the top: conv + attention base has a higher r_spec than U-Net base (0.296 against 0.286, a difference of 0.010 with seed ranges of 0.003), while U-Net base has the higher variance recovered (0.432 against 0.387). The signal-space versions (`r_spec.signal_arcsinh`, `signal_arcsinh_plug_in`) are undefined for every model and baseline, because the scorer cannot turn a predicted count into signal: "covariates lack ['frag_len']".

## 4. Metric 3: count CRPS skill against the position average

This metric is the CRPS (continuous ranked probability score, i.e. the mean distance between the predicted distribution and the observed count; lower is better) of the position average minus the CRPS of the model, per assay, then the macro (`count_crps_skill_vs_position_average`). Reference points: 0 for the position average by construction (the no-skill baseline); positive means the model is better; the maximum is the CRPS of the position average minus the lowest attainable CRPS, which the scorer does not compute for imputation items. The CRPS of the model is split as CRPS = oracle_scaled + scale_error (`count_crps_oracle_scaled`, `count_crps_scale_error`), as the quoting rule requires.

| Model | Skill, macro | CRPS | CRPS oracle_scaled | CRPS scale_error |
|---|---|---|---|---|
| Conv autoencoder, base | 0.095 (0.023) | 0.700 (0.023) | 0.690 (0.020) | 0.009 (0.007) |
| Conv autoencoder, 3 levels | 0.080 (0.027) | 0.714 (0.027) | 0.699 (0.020) | 0.015 (0.007) |
| Conv + attention, base | 0.091 (0.011) | 0.703 (0.011) | 0.687 (0.006) | 0.016 (0.008) |
| Conv + attention, attention at 128 positions | 0.086 (0.015) | 0.709 (0.015) | 0.697 (0.012) | 0.012 (0.003) |
| U-Net, base | 0.098 (0.007) | 0.696 (0.007) | 0.686 (0.009) | 0.011 (0.009) |
| U-Net, lr 3e-3 | 0.103 (0.002) | 0.692 (0.002) | 0.684 (0.002) | 0.008 (0.005) |
| U-Net, skips carry DNA only | 0.076 (0.013) | 0.719 (0.013) | 0.703 (0.010) | 0.016 (0.018) |
| U-Net, skips carry counts only | 0.095 (0.013) | 0.700 (0.013) | 0.688 (0.004) | 0.012 (0.009) |
| Axial transformer, base | 0.087 (0.014) | 0.708 (0.014) | 0.698 (0.008) | 0.010 (0.006) |
| Axial, attention along positions only | 0.093 (0.008) | 0.702 (0.008) | 0.693 (0.003) | 0.008 (0.005) |
| Axial, attention along assays only | 0.068 (0.008) | 0.726 (0.008) | 0.713 (0.003) | 0.014 (0.007) |
| Axial, one shared attention | 0.086 (0.024) | 0.709 (0.024) | 0.695 (0.011) | 0.014 (0.013) |
| Axial, FiLM covariate injection | 0.090 (0.033) | 0.705 (0.033) | 0.696 (0.026) | 0.009 (0.006) |
| Perceiver IO, base | 0.006 (0.015) | 0.789 (0.015) | 0.779 (0.005) | 0.010 (0.009) |
| Perceiver IO, FiLM covariate injection | 0.018 (0.016) | 0.777 (0.016) | 0.769 (0.006) | 0.008 (0.011) |
| Marginal NB (baseline) | -0.216 | 1.011 | 0.845 | 0.166 |
| Position average (baseline) | 0.000 | 0.795 | 0.775 | 0.019 |
| Depth-only scaling (baseline) | undefined | undefined | undefined | undefined |
| Half-read copy (baseline) | undefined | undefined | undefined | undefined |

Per assay, base recipes:

| Model | H3K27ac | H3K27me3 | H3K36me3 | H3K4me1 | H3K4me2 | H3K4me3 | H3K79me2 | H3K9ac | H3K9me3 |
|---|---|---|---|---|---|---|---|---|---|
| Conv autoencoder, base | 0.01 (0.14) | -0.01 (0.01) | -0.13 (0.03) | 0.14 (0.01) | 0.31 (0.06) | 0.06 (0.01) | 0.27 (0.01) | 0.10 (0.01) | 0.10 (0.01) |
| Conv + attention, base | 0.06 (0.02) | -0.00 (0.01) | -0.17 (0.07) | 0.15 (0.01) | 0.34 (0.01) | -0.02 (0.08) | 0.27 (0.03) | 0.09 (0.02) | 0.11 (0.02) |
| U-Net, base | 0.08 (0.00) | -0.01 (0.01) | -0.10 (0.01) | 0.13 (0.01) | 0.30 (0.06) | 0.05 (0.03) | 0.23 (0.09) | 0.09 (0.01) | 0.10 (0.02) |
| Axial transformer, base | 0.03 (0.08) | -0.00 (0.01) | -0.14 (0.01) | 0.12 (0.01) | 0.28 (0.02) | 0.05 (0.07) | 0.23 (0.02) | 0.09 (0.07) | 0.12 (0.00) |
| Perceiver IO, base | -0.01 (0.01) | -0.04 (0.01) | -0.14 (0.02) | -0.07 (0.06) | 0.09 (0.02) | -0.05 (0.03) | 0.17 (0.04) | 0.01 (0.01) | 0.09 (0.01) |
| Marginal NB (baseline) | -0.07 | -0.10 | -0.23 | -0.27 | -0.07 | -0.38 | -0.44 | -0.10 | -0.29 |

How to read it. The models beat the position average by 0.07 to 0.10 counts of CRPS per bin; almost all of the CRPS is the oracle_scaled part, so the models lose little to a wrong total scale (scale_error 0.008 to 0.016, against 0.166 for the marginal NB). The Perceiver IO does not beat the position average beyond its seed range (0.006 with range 0.015). This metric disagrees with variance recovered per assay: on H3K36me3 every base recipe is worse than the position average (-0.10 to -0.17), while variance recovered for H3K36me3 is positive (0.353 for the conv autoencoder base, `summary.tsv`); on H3K4me3 the conv autoencoder has a positive skill (0.06) and a negative variance recovered (-0.024).

## 5. Metric 5: depth-law error on never-trained pairs

This metric is the median, over correction pairs never seen in training, of the absolute difference between the log2 of the predicted total scale ratio and the log2 of the true depth ratio (`depth_law_abs_log2_error_never_trained`, 102 pairs over 11 assays). Reference points: 0 is perfect; depth-only scaling reaches 0.001 because it scales by the true depth ratio; the marginal NB (0.963) and the position average (0.316) are the no-skill baselines; lower is better.

| Model | Depth-law error, macro over 11 assays |
|---|---|
| Conv autoencoder, base | 0.177 (0.007) |
| Conv autoencoder, 3 levels | 0.186 (0.027) |
| Conv + attention, base | 0.162 (0.068) |
| Conv + attention, attention at 128 positions | 0.163 (0.043) |
| U-Net, base | 0.127 (0.027) |
| U-Net, lr 3e-3 | 0.106 (0.019) |
| U-Net, skips carry DNA only | 0.170 (0.027) |
| U-Net, skips carry counts only | 0.111 (0.011) |
| Axial transformer, base | 0.128 (0.029) |
| Axial, attention along positions only | 0.125 (0.031) |
| Axial, attention along assays only | 0.164 (0.026) |
| Axial, one shared attention | 0.147 (0.020) |
| Axial, FiLM covariate injection | 0.122 (0.037) |
| Perceiver IO, base | 0.196 (0.058) |
| Perceiver IO, FiLM covariate injection | 0.131 (0.021) |
| Marginal NB (baseline) | 0.963 |
| Position average (baseline) | 0.316 |
| Depth-only scaling (baseline) | 0.001 |
| Half-read copy (baseline) | 0.171 |

Per assay, base recipes:

| Model | H2AFZ | H3K27ac | H3K27me3 | H3K36me3 | H3K4me1 | H3K4me2 | H3K4me3 | H3K79me2 | H3K9ac | H3K9me3 | H4K20me1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Conv autoencoder, base | 0.22 (0.05) | 0.20 (0.08) | 0.10 (0.02) | 0.22 (0.18) | 0.10 (0.11) | 0.25 (0.11) | 0.15 (0.05) | 0.13 (0.12) | 0.26 (0.01) | 0.17 (0.08) | 0.15 (0.11) |
| Conv + attention, base | 0.16 (0.18) | 0.14 (0.10) | 0.17 (0.12) | 0.14 (0.05) | 0.19 (0.07) | 0.14 (0.10) | 0.14 (0.05) | 0.14 (0.02) | 0.15 (0.10) | 0.18 (0.05) | 0.22 (0.13) |
| U-Net, base | 0.12 (0.06) | 0.18 (0.06) | 0.13 (0.03) | 0.12 (0.03) | 0.08 (0.04) | 0.14 (0.02) | 0.13 (0.06) | 0.08 (0.05) | 0.16 (0.04) | 0.12 (0.06) | 0.13 (0.04) |
| Axial transformer, base | 0.14 (0.09) | 0.11 (0.04) | 0.12 (0.01) | 0.10 (0.05) | 0.11 (0.04) | 0.11 (0.05) | 0.16 (0.03) | 0.16 (0.02) | 0.22 (0.07) | 0.12 (0.05) | 0.07 (0.11) |
| Perceiver IO, base | 0.09 (0.07) | 0.31 (0.31) | 0.22 (0.21) | 0.15 (0.08) | 0.17 (0.15) | 0.18 (0.06) | 0.16 (0.12) | 0.30 (0.23) | 0.27 (0.04) | 0.13 (0.10) | 0.18 (0.25) |
| Position average (baseline) | 0.30 | 0.27 | 0.19 | 0.26 | 0.42 | 0.33 | 0.31 | 0.36 | 0.19 | 0.35 | 0.50 |
| Half-read copy (baseline) | 0.18 | 0.14 | 0.12 | 0.12 | 0.13 | 0.17 | 0.18 | 0.14 | 0.35 | 0.23 | 0.12 |

How to read it. An error of 0.177 means the predicted depth ratio is off by a factor of 2^0.177 = 1.13 at the median pair. This metric disagrees with the correction headline: U-Net, lr 3e-3 has a lower depth-law error than the conv autoencoder base (0.106 against 0.177; the difference 0.071 is larger than twice the larger seed range, 2 x 0.019 = 0.038), but a lower gap closed (0.534 against 0.718). The model with the best gap closed is therefore not the one that follows the depth law best; the per-assay seed ranges (up to 0.31) are often as large as the values.

## 6. Metric 7: predicted variance over replicate variance

This metric is the mean predicted NB variance divided by the variance between the two read halves (pseudoreplicates) of the target, i.e. the sampling noise; the scorer averages the ratio per target, then per assay, then over assays (`predicted_over_replicate_variance`). Reference points: a calibrated model scores 1 or more at this rung, because the true spread includes the sampling noise and more (lower-bound check, PI ruling 31); a ratio of 1 means the model predicts only sampling noise; the upper bound (ratio at most 1 against a second experiment of the same cell, the other-experiment rung) is not computed, so this is a lower bound only. No baseline is a no-skill point for this metric; the marginal NB and the position average are shown for comparison.

| Model | Ratio, pseudoreplicate rung, macro | Ratio, isogenic rung, macro | Ratio, other cell type, macro | Pass rate (both bounds) |
|---|---|---|---|---|
| Conv autoencoder, base | 2.455 (1.536) | undefined | undefined | undefined |
| Conv autoencoder, 3 levels | 3.942 (4.853) | undefined | undefined | undefined |
| Conv + attention, base | 2.356 (0.570) | undefined | undefined | undefined |
| Conv + attention, attention at 128 positions | 1.833 (0.859) | undefined | undefined | undefined |
| U-Net, base | 2.495 (0.521) | undefined | undefined | undefined |
| U-Net, lr 3e-3 | 2.473 (0.967) | undefined | undefined | undefined |
| U-Net, skips carry DNA only | 3.299 (0.927) | undefined | undefined | undefined |
| U-Net, skips carry counts only | 2.346 (0.610) | undefined | undefined | undefined |
| Axial transformer, base | 1.861 (0.378) | undefined | undefined | undefined |
| Axial, attention along positions only | 1.705 (0.340) | undefined | undefined | undefined |
| Axial, attention along assays only | 2.564 (0.391) | undefined | undefined | undefined |
| Axial, one shared attention | 1.820 (0.083) | undefined | undefined | undefined |
| Axial, FiLM covariate injection | 1.922 (0.210) | undefined | undefined | undefined |
| Perceiver IO, base | 1.886 (0.501) | undefined | undefined | undefined |
| Perceiver IO, FiLM covariate injection | 2.384 (0.434) | undefined | undefined | undefined |
| Marginal NB (baseline) | 2.543 | undefined | undefined | undefined |
| Position average (baseline) | 4.777 | undefined | undefined | undefined |
| Depth-only scaling (baseline) | undefined | absent | absent | undefined |
| Half-read copy (baseline) | undefined | absent | absent | undefined |

Per assay, pseudoreplicate rung, base recipes:

| Model | H3K27ac | H3K27me3 | H3K36me3 | H3K4me1 | H3K4me2 | H3K4me3 | H3K79me2 | H3K9ac | H3K9me3 |
|---|---|---|---|---|---|---|---|---|---|
| Conv autoencoder, base | 7.90 (13.32) | 1.25 (0.11) | 1.39 (0.32) | 1.29 (0.17) | 2.50 (1.08) | 2.63 (0.19) | 1.98 (0.62) | 2.02 (0.12) | 1.14 (0.21) |
| Conv + attention, base | 5.67 (3.51) | 1.03 (0.09) | 1.70 (1.02) | 1.18 (0.14) | 2.03 (0.45) | 2.03 (0.95) | 2.42 (0.93) | 4.02 (4.92) | 1.11 (0.24) |
| U-Net, base | 4.12 (1.14) | 1.30 (0.15) | 1.47 (0.17) | 1.81 (0.25) | 4.79 (2.90) | 2.80 (1.36) | 2.85 (0.87) | 2.09 (0.64) | 1.22 (0.09) |
| Axial transformer, base | 4.42 (1.96) | 1.26 (0.10) | 1.41 (0.12) | 1.38 (0.25) | 2.02 (1.41) | 2.01 (0.93) | 1.43 (0.42) | 1.64 (0.15) | 1.19 (0.07) |
| Perceiver IO, base | 2.65 (1.71) | 1.30 (0.20) | 1.38 (0.14) | 1.52 (0.24) | 2.96 (0.90) | 2.07 (0.51) | 2.45 (2.43) | 1.48 (0.69) | 1.16 (0.14) |
| Marginal NB (baseline) | 2.57 | 3.21 | 2.45 | 3.99 | 1.81 | 1.44 | 1.40 | 2.69 | 3.32 |
| Position average (baseline) | 3.19 | 1.43 | 1.56 | 2.89 | 16.97 | 7.89 | 5.29 | 2.57 | 1.21 |

How to read it. Every macro and every per-assay mean in these tables is above 1, so the lower-bound check excludes no model; the prediction from the earlier baseline work (models also above 1 at this rung) holds. This metric disagrees with the calibration headline: the Perceiver IO base has the lowest calibration (0.487) but a ratio nearer 1 (1.886) than the conv autoencoder base (2.455) and the U-Net base (2.495). The seed ranges are as large as the differences, so the ratio cannot rank models at this rung.

## 7. Error budget (Max's development eval)

The error budget splits the mean squared error of a calibrated prediction, in units of the truth's variance, into level (offset, scale and nonlinear error), shared (error in what the cells of an assay have in common) and specific (error in what is specific to one cell); it also splits the truth's variance into explained, unexplained and noise (the sampling noise from the read halves), and "added" is the error the prediction adds beyond what it fails to explain (`error_budget`, `tools/onion_eval/error_budget.py`). Reference points: the error terms (level, shared, specific, added) are 0 for a perfect prediction and lower is better; explained + unexplained + noise = 1, so the highest attainable explained share is 1 - noise; the marginal NB explains 0 (the no-skill baseline).

| Model | level | shared | specific | explained | unexplained | noise | added |
|---|---|---|---|---|---|---|---|
| Conv autoencoder, base | 0.100 (0.015) | 0.074 (0.004) | 0.689 (0.005) | 0.257 (0.005) | 0.176 (0.005) | 0.567 (0.000) | 0.020 (0.004) |
| Conv autoencoder, 3 levels | 0.121 (0.035) | 0.067 (0.001) | 0.689 (0.001) | 0.271 (0.001) | 0.163 (0.001) | 0.567 (0.000) | 0.027 (0.002) |
| Conv + attention, base | 0.120 (0.017) | 0.068 (0.001) | 0.688 (0.002) | 0.269 (0.002) | 0.165 (0.002) | 0.567 (0.000) | 0.025 (0.002) |
| Conv + attention, attention at 128 positions | 0.112 (0.019) | 0.068 (0.003) | 0.691 (0.001) | 0.269 (0.001) | 0.164 (0.001) | 0.567 (0.000) | 0.028 (0.004) |
| U-Net, base | 0.122 (0.024) | 0.074 (0.000) | 0.680 (0.005) | 0.255 (0.004) | 0.178 (0.004) | 0.567 (0.000) | 0.010 (0.001) |
| U-Net, lr 3e-3 | 0.121 (0.013) | 0.071 (0.002) | 0.678 (0.003) | 0.260 (0.005) | 0.173 (0.005) | 0.567 (0.000) | 0.009 (0.001) |
| U-Net, skips carry DNA only | 0.132 (0.061) | 0.077 (0.005) | 0.693 (0.005) | 0.243 (0.005) | 0.190 (0.005) | 0.567 (0.000) | 0.013 (0.003) |
| U-Net, skips carry counts only | 0.139 (0.034) | 0.078 (0.001) | 0.680 (0.002) | 0.253 (0.001) | 0.181 (0.001) | 0.567 (0.000) | 0.011 (0.002) |
| Axial transformer, base | 0.100 (0.033) | 0.076 (0.002) | 0.693 (0.009) | 0.250 (0.006) | 0.183 (0.006) | 0.567 (0.000) | 0.019 (0.005) |
| Axial, attention along positions only | 0.090 (0.033) | 0.077 (0.003) | 0.693 (0.000) | 0.254 (0.004) | 0.180 (0.004) | 0.567 (0.000) | 0.023 (0.003) |
| Axial, attention along assays only | 0.126 (0.040) | 0.083 (0.004) | 0.696 (0.001) | 0.242 (0.003) | 0.191 (0.003) | 0.567 (0.000) | 0.020 (0.003) |
| Axial, one shared attention | 0.111 (0.043) | 0.077 (0.001) | 0.691 (0.003) | 0.253 (0.003) | 0.181 (0.003) | 0.567 (0.000) | 0.021 (0.006) |
| Axial, FiLM covariate injection | 0.101 (0.042) | 0.077 (0.004) | 0.688 (0.004) | 0.258 (0.004) | 0.175 (0.004) | 0.567 (0.000) | 0.023 (0.003) |
| Perceiver IO, base | 0.104 (0.049) | 0.157 (0.002) | 0.746 (0.008) | 0.102 (0.008) | 0.331 (0.008) | 0.567 (0.000) | 0.005 (0.003) |
| Perceiver IO, FiLM covariate injection | 0.148 (0.104) | 0.154 (0.002) | 0.740 (0.003) | 0.110 (0.004) | 0.324 (0.004) | 0.567 (0.000) | 0.004 (0.001) |
| Marginal NB (baseline) | 0.718 | 0.210 | 0.790 | 0.000 | 0.433 | 0.567 | 0.000 |
| Position average (baseline) | 0.196 | 0.022 | 0.764 | 0.221 | 0.212 | 0.567 | 0.007 |
| Depth-only scaling (baseline) | undefined | undefined | undefined | undefined | undefined | undefined | undefined |
| Half-read copy (baseline) | undefined | undefined | undefined | undefined | undefined | undefined | undefined |

Explained share per assay, base recipes (the assays with at least 2 targets; the others are undefined):

| Model | H2AFZ | H3K27ac | H3K27me3 | H3K36me3 | H3K4me1 | H3K4me2 | H3K4me3 | H3K79me2 | H3K9ac | H3K9me3 | H4K20me1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Conv autoencoder, base | 0.43 (0.01) | 0.24 (0.01) | 0.15 (0.04) | 0.21 (0.02) | 0.33 (0.01) | 0.42 (0.01) | 0.18 (0.01) | 0.32 (0.02) | 0.35 (0.01) | 0.10 (0.01) | 0.27 (0.01) |
| Conv + attention, base | 0.45 (0.00) | 0.24 (0.01) | 0.20 (0.01) | 0.23 (0.02) | 0.34 (0.00) | 0.42 (0.01) | 0.17 (0.00) | 0.34 (0.02) | 0.35 (0.01) | 0.11 (0.01) | 0.26 (0.00) |
| U-Net, base | 0.43 (0.02) | 0.26 (0.00) | 0.13 (0.01) | 0.21 (0.00) | 0.33 (0.00) | 0.42 (0.01) | 0.18 (0.01) | 0.31 (0.02) | 0.35 (0.00) | 0.10 (0.00) | 0.26 (0.01) |
| Axial transformer, base | 0.40 (0.06) | 0.25 (0.01) | 0.14 (0.01) | 0.20 (0.02) | 0.32 (0.00) | 0.40 (0.02) | 0.17 (0.02) | 0.34 (0.03) | 0.34 (0.04) | 0.10 (0.00) | 0.27 (0.01) |
| Perceiver IO, base | 0.11 (0.00) | 0.11 (0.01) | 0.06 (0.02) | 0.12 (0.01) | 0.10 (0.02) | 0.14 (0.01) | 0.04 (0.00) | 0.17 (0.02) | 0.13 (0.01) | 0.04 (0.00) | 0.10 (0.02) |
| Marginal NB (baseline) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Position average (baseline) | 0.21 | 0.18 | 0.21 | 0.34 | 0.20 | 0.21 | 0.21 | 0.27 | 0.22 | 0.14 | 0.16 |

How to read it. The sampling noise is 0.567 of the truth's variance at 25 bp, so at most 1 - 0.567 = 0.433 can be explained; the conv autoencoder base explains 0.257 / 0.433 = 0.59 of that, the position average 0.221 / 0.433 = 0.51 and the Perceiver IO base 0.102 / 0.433 = 0.24. This metric is much less favorable to the models than variance recovered (0.382 against 0.119 for the position average), because "explained" here includes the variance shared between cells, which the position average already captures. The noise term is a property of the data, so it is the same for every model (range 0.000).

## 8. Explained fraction per variance layer

For a pair of tracks that differ in one layer (here: two cell types), the explained fraction is the part of the true difference between the pair (the layer term) that the difference between the two predictions captures, averaged over cells (`explained_fraction_per_layer`, `layers.explained_table`; reported only, never a selection input, PI ruling 32). Reference points: 0 means the prediction does not change between the two members; 1 means it captures the whole layer term; the scorer does not clip it, so with few pairs it can be below 0 or above 1; both baselines give the same prediction to both members, so they score 0 (the no-skill baseline).

Only one layer has rows: the cell-type layer (rung `other_cell_type`; for H3K4me1 the layer is cell type + experiment, for the others donor + experiment + cell type). The scorer lists the isogenic and anisogenic layers under `zero_by_shared_inputs`: both members get the same inputs, so the model cannot separate them and the fraction is 0 by construction. The sampling (pseudoreplicate) and other-experiment layers have no pair in this evaluation set (`scorable.layers`). Eleven assays have no row ("no other_experiment or other_cell_type pair with a prediction for both members").

| Model | H2AFZ (1 pair, 1 cell) | H3K27ac (3, 1) | H3K27me3 (3, 1) | H3K36me3 (3, 1) | H3K4me1 (3, 1) | H3K4me2 (3, 2) | H3K79me2 (3, 2) | H3K9ac (1, 1) | H3K9me3 (3, 1) | H4K20me1 (3, 2) |
|---|---|---|---|---|---|---|---|---|---|---|
| Conv autoencoder, base | 5.78 (0.77) | 0.41 (0.12) | 0.11 (0.01) | 0.41 (0.09) | 0.18 (0.04) | 1.82 (0.49) | 0.28 (0.11) | 0.38 (0.13) | 0.14 (0.06) | 0.29 (0.10) |
| Conv autoencoder, 3 levels | 7.43 (0.67) | 0.44 (0.02) | 0.08 (0.08) | 0.39 (0.06) | 0.22 (0.04) | 2.30 (0.70) | 0.31 (0.05) | 0.58 (0.14) | 0.16 (0.02) | 0.30 (0.02) |
| Conv + attention, base | 8.29 (4.13) | 0.41 (0.07) | 0.12 (0.06) | 0.48 (0.18) | 0.20 (0.05) | 2.11 (0.32) | 0.30 (0.09) | 0.52 (0.16) | 0.16 (0.06) | 0.29 (0.09) |
| Conv + attention, attention at 128 positions | 7.36 (3.95) | 0.41 (0.09) | 0.00 (0.04) | 0.43 (0.08) | 0.18 (0.04) | 2.11 (0.16) | 0.28 (0.07) | 0.30 (0.19) | 0.17 (0.04) | 0.30 (0.10) |
| U-Net, base | 4.81 (0.81) | 0.37 (0.03) | 0.05 (0.06) | 0.30 (0.03) | 0.18 (0.03) | 1.88 (0.35) | 0.25 (0.02) | 0.38 (0.03) | 0.07 (0.01) | 0.23 (0.02) |
| U-Net, lr 3e-3 | 4.91 (0.91) | 0.40 (0.09) | 0.13 (0.05) | 0.36 (0.03) | 0.17 (0.03) | 1.71 (0.47) | 0.28 (0.05) | 0.38 (0.12) | 0.07 (0.01) | 0.26 (0.05) |
| U-Net, skips carry DNA only | 5.77 (0.70) | 0.46 (0.07) | 0.10 (0.03) | 0.31 (0.02) | 0.19 (0.06) | 2.29 (0.28) | 0.31 (0.09) | 0.48 (0.03) | 0.05 (0.02) | 0.27 (0.07) |
| U-Net, skips carry counts only | 5.32 (0.46) | 0.35 (0.07) | 0.08 (0.04) | 0.29 (0.05) | 0.15 (0.00) | 1.68 (0.34) | 0.25 (0.04) | 0.38 (0.10) | 0.07 (0.01) | 0.22 (0.04) |
| Axial transformer, base | 5.10 (1.52) | 0.40 (0.06) | 0.03 (0.04) | 0.37 (0.03) | 0.20 (0.01) | 1.95 (0.06) | 0.25 (0.06) | 0.39 (0.21) | 0.10 (0.01) | 0.38 (0.04) |
| Axial, attention along positions only | 5.43 (1.53) | 0.39 (0.06) | 0.01 (0.03) | 0.37 (0.02) | 0.18 (0.01) | 2.05 (0.11) | 0.22 (0.01) | 0.41 (0.03) | 0.09 (0.01) | 0.33 (0.10) |
| Axial, attention along assays only | 7.31 (0.57) | 0.37 (0.01) | 0.04 (0.03) | 0.36 (0.04) | 0.17 (0.01) | 1.66 (0.11) | 0.28 (0.03) | 0.37 (0.01) | 0.11 (0.00) | 0.27 (0.02) |
| Axial, one shared attention | 5.15 (0.21) | 0.39 (0.09) | 0.07 (0.08) | 0.35 (0.02) | 0.18 (0.05) | 2.01 (0.31) | 0.24 (0.01) | 0.35 (0.06) | 0.10 (0.01) | 0.34 (0.05) |
| Axial, FiLM covariate injection | 6.46 (0.34) | 0.42 (0.01) | 0.09 (0.09) | 0.40 (0.03) | 0.20 (0.02) | 2.05 (0.49) | 0.26 (0.02) | 0.42 (0.10) | 0.11 (0.03) | 0.28 (0.09) |
| Perceiver IO, base | 0.77 (0.90) | 0.10 (0.06) | -0.01 (0.05) | 0.08 (0.05) | 0.03 (0.01) | 0.49 (0.17) | 0.11 (0.05) | 0.05 (0.05) | 0.03 (0.01) | 0.07 (0.04) |
| Perceiver IO, FiLM covariate injection | 1.57 (0.17) | 0.14 (0.04) | 0.03 (0.08) | 0.12 (0.01) | 0.06 (0.00) | 0.62 (0.12) | 0.15 (0.06) | 0.09 (0.04) | 0.03 (0.03) | 0.10 (0.01) |
| Marginal NB (baseline) | -0.00 | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 |
| Position average (baseline) | -0.00 | -0.00 | -0.00 | 0.00 | -0.00 | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 |

How to read it. On the assays with 3 pairs and values below 1, the conv, U-Net and axial models capture 0.00 to 0.48 of the cell-type difference, and the Perceiver IO models capture at most 0.15. The values for H2AFZ (4.81 to 8.29) and H3K4me2 (1.66 to 2.30) are above 1, i.e. the predictions of the two cells differ more than the truths do; H2AFZ rests on 1 pair in 1 cell, so it cannot be read. The earlier prediction that the explained fraction is informative only on the cell-type layer holds by construction: the scorer computes no other layer.

## 9. Accessibility-only regime (second table)

In this regime only the accessibility track (DNase-seq or ATAC-seq) of the held-out cell is given and the other assays are imputed (D11b.2, key `accessibility_only`). It has 20 items over 8 scorable assays (no H3K79me2 item), against 43 items in the headline regime. The metrics and reference points are those of sections 3, 4 and 6 and the level term of section 7.

| Model | r_spec | CRPS skill vs position average | CRPS (oracle_scaled + scale_error) | PIT deviation, pooled | PIT deviation, worst stratum | Ratio, pseudoreplicate | Ratio, other cell type | Error budget, level |
|---|---|---|---|---|---|---|---|---|
| Conv autoencoder, base | 0.180 (0.033) | -0.013 (0.019) | 0.718 (0.019) = 0.706 + 0.012 | 0.040 (0.007) | 0.111 (0.011) | 1.590 (0.474) | 0.636 (0.132) | 0.115 (0.047) |
| Conv autoencoder, 3 levels | 0.209 (0.011) | -0.008 (0.027) | 0.713 (0.027) = 0.703 + 0.009 | 0.036 (0.017) | 0.111 (0.009) | 1.401 (0.367) | 0.572 (0.107) | 0.133 (0.061) |
| Conv + attention, base | 0.190 (0.006) | -0.019 (0.013) | 0.724 (0.013) = 0.714 + 0.010 | 0.039 (0.014) | 0.109 (0.014) | 1.398 (0.084) | 0.587 (0.053) | 0.150 (0.061) |
| Conv + attention, attention at 128 positions | 0.201 (0.033) | -0.024 (0.036) | 0.729 (0.036) = 0.718 + 0.011 | 0.046 (0.002) | 0.108 (0.010) | 1.336 (0.198) | 0.585 (0.076) | 0.184 (0.024) |
| U-Net, base | 0.232 (0.060) | 0.010 (0.052) | 0.695 (0.052) = 0.684 + 0.011 | 0.038 (0.013) | 0.112 (0.021) | 2.337 (0.619) | 0.810 (0.118) | 0.129 (0.018) |
| U-Net, lr 3e-3 | 0.262 (0.045) | 0.033 (0.037) | 0.672 (0.037) = 0.662 + 0.010 | 0.032 (0.008) | 0.102 (0.010) | 1.933 (0.540) | 0.743 (0.161) | 0.096 (0.014) |
| U-Net, skips carry DNA only | 0.242 (0.020) | 0.010 (0.045) | 0.694 (0.045) = 0.688 + 0.006 | 0.034 (0.019) | 0.100 (0.015) | 2.569 (0.252) | 0.880 (0.095) | 0.099 (0.025) |
| U-Net, skips carry counts only | 0.247 (0.032) | 0.018 (0.039) | 0.686 (0.039) = 0.677 + 0.009 | 0.031 (0.006) | 0.118 (0.020) | 2.111 (1.034) | 0.770 (0.261) | 0.107 (0.074) |
| Axial transformer, base | 0.223 (0.006) | 0.005 (0.023) | 0.699 (0.023) = 0.694 + 0.005 | 0.029 (0.004) | 0.100 (0.007) | 2.833 (1.954) | 0.906 (0.389) | 0.107 (0.047) |
| Axial, attention along positions only | 0.239 (0.010) | 0.004 (0.040) | 0.701 (0.040) = 0.692 + 0.009 | 0.032 (0.017) | 0.123 (0.025) | 2.733 (2.402) | 0.865 (0.440) | 0.103 (0.022) |
| Axial, attention along assays only | 0.166 (0.009) | -0.051 (0.013) | 0.756 (0.013) = 0.749 + 0.007 | 0.031 (0.006) | 0.094 (0.012) | 3.336 (1.412) | 1.026 (0.287) | 0.158 (0.063) |
| Axial, one shared attention | 0.237 (0.030) | 0.013 (0.016) | 0.692 (0.016) = 0.685 + 0.006 | 0.030 (0.006) | 0.104 (0.005) | 2.931 (0.823) | 0.866 (0.152) | 0.100 (0.049) |
| Axial, FiLM covariate injection | 0.219 (0.023) | -0.016 (0.033) | 0.721 (0.033) = 0.714 + 0.007 | 0.035 (0.018) | 0.116 (0.015) | 3.204 (1.225) | 0.927 (0.192) | 0.147 (0.061) |
| Perceiver IO, base | 0.112 (0.010) | -0.073 (0.006) | 0.777 (0.006) = 0.766 + 0.011 | 0.036 (0.006) | 0.183 (0.011) | 2.132 (0.494) | 0.783 (0.179) | 0.105 (0.040) |
| Perceiver IO, FiLM covariate injection | 0.117 (0.014) | -0.066 (0.010) | 0.770 (0.010) = 0.764 + 0.007 | 0.034 (0.014) | 0.180 (0.011) | 2.470 (0.301) | 0.874 (0.097) | 0.121 (0.051) |
| Marginal NB (baseline) | 0.000 | -0.225 | 0.930 = 0.801 + 0.129 | 0.118 | 0.234 | 2.769 | 1.570 | 0.749 |
| Position average (baseline) | 0.000 | 0.000 | 0.705 = 0.691 + 0.014 | 0.056 | 0.093 | 6.544 | 1.713 | 0.183 |

How to read it. With accessibility alone, no model beats the position average on count CRPS beyond its seed range (skills -0.051 to 0.033 for the conv, U-Net and axial models; the Perceiver IO base is worse, -0.073), although r_spec stays positive (0.11 to 0.26). The models keep a better calibration than the position average on the pooled PIT deviation (0.029 to 0.046 against 0.056). Variance recovered, the explained fraction and every error-budget term except level are undefined here, because each assay has one target in this regime, and the shared/specific split needs at least 2 targets of the assay; the depth-law error is absent, because the regime has no correction pairs.

## 10. Artifact variance: by depth, platform, antigen and scale

The scorer computes the correction gap closed per artifact knob (`arm_report.py per-knob`, from `correction_gap_closed_vs_depth_only_never_trained` and the oracle reference). The gap closed is (CRPS of depth-only scaling - CRPS of the model) / (CRPS of depth-only scaling - oracle floor), on never-trained pairs. Reference points: 0 for depth-only scaling (the no-skill baseline); 1 for a model that reaches the oracle floor. The knobs are: crop (shorter reads on one side; 14 pairs), crop + single-end (4 pairs), and MAPQ + no dedup (no mapping-quality filter and no duplicate removal on one side; 84 pairs). The last column is the pure depth-thinning pairs (516 pairs), which the scorer writes only on the superseded half-read scale (0 = depth-only scaling, 1 = half-read copy).

| Model | crop | crop + single-end | MAPQ + no dedup | macro, all pairs | macro, no duplicate-keeping side | depth thinning (half-read scale) |
|---|---|---|---|---|---|---|
| Conv autoencoder, base | 0.712 (0.014) | 0.674 (0.025) | 0.498 (0.015) | 0.568 (0.014) | 0.718 (0.012) | 2.223 (0.026) |
| Conv autoencoder, 3 levels | 0.611 (0.018) | 0.681 (0.037) | 0.483 (0.026) | 0.534 (0.026) | 0.644 (0.019) | 2.242 (0.086) |
| Conv + attention, base | 0.652 (0.032) | 0.671 (0.017) | 0.497 (0.024) | 0.566 (0.026) | 0.673 (0.031) | 2.329 (0.008) |
| Conv + attention, attention at 128 positions | 0.619 (0.045) | 0.700 (0.009) | 0.489 (0.008) | 0.545 (0.018) | 0.643 (0.039) | 2.280 (0.047) |
| U-Net, base | 0.507 (0.076) | 0.516 (0.078) | 0.272 (0.049) | 0.300 (0.058) | 0.507 (0.071) | 1.918 (0.026) |
| U-Net, lr 3e-3 | 0.536 (0.069) | 0.543 (0.033) | 0.136 (0.077) | 0.184 (0.057) | 0.534 (0.063) | 1.854 (0.061) |
| U-Net, skips carry DNA only | 0.739 (0.066) | 0.719 (0.046) | 0.549 (0.020) | 0.603 (0.026) | 0.734 (0.063) | 2.354 (0.110) |
| U-Net, skips carry counts only | 0.522 (0.035) | 0.492 (0.056) | 0.181 (0.027) | 0.224 (0.032) | 0.518 (0.037) | 1.853 (0.022) |
| Axial transformer, base | 0.577 (0.025) | 0.557 (0.125) | 0.436 (0.076) | 0.443 (0.068) | 0.578 (0.009) | 2.159 (0.102) |
| Axial, attention along positions only | 0.676 (0.063) | 0.603 (0.096) | 0.514 (0.086) | 0.551 (0.096) | 0.668 (0.065) | 2.333 (0.150) |
| Axial, attention along assays only | 0.593 (0.004) | 0.489 (0.019) | 0.467 (0.020) | 0.518 (0.019) | 0.592 (0.009) | 2.224 (0.030) |
| Axial, one shared attention | 0.605 (0.101) | 0.560 (0.172) | 0.462 (0.069) | 0.470 (0.080) | 0.605 (0.098) | 2.245 (0.092) |
| Axial, FiLM covariate injection | 0.646 (0.019) | 0.658 (0.054) | 0.513 (0.009) | 0.542 (0.011) | 0.645 (0.017) | 2.380 (0.030) |
| Perceiver IO, base | 0.472 (0.084) | 0.385 (0.074) | 0.250 (0.082) | 0.333 (0.084) | 0.495 (0.094) | 1.731 (0.075) |
| Perceiver IO, FiLM covariate injection | 0.520 (0.045) | 0.416 (0.030) | 0.309 (0.036) | 0.386 (0.032) | 0.546 (0.039) | 1.845 (0.105) |
| Marginal NB (baseline) | 0.186 | 0.177 | -0.355 | -0.265 | 0.207 | -6.353 |
| Position average (baseline) | 0.369 | 0.115 | 0.170 | 0.235 | 0.383 | 1.850 |
| Depth-only scaling (baseline) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Half-read copy (baseline) | 0.065 | 0.134 | 0.213 | 0.151 | 0.077 | 1.000 |

How to read it. The read-length knobs (crop, crop + single-end) are the nearest stand-in for a sequencing-platform difference, and the models close 0.39 to 0.74 of that gap; the U-Net variants that pass counts through the skips fall to 0.14 to 0.27 on the MAPQ + no dedup pairs, which is the copy effect that the correction-metric review found (`results/correction_metric_review/REVIEW.md`). The crop + single-end column rests on 4 pairs. What is not computed:
- Depth: the scorer has no variance split by depth alone. The depth-thinning gap closed (last column) and the depth-law error (section 5) are the only depth measures, and the first is on the superseded half-read scale.
- Platform and antigen: not computed. No pair or covariate in `eic_v1` varies the sequencing platform or the antibody with the biology fixed (audit rows A1, A2, B6); this waits for the lab and antibody covariates (item D1) and the second experiments (item D2).
- Artifact variance by scale: not computed. `scores.json` has no key for it; the error budget is stored summed over the four Haar bands only (`arm_report.py` dev-evals note), and the variance-by-scale analysis (`tools/onion_eval/scale_variance.py`, `results/t305`) is on the data, not on model errors (audit row B12).

## 11. Per-stratum values

The scorer stores one per-stratum value per assay: the PIT deviation in the worst region stratum (`pit_deviation_worst_stratum`). It stores no value for each stratum (promoters, enhancers, gene bodies, heterochromatin, mappability deciles) and no per-stratum variance recovered. A per-stratum table therefore needs a scorer change and a rescore; this item changed no scorer code.

| Model | H3K27ac | H3K27me3 | H3K36me3 | H3K4me1 | H3K4me2 | H3K4me3 | H3K79me2 | H3K9ac | H3K9me3 | Macro |
|---|---|---|---|---|---|---|---|---|---|---|
| Conv autoencoder, base | 0.08 (0.02) | 0.05 (0.00) | 0.07 (0.03) | 0.07 (0.02) | 0.05 (0.01) | 0.09 (0.03) | 0.07 (0.02) | 0.06 (0.04) | 0.13 (0.03) | 0.074 (0.006) |
| Conv + attention, base | 0.06 (0.00) | 0.04 (0.01) | 0.07 (0.03) | 0.05 (0.02) | 0.03 (0.01) | 0.07 (0.03) | 0.07 (0.04) | 0.04 (0.02) | 0.11 (0.02) | 0.062 (0.006) |
| U-Net, base | 0.07 (0.02) | 0.05 (0.02) | 0.07 (0.04) | 0.11 (0.02) | 0.04 (0.03) | 0.09 (0.05) | 0.13 (0.03) | 0.09 (0.04) | 0.12 (0.02) | 0.085 (0.005) |
| Axial transformer, base | 0.07 (0.01) | 0.07 (0.00) | 0.11 (0.00) | 0.10 (0.02) | 0.04 (0.03) | 0.11 (0.07) | 0.06 (0.04) | 0.08 (0.05) | 0.13 (0.01) | 0.086 (0.012) |
| Perceiver IO, base | 0.18 (0.01) | 0.08 (0.01) | 0.10 (0.01) | 0.11 (0.01) | 0.21 (0.03) | 0.23 (0.02) | 0.09 (0.04) | 0.20 (0.03) | 0.15 (0.03) | 0.151 (0.012) |
| Marginal NB (baseline) | 0.16 | 0.16 | 0.09 | 0.16 | 0.28 | 0.31 | 0.15 | 0.21 | 0.14 | 0.185 |
| Position average (baseline) | 0.05 | 0.01 | 0.07 | 0.06 | 0.17 | 0.12 | 0.09 | 0.03 | 0.06 | 0.074 |

How to read it. Reference points: 0 is a calibrated forecast in every stratum, lower is better, and the marginal NB (0.185) is the no-skill baseline. In the worst stratum, the conv + attention base (0.062) is the only base recipe below the position average (0.074), by 0.012, which is twice its seed range of 0.006; the conv autoencoder base equals it (0.074) and the others are above it.

## 12. Values that look wrong, and values that are undefined

Values that look wrong (flagged, not fixed):
1. Metric 7 is dominated by single targets. In `rung2_base_heldout/seed0`, one H3K27ac target (ENCSR000AQW) has a ratio of 47.07, and the H3K27ac value is 16.73; across seeds, the H3K27ac cell of the conv autoencoder base is 7.90 with a range of 13.32. A mean of per-target ratios is unstable when the replicate variance of a target is small; a median, or a ratio of means, would be more robust.
2. The explained fraction is above 1 for H2AFZ (4.81 to 8.29, 1 pair in 1 cell) and H3K4me2 (1.66 to 2.30, 3 pairs in 2 cells). The scorer allows this by design (no clipping), but these cells should not be read as "more than all of the layer".
3. The `selection` block of each `scores.json` holds the composite of version 1, with the superseded half-read correction term (for `rung2_base_heldout/seed0`: `selection.composite` = 1.938 and `selection.correction_gap_closed` = 6.411). The per-knob values under `correction_gap_closed_vs_depth_only_never_trained.per_knob` are on the same superseded scale (crop 11.116 for that file). A reader who takes the composite from `scores.json` directly gets a wrong number; the version-3 composite comes from `select_rule.py arm-summary`.
4. Every signal-space metric (`r_spec.signal_arcsinh`, `specific_variance_recovered_above_noise.signal_arcsinh` and their plug-in versions) is undefined for every model and baseline, with the reason "covariates lack ['frag_len']". The scorer computes these keys but can never fill them on this evaluation set.

Values that are undefined (by design or by missing data):
- Metric 7 pass rate and the upper bound: the other-experiment rung is not populated (waits for item D2).
- Metric 7 at the isogenic rung (both regimes) and at the other-cell-type rung (headline regime): the macro is undefined because at least one scorable assay has no defined ratio (for example H3K4me1 and H3K9ac at the other-cell-type rung in `rung2_base_heldout/seed0`); per-assay values exist for the other assays.
- Depth-only scaling and half-read copy: every imputation metric, because they predict only from a source track.
- Accessibility-only: variance recovered, the explained fraction and every error-budget term except level (one target per assay); the depth-law error is absent.

## 13. What this means

The secondary metrics confirm the main split of the headline set: the Perceiver IO trails on every imputation metric, and the conv, U-Net and axial models are close to each other. They do not rank the top models in the same order as the headline metrics, and metric 7 and the explained fraction are too noisy on the screening set to rank anything. The accessibility-only table shows that, with accessibility alone, no model beats the position average on count CRPS.
