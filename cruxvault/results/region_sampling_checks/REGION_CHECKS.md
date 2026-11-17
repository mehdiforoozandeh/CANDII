Check 1: the 16 pair regions are not poor in signal; their median signal is at the 58th percentile of all 6,724 candidate blocks (lowest at the 18th), and none of the 16 has near-zero signal.
Check 2: every ablation effect on the two correction terms that is larger than the seed range keeps its sign in at least 95% of 5,000 paired region resamples, so no claim of the ablation read depends on the choice of the 16 regions; the effects in doubt are only those that the read already called within the seed range.

# Region-sampling checks of the miniCANDI ablations

Close-out item CO-REG, 2026-11-17. The PI asked whether the ablation results would change with more regions, and whether the 16 regions are "dead" regions of the genome. This file answers with two checks. Check 1 measures the signal content of the regions. Check 2 resamples the 16 regions (a bootstrap, i.e. drawing the regions again with replacement) and recomputes each ablation effect on each draw.

**Code.** `tools/onion_eval/region_checks.py` (this branch), run with kit CO6 (`PYTHONPATH=$K/tools:$K/src`) and the venv `/project/def-maxwl/mforooz/candi_venv`. Outputs on Fir: `/project/def-maxwl/mforooz/ONION/region_checks/{signal,bootstrap}/`. The tables beside this file are copies, except `region_signal_candidates.tsv` (1.5 MB), which stays on Fir.

## Which terms use which regions

I read this in the scorer (kit code equals this branch) and confirmed it in `runs_fir/rung2_base_heldout/seed0/p16_sv/scores.json` (`n_regions` 256, `pair_regions_n` 16, `region_bins` 16384, `context_bins` 1024).

| term | regions | where in the code |
|---|---|---|
| imputation (variance recovered) | all 256 | imputation jobs are not pair jobs; `_job_arrays` keeps every region (`score.py:1683`, `1697`); metrics at `score.py:2906` |
| calibration | all 256 | the PIT (probability integral transform) curves of the same imputation records (`score.py:2972`, `3232`) |
| gap closed on crop pairs with the oracle reference | 16 pair regions | correction pairs are pair jobs (`score.py:1683`, `1697`); the term is rebuilt from `pairs_per_region.npz` and the oracle reference at report time (`oracle_floor.py:303`, `select_rule.py:297`, `841`) |
| correction-only | 16 pair regions | the correction pairs and the same-view jobs (kind "correction", so also pair jobs) (`score.py:830`, `844`, `2335`) |

The 16 pair regions are the eval set's `pair_regions` [0, 16, ..., 240] (`score.py:1114-1116`). Each region is 16,384 bins of 25 bp (409.6 kb). The scorer cuts a region into 16 windows of 1,024 bins (25.6 kb) with no overlap; each window is predicted alone (`score.py:1401-1403`; the prediction store uses the same tiles, `predict_store.py:164-172`). The regions are sorted by chromosome name and start, so every 16th region spreads the 16 over 16 different chromosome positions (table below).

Per-region data exist only for the two correction terms. `suffstats.npz` holds the imputation Gram matrices per assay and the calibration PIT curves per target and stratum, pooled over the 256 regions. So imputation and calibration cannot be resampled by region without a rescore. They already use 256 regions.

## Check 1: signal content of the regions

**Method.** The population is the 6,724 candidate blocks of the eval set (`evalset.candidate_blocks` on the genome mask; I checked that the count equals the eval set's `n_candidate_regions`). These blocks cover about 2.75 Gb, so they are nearly the whole mappable genome. Per block I measured the following. First, the share of bins in each stratum of `masks.h5`. Second, the mean Umap k36 mappability decile (0 = least mappable). Third, direct signal from the 43 distinct imputation target tracks of the eval set (validation tracks, raw counts, 22 assays, 25 cells; `signal_tracks.tsv`). Per track, the "count enrichment" of a block is its mean count over the mean count of all candidate bins. A "high bin" is a bin above the track's 90th percentile over all candidate bins; the "high-bin enrichment" of a block is its share of high bins over that share in all candidate blocks. A block's value is the median over the 43 tracks. A value of 1 is the genome average. I call a block "near-zero signal" when its median count enrichment is below 0.25.

**Result** (`region_signal_summary.tsv`; min / quartiles / max):

| measure | 16 pair regions | 256 eval regions | 6,724 candidate blocks | genome (all bins) |
|---|---|---|---|---|
| count enrichment, median over tracks | 0.61 / 0.85 / 0.99 / 1.24 / 1.71 | 0.12 / 0.68 / 0.92 / 1.14 / 2.13 | 0.04 / 0.69 / 0.92 / 1.17 / 2.33 | 1 by definition |
| percentile of count enrichment among candidates | 18 / 42 / 58 / 80 / 98 | 0.3 / 25 / 49 / 73 / 99.9 | 0 / 25 / 50 / 75 / 100 | |
| high-bin enrichment, median over tracks | 0.32 / 0.69 / 1.00 / 1.32 / 2.40 | 0.02 / 0.47 / 0.81 / 1.18 / 2.57 | 0.004 / 0.46 / 0.82 / 1.21 / 3.40 | |
| share of promoter bins (mean) | 0.0042 | 0.0054 | 0.0051 | 0.0048 |
| share of enhancer bins (mean) | 0.214 | 0.176 | 0.177 | 0.164 |
| share of gene-body bins (mean) | 0.433 | 0.441 | 0.437 | 0.405 |
| share of heterochromatin bins (mean) | 0.058 | 0.070 | 0.072 | 0.070 |
| mean mappability decile (mean) | 5.23 | 4.78 | 4.75 | |
| blocks with near-zero signal (count enrichment below 0.25) | 0 of 16 | 2 of 256 | 42 of 6,724 | |
| blocks with count enrichment below 0.5 | 0 of 16 | 15 of 256 | 476 of 6,724 | |

The 16 pair regions (`region_signal_eval256.tsv`, rows with `in_pair16` = 1):

| eval index | position | promoters | enhancers | gene bodies | heterochromatin | count enrichment | percentile |
|---|---|---|---|---|---|---|---|
| 0 | chr1:8.19 Mb | 0.006 | 0.378 | 0.672 | 0.004 | 1.60 | 95.6 |
| 16 | chr1:218.32 Mb | 0.003 | 0.264 | 0.294 | 0.034 | 1.03 | 62.5 |
| 32 | chr10:90.93 Mb | 0.006 | 0.201 | 0.296 | 0.038 | 1.03 | 62.6 |
| 48 | chr12:100.35 Mb | 0.006 | 0.141 | 0.621 | 0.043 | 0.88 | 44.8 |
| 64 | chr13:75.78 Mb | 0.002 | 0.156 | 0.205 | 0.058 | 0.78 | 34.6 |
| 80 | chr15:46.28 Mb | 0.003 | 0.082 | 0.000 | 0.090 | 0.65 | 22.1 |
| 96 | chr18:48.74 Mb | 0.005 | 0.440 | 0.653 | 0.024 | 1.71 | 97.8 |
| 112 | chr2:54.89 Mb | 0.010 | 0.293 | 0.656 | 0.021 | 1.37 | 87.2 |
| 128 | chr20:15.97 Mb | 0.001 | 0.173 | 0.466 | 0.095 | 0.92 | 49.9 |
| 144 | chr3:13.52 Mb | 0.005 | 0.293 | 0.374 | 0.198 | 1.33 | 85.2 |
| 160 | chr3:138.85 Mb | 0.004 | 0.159 | 0.659 | 0.100 | 0.93 | 50.7 |
| 176 | chr4:92.98 Mb | 0.002 | 0.065 | 1.000 | 0.020 | 0.61 | 18.5 |
| 192 | chr5:109.36 Mb | 0.004 | 0.183 | 0.317 | 0.039 | 0.96 | 54.0 |
| 208 | chr6:91.34 Mb | 0.002 | 0.097 | 0.000 | 0.048 | 0.65 | 21.8 |
| 224 | chr7:134.35 Mb | 0.005 | 0.273 | 0.265 | 0.029 | 1.13 | 71.8 |
| 240 | chr9:88.88 Mb | 0.005 | 0.219 | 0.458 | 0.092 | 1.21 | 78.0 |

**Reading.** The 16 pair regions are slightly richer in signal than the genome: their median is at the 58th percentile and their mean enhancer share is 0.214 against 0.164 genome-wide. The 256 eval regions match the candidate distribution closely (quartiles 25 / 49 / 73). In my judgment the "dead region" concern does not hold. The 16 do under-sample the low tail: the poorest is at the 18th percentile, and 476 candidate blocks (7.1%) have count enrichment below 0.5. Every pair-region term therefore says nothing about the lowest-signal 18% of the genome. The genome mask had already removed blacklisted and unassembled blocks before the draw.

## Check 2: bootstrap of the 16 pair regions

**Reproduction first.** For each seed, the script recomputes both correction terms from the per-region sums: per pair, CRPS (continuous ranked probability score) = sum of the region sums over the sum of the bins; then the scorer's aggregation (per assay a ratio of pair means, mean over the correction-scorable assays, for the gap closed; per pair type a pooled ratio with the denominator tolerance, then a mean over pair types, for the correction-only term). With all 16 regions, the recomputed values equal the stored values for all 9 base seeds; the worst absolute difference was 4.6e-15 (`reproduction.tsv`). The stored gap closed is the value `oracle_floor.seed_terms` returns (subset with no duplicate-keeping side, the column `correction_v3_raw` of the arm summary), because `scores.json` does not store this term. The stored correction-only term is `correction_only_covariate_cost_closed.value` of `scores.json`. The base means equal the arm summary (0.7177 and 0.6904 for the conv autoencoder).

**Method.** 5,000 draws of the 16 regions with replacement, seed 20261117. One draw gives a weight per region (how often the region was drawn). The SAME weights apply to every seed of the arm and of the base (a paired bootstrap). Effect = (mean over arm seeds) minus (mean over base seeds). "Sign flips" is the share of draws where the effect has the opposite sign to the all-region effect. The leave-one-region-out range sets one weight to 0, for each of the 16 regions. "Beyond seed range" is the rule of the ablation read: the absolute effect is larger than the larger of the arm and base seed ranges (max minus min over seeds). "Region-robust" means the 95% interval excludes 0 and the sign flips in at most 5% of draws. The bootstrap holds the trained models fixed, so its interval measures region sampling only, not seed noise.

### Region-sampling noise floor of the base models (`bootstrap_base_floor.tsv`)

| model | term | value, all 16 regions | seed range | 95% bootstrap interval | interval width |
|---|---|---|---|---|---|
| conv autoencoder | gap closed (crop pairs, oracle) | 0.718 | 0.012 | [0.656, 0.788] | 0.132 |
| conv autoencoder | correction-only | 0.690 | 0.019 | [0.648, 0.756] | 0.107 |
| conv + attention | gap closed (crop pairs, oracle) | 0.673 | 0.031 | [0.597, 0.757] | 0.160 |
| conv + attention | correction-only | 0.659 | 0.043 | [0.617, 0.725] | 0.107 |
| U-Net | gap closed (crop pairs, oracle) | 0.507 | 0.071 | [0.485, 0.524] | 0.039 |
| U-Net | correction-only | 0.318 | 0.127 | [0.203, 0.517] | 0.313 |

The level of each term depends on the regions much more than on the seed (interval widths 0.04 to 0.31 against seed ranges 0.01 to 0.13). The arm-minus-base differences below are much narrower, because the arm and the base see the same regions and a hard region is hard for both.

### Effects (`bootstrap_effects.tsv`, `loro_regions.tsv`)

| arm (what it changes) | model | term | arm minus base | 95% interval | sign flips | leave-one-out range | beyond seed range | region-robust |
|---|---|---|---|---|---|---|---|---|
| covariates scrambled across tracks | conv autoencoder | gap closed (crop pairs, oracle) | -0.052 | [-0.064, -0.040] | 0.0% | [-0.056, -0.049] | yes | yes |
| covariates scrambled across tracks | conv autoencoder | correction-only | -0.164 | [-0.251, -0.114] | 0.0% | [-0.190, -0.156] | yes | yes |
| covariates scrambled across tracks | conv + attention | gap closed (crop pairs, oracle) | -0.004 | [-0.033, +0.024] | 37.1% | [-0.017, -0.000] | no | no |
| covariates scrambled across tracks | conv + attention | correction-only | -0.198 | [-0.295, -0.143] | 0.0% | [-0.229, -0.188] | yes | yes |
| covariates scrambled across tracks | U-Net | gap closed (crop pairs, oracle) | -0.075 | [-0.089, -0.063] | 0.0% | [-0.080, -0.073] | no | yes |
| covariates scrambled across tracks | U-Net | correction-only | -0.212 | [-0.328, -0.137] | 0.0% | [-0.249, -0.200] | yes | yes |
| training on base views only | conv autoencoder | gap closed (crop pairs, oracle) | +0.014 | [-0.025, +0.051] | 28.3% | [-0.004, +0.019] | no | no |
| training on base views only | conv autoencoder | correction-only | -0.289 | [-0.369, -0.237] | 0.0% | [-0.306, -0.279] | yes | yes |
| training on base views only | conv + attention | gap closed (crop pairs, oracle) | +0.003 | [-0.037, +0.042] | 49.2% | [-0.016, +0.008] | no | no |
| training on base views only | conv + attention | correction-only | -0.270 | [-0.333, -0.206] | 0.0% | [-0.288, -0.260] | yes | yes |
| training on base views only | U-Net | gap closed (crop pairs, oracle) | +0.046 | [+0.039, +0.052] | 0.0% | [+0.043, +0.047] | no | yes |
| training on base views only | U-Net | correction-only | -0.249 | [-0.340, -0.202] | 0.0% | [-0.285, -0.240] | yes | yes |
| training on base + alignment views | conv autoencoder | gap closed (crop pairs, oracle) | -0.102 | [-0.146, -0.061] | 0.0% | [-0.121, -0.097] | yes | yes |
| training on base + alignment views | conv autoencoder | correction-only | -0.212 | [-0.298, -0.167] | 0.0% | [-0.241, -0.204] | yes | yes |
| training on base + alignment views | conv + attention | gap closed (crop pairs, oracle) | -0.078 | [-0.122, -0.035] | 0.0% | [-0.098, -0.073] | yes | yes |
| training on base + alignment views | conv + attention | correction-only | -0.222 | [-0.292, -0.177] | 0.0% | [-0.236, -0.214] | yes | yes |
| training on base + alignment views | U-Net | gap closed (crop pairs, oracle) | +0.011 | [+0.006, +0.015] | 0.0% | [+0.010, +0.012] | no | yes |
| training on base + alignment views | U-Net | correction-only | -0.271 | [-0.357, -0.226] | 0.0% | [-0.300, -0.264] | yes | yes |
| all views, same-view pairs only | conv autoencoder | gap closed (crop pairs, oracle) | +0.012 | [-0.003, +0.028] | 7.7% | [+0.005, +0.015] | no | no |
| all views, same-view pairs only | conv autoencoder | correction-only | -0.036 | [-0.067, +0.016] | 11.7% | [-0.041, -0.007] | no | no |
| all views, same-view pairs only | conv + attention | gap closed (crop pairs, oracle) | +0.010 | [-0.008, +0.032] | 15.2% | [+0.007, +0.016] | no | no |
| all views, same-view pairs only | conv + attention | correction-only | +0.040 | [+0.029, +0.054] | 0.0% | [+0.038, +0.043] | no | yes |
| all views, same-view pairs only | U-Net | gap closed (crop pairs, oracle) | -14.349 | [-16.576, -12.036] | 0.0% | [-14.630, -13.531] | no | yes |
| all views, same-view pairs only | U-Net | correction-only | undefined | undefined | n/a | n/a | n/a | n/a |
| two-step correction, native base depth | conv autoencoder | gap closed (crop pairs, oracle) | -0.115 | [-0.150, -0.081] | 0.0% | [-0.127, -0.109] | yes | yes |
| two-step correction, native base depth | conv autoencoder | correction-only | +0.017 | [-0.012, +0.044] | 11.2% | [+0.012, +0.027] | no | no |
| two-step correction, native base depth | conv + attention | gap closed (crop pairs, oracle) | -0.059 | [-0.073, -0.046] | 0.0% | [-0.063, -0.056] | yes | yes |
| two-step correction, native base depth | conv + attention | correction-only | +0.106 | [+0.072, +0.125] | 0.0% | [+0.094, +0.109] | yes | yes |
| two-step correction, native base depth | U-Net | gap closed (crop pairs, oracle) | -0.064 | [-0.079, -0.054] | 0.0% | [-0.068, -0.062] | no | yes |
| two-step correction, native base depth | U-Net | correction-only | -0.233 | [-0.273, -0.204] | 0.0% | [-0.242, -0.228] | yes | yes |
| two-step correction, fixed base depth | conv autoencoder | gap closed (crop pairs, oracle) | -0.085 | [-0.106, -0.064] | 0.0% | [-0.089, -0.078] | yes | yes |
| two-step correction, fixed base depth | conv autoencoder | correction-only | +0.116 | [+0.109, +0.124] | 0.0% | [+0.114, +0.118] | yes | yes |
| two-step correction, fixed base depth | conv + attention | gap closed (crop pairs, oracle) | -0.055 | [-0.070, -0.039] | 0.0% | [-0.060, -0.052] | yes | yes |
| two-step correction, fixed base depth | conv + attention | correction-only | +0.141 | [+0.121, +0.154] | 0.0% | [+0.129, +0.144] | yes | yes |
| two-step correction, fixed base depth | U-Net | gap closed (crop pairs, oracle) | +0.129 | [+0.117, +0.141] | 0.0% | [+0.127, +0.132] | yes | yes |
| two-step correction, fixed base depth | U-Net | correction-only | +0.060 | [-0.001, +0.095] | 2.7% | [+0.028, +0.065] | no | no |
| fixed input depth | conv autoencoder | gap closed (crop pairs, oracle) | -0.007 | [-0.020, +0.005] | 15.3% | [-0.010, -0.001] | no | no |
| fixed input depth | conv autoencoder | correction-only | +0.038 | [+0.019, +0.067] | 0.0% | [+0.033, +0.050] | no | yes |
| fixed input depth | conv + attention | gap closed (crop pairs, oracle) | +0.049 | [+0.037, +0.062] | 0.0% | [+0.045, +0.054] | yes | yes |
| fixed input depth | conv + attention | correction-only | +0.007 | [-0.021, +0.055] | 33.0% | [+0.002, +0.027] | no | no |
| fixed input depth | U-Net | gap closed (crop pairs, oracle) | +0.168 | [+0.133, +0.207] | 0.0% | [+0.163, +0.183] | yes | yes |
| fixed input depth | U-Net | correction-only | +0.104 | [+0.021, +0.155] | 0.7% | [+0.070, +0.114] | no | yes |
| training-cell average as an extra input | conv autoencoder | gap closed (crop pairs, oracle) | +0.026 | [+0.007, +0.042] | 0.2% | [+0.018, +0.030] | no | yes |
| training-cell average as an extra input | conv autoencoder | correction-only | +0.071 | [+0.056, +0.086] | 0.0% | [+0.068, +0.082] | yes | yes |
| training-cell average as an extra input | conv + attention | gap closed (crop pairs, oracle) | +0.025 | [-0.000, +0.048] | 2.6% | [+0.014, +0.028] | no | no |
| training-cell average as an extra input | conv + attention | correction-only | +0.054 | [+0.041, +0.070] | 0.0% | [+0.051, +0.063] | no | yes |
| training-cell average as an extra input | U-Net | gap closed (crop pairs, oracle) | -0.006 | [-0.015, +0.001] | 6.2% | [-0.008, -0.002] | no | no |
| training-cell average as an extra input | U-Net | correction-only | +0.081 | [+0.046, +0.104] | 0.0% | [+0.061, +0.086] | no | yes |
| imputation loss weight 4 | conv autoencoder | gap closed (crop pairs, oracle) | -0.038 | [-0.057, -0.015] | 0.0% | [-0.041, -0.030] | yes | yes |
| imputation loss weight 4 | conv autoencoder | correction-only | +0.060 | [+0.045, +0.083] | 0.0% | [+0.058, +0.072] | yes | yes |
| imputation loss weight 4 | conv + attention | gap closed (crop pairs, oracle) | +0.000 | [-0.027, +0.032] | 46.5% | [-0.004, +0.010] | no | no |
| imputation loss weight 4 | conv + attention | correction-only | +0.101 | [+0.091, +0.114] | 0.0% | [+0.099, +0.105] | yes | yes |
| imputation loss weight 4 | U-Net | gap closed (crop pairs, oracle) | +0.025 | [+0.020, +0.030] | 0.0% | [+0.023, +0.026] | no | yes |
| imputation loss weight 4 | U-Net | correction-only | +0.051 | [+0.011, +0.078] | 0.1% | [+0.025, +0.054] | no | yes |
| numeric-only covariate embedding | conv autoencoder | gap closed (crop pairs, oracle) | -0.030 | [-0.050, -0.009] | 0.2% | [-0.039, -0.027] | no | yes |
| numeric-only covariate embedding | conv autoencoder | correction-only | -0.004 | [-0.016, +0.017] | 37.3% | [-0.006, +0.005] | no | no |
| numeric-only covariate embedding | conv + attention | gap closed (crop pairs, oracle) | -0.049 | [-0.085, -0.014] | 0.3% | [-0.065, -0.045] | no | yes |
| numeric-only covariate embedding | conv + attention | correction-only | -0.062 | [-0.081, -0.026] | 0.0% | [-0.068, -0.048] | no | yes |
| numeric-only covariate embedding | U-Net | gap closed (crop pairs, oracle) | -0.133 | [-0.152, -0.118] | 0.0% | [-0.138, -0.130] | yes | yes |
| numeric-only covariate embedding | U-Net | correction-only | -0.172 | [-0.197, -0.135] | 0.0% | [-0.175, -0.156] | no | yes |
| count transform local ratio | conv autoencoder | gap closed (crop pairs, oracle) | -0.138 | [-0.171, -0.101] | 0.0% | [-0.142, -0.121] | yes | yes |
| count transform local ratio | conv autoencoder | correction-only | +0.137 | [+0.111, +0.158] | 0.0% | [+0.131, +0.143] | yes | yes |
| count transform local ratio | conv + attention | gap closed (crop pairs, oracle) | -0.094 | [-0.117, -0.066] | 0.0% | [-0.097, -0.082] | yes | yes |
| count transform local ratio | conv + attention | correction-only | +0.126 | [+0.107, +0.137] | 0.0% | [+0.120, +0.129] | yes | yes |
| count transform local ratio | U-Net | gap closed (crop pairs, oracle) | -0.111 | [-0.164, -0.065] | 0.0% | [-0.117, -0.082] | yes | yes |
| count transform local ratio | U-Net | correction-only | +0.250 | [+0.082, +0.370] | 0.0% | [+0.144, +0.262] | yes | yes |
| count transform quantile | conv autoencoder | gap closed (crop pairs, oracle) | -0.135 | [-0.156, -0.115] | 0.0% | [-0.139, -0.125] | yes | yes |
| count transform quantile | conv autoencoder | correction-only | +0.086 | [+0.077, +0.095] | 0.0% | [+0.084, +0.089] | yes | yes |
| count transform quantile | conv + attention | gap closed (crop pairs, oracle) | -0.077 | [-0.084, -0.067] | 0.0% | [-0.079, -0.074] | yes | yes |
| count transform quantile | conv + attention | correction-only | +0.039 | [+0.024, +0.048] | 0.0% | [+0.033, +0.044] | no | yes |
| count transform quantile | U-Net | gap closed (crop pairs, oracle) | -0.083 | [-0.137, -0.033] | 0.0% | [-0.089, -0.054] | yes | yes |
| count transform quantile | U-Net | correction-only | +0.238 | [+0.070, +0.362] | 0.0% | [+0.128, +0.252] | yes | yes |

Counts over the 71 defined rows: 39 are beyond the seed range and region-robust; 18 are region-robust but within the seed range; 14 are neither; none is beyond the seed range and not region-robust.

The U-Net arm trained on all views with same-view pairs only is not a usable reading. Its crop-pair gap closed has a seed range of 29.2 (one seed far below the others), and its correction-only term is undefined in at least one seed, so the mean is undefined.

### Single regions

Two regions are the most influential in most rows: region 96 (chr18:48.74 Mb, the richest of the 16, 97.8th percentile, enhancer share 0.44) for the correction-only term, and region 192 (chr5:109.36 Mb, 54th percentile) for the crop-pair gap closed. No single region flips the sign of an effect that is beyond the seed range. Removing one region moves the following beyond-seed-range effects by 20% or more:

| arm | model | term | all 16 regions | without the region | region removed | still beyond seed range |
|---|---|---|---|---|---|---|
| count transform quantile | U-Net | correction-only | +0.238 | +0.128 | 96 | yes (seed range 0.127) |
| count transform local ratio | U-Net | correction-only | +0.250 | +0.144 | 96 | yes (0.127) |
| count transform quantile | U-Net | gap closed (crop pairs, oracle) | -0.083 | -0.054 | 192 | no (0.071) |
| count transform local ratio | U-Net | gap closed (crop pairs, oracle) | -0.111 | -0.082 | 192 | yes (0.071) |
| base + alignment views | conv + attention | gap closed (crop pairs, oracle) | -0.078 | -0.098 | 192 | yes (0.031) |
| imputation loss weight 4 | conv autoencoder | gap closed (crop pairs, oracle) | -0.038 | -0.030 | 192 | no (0.031) |
| imputation loss weight 4 | conv autoencoder | correction-only | +0.060 | +0.072 | 96 | yes (0.058) |

In my judgment, two effects are near the edge of the seed-range rule and their "beyond" label depends on one region: the U-Net crop-pair gap closed under the quantile transform, and the conv autoencoder crop-pair gap closed under imputation loss weight 4. Their sign does not depend on the regions.

## Which ablation effects are in doubt

Rule: the paired bootstrap interval includes 0, or the sign flips in more than 5% of draws. Each of these was already "within the seed range" in the ablation read, so the read made no claim on any of them.

1. Covariates scrambled, conv + attention, crop-pair gap closed: -0.004, interval [-0.033, +0.024], flips 37%.
2. Base views only, conv autoencoder, crop-pair gap closed: +0.014, [-0.025, +0.051], flips 28%.
3. Base views only, conv + attention, crop-pair gap closed: +0.003, [-0.037, +0.042], flips 49%.
4. All views with same-view pairs only, conv autoencoder, crop-pair gap closed: +0.012, [-0.003, +0.028], flips 7.7%.
5. All views with same-view pairs only, conv autoencoder, correction-only: -0.036, [-0.067, +0.016], flips 11.7%.
6. All views with same-view pairs only, conv + attention, crop-pair gap closed: +0.010, [-0.008, +0.032], flips 15.2%.
7. All views with same-view pairs only, U-Net, correction-only: undefined (see above).
8. Two-step correction with native base depth, conv autoencoder, correction-only: +0.017, [-0.012, +0.044], flips 11.2%.
9. Two-step correction with fixed base depth, U-Net, correction-only: +0.060, [-0.001, +0.095], flips 2.7%.
10. Fixed input depth, conv autoencoder (2 seeds), crop-pair gap closed: -0.007, [-0.020, +0.005], flips 15.3%.
11. Fixed input depth, conv + attention, correction-only: +0.007, [-0.021, +0.055], flips 33%.
12. Training-cell average as input, conv + attention, crop-pair gap closed: +0.025, [-0.0002, +0.048], flips 2.6%.
13. Training-cell average as input, U-Net, crop-pair gap closed: -0.006, [-0.015, +0.001], flips 6.2%.
14. Imputation loss weight 4, conv + attention, crop-pair gap closed: +0.000, [-0.027, +0.032], flips 47%.
15. Numeric-only covariate embedding, conv autoencoder, correction-only: -0.004, [-0.016, +0.017], flips 37%.

## Effects that hold

All 39 effects that are beyond the seed range are also region-robust. Among them: scrambled covariates lower the correction-only term on all three models (-0.164, -0.198, -0.212); base views only and base + alignment views lower the correction-only term on all three models (-0.21 to -0.29); base + alignment views lower the crop-pair gap closed on both conv models; native two-step lowers the crop-pair gap closed on both conv models (-0.115, -0.059); fixed input depth raises the U-Net crop-pair gap closed (+0.168, interval [+0.133, +0.207]); numeric-only covariates lower the U-Net crop-pair gap closed (-0.133, interval [-0.152, -0.118]); local-ratio and quantile transforms lower the crop-pair gap closed on all three models; the local-ratio transform raises the correction-only term on all three models. The full list is the rows with "yes" in both verdict columns.

## What I did not check

- Imputation and calibration were not resampled. They use all 256 regions, and no per-region data exist for them; a per-region split needs a rescore with new per-region outputs.
- The bootstrap resamples the 16 regions that were scored. It cannot show how an effect behaves in region types that none of the 16 represents, in particular the lowest-signal 18% of the genome (the poorest pair region is at the 18th percentile).
- The interval does not include seed noise. I used the seed range rule of the ablation read for that, unchanged; I did not build a joint seed and region bootstrap.
- I did not score the ablations on the 256 regions; that is the planned 16-versus-256 pair-region run.
- Signal content uses the 43 validation target tracks only (raw counts, base views). I did not read the training tracks or the correction-pair tracks for check 1.
- Ablations outside the list of the brief (for example the DNA, injection-form and mask-mode ablations) were not resampled; the script does it by adding their names to `EFFECTS`.
