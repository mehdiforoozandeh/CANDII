# Observed-data ladder probe (t390), 3 blind EIC experiments

Written 2026-11-22 by the ladder build agent on branch `implementation/t390-eic-pvalue-entry`.
No model is involved. Each rung is an "entry" track on the EIC 25 bp grid, scored by the exact EIC scorer
(`tools/onion_eval/eic_score.py score`, nine measures, blacklist deleted, chromosome set all23) against two truths:
**qn** (Jacob Schreiber's SE reprocessing, quantile normalized, Zenodo 14736610, the MOESM4 truth) and
**se** (the same tracks unnormalized, Zenodo 14736608). Code: `tools/eic_entry/` (`ladder_probe.py`, `run_probe.sh`, `collect.py`).

## Experiments (assay codes from `tools/t112/arms.py` TRACKS)

| experiment | assay | treatment tagAlign (lines) | control (lines) | d (from `metadata.json` `--fraglen` / `--smooth-win`) |
|---|---|---|---|---|
| C19M22 | H3K4me3 (sharp) | ENCFF748TKZ ... nodup.30M (30,000,000) | ENCFF433TZR, full (107,039,349) | 205 |
| C40M17 | H3K27me3 (broad) | ENCFF581NOL ... nodup.30M (30,000,000) | ENCFF337JNL, full (150,302,589) | 200 |
| C12M02 | DNase-seq | ENCFF211XVI ... no_chrM_MT.50M (50,000,000) | none | 150 |

TagAligns: Nibi `/project/def-maxwl/mforooz/t112_cf/tagaligns/ta/`; line counts from the `starts` step (`checks.tsv`).
Their names match the inputs of `chip.macs2_signal_track` / `atac.macs2_signal_track` in each `metadata.json`
(`bam2ta --subsample 30000000` / `50000000`; `choose_ctl` used the full control). Mean read length: 101.0 bp (ChIP), 75.5 bp (DNase).

## Rungs

| rung | what | how |
|---|---|---|
| L1 | Jacob's SE track, raw | `zenodo/blind_se_reprocessed/<exp>.bigwig` |
| L2 | L1 quantile normalized (QN) by Max's rule, Jacob's peaks | `ladder_probe.py qnorm` (port of Max's `apply_bp.py`), ref `s04/qnorm/ref/ref_M##.npz` |
| L3 | our pipeline's -log10 p bigWig, raw | `EIC_REPRO/003_pipeline/results/<exp>/macs2_signal_track_*.pval.signal.bigwig` |
| L4 | L3 QN, our pipeline's raw call-peak set (`call_peak_shard0_*.bfilt.narrowPeak.gz`, 500K / 300K cap) | `qnorm` |
| L4o | (added) L3 QN, our pipeline's pseudoreplicate-overlap peaks (`overlap_pr_shard0_*.overlap.bfilt.narrowPeak.gz`) | `qnorm` |
| L5 | MACS2 formula from 1 bp read 5' ends of the treatment and control tagAligns | `ladder_probe.py formula` (port of Max's sprint 10 `phys1.py`, conv gtr64) |
| L6 | count route: 25 bp overlap counts -> `signal_from_counts.signal_track` (plug_in, box_d, own control; DNase macs2_no_control) | `ladder_probe.py counts_route` |
| L7 | L6 QN, peaks called on L6 | `callpeaks` (bins >= 2, merge gaps <= read length, drop runs < d, top 500K / 300K by max), then `qnorm` |

L4o was added after the first results: Jacob's `.bed.gz` peaks were a 95-100% subset of our raw call-peak set, and for C12M02
their intervals are identical to our overlap set (`comm` of the sorted first 3 columns: 0 lines differ; peakcheck Jaccard 1.0).

## Checks (each must pass before a rung is read)

| check | C19M22 | C40M17 | C12M02 | source |
|---|---|---|---|---|
| L2 equals the qn truth, per base (chr1-22, X) | 0 of 3,031,042,417 bases differ; max abs 0 | 0 of 3,031,042,417; 0 | 0 of 3,031,042,417; 0 | `rungs/L2/<exp>.npy.json` |
| L5 vs L3, normalized MSE (max abs diff; Pearson) | 6.14e-5 (1.46; 0.999985) | 1.12e-4 (0.96; 0.999945) | 1.36e-5 (1.27; 0.999994) | `qc/<exp>_L5_vs_L3.json` |
| `callpeaks` rule on L3 vs pipeline raw peaks, base Jaccard | 0.902 | 0.636 | 0.817 | `qc/<exp>_peakcheck_L3rule_vs_pipeline.json` |
| Jacob's peaks vs our overlap peaks, base Jaccard | 0.952 | 0.615 | 1.000 | `qc/<exp>_peakcheck_jacob_vs_overlap.json` |
| Jacob's peaks vs our raw call-peak set, base Jaccard | 0.862 | 0.212 | 0.616 | `qc/<exp>_peakcheck_jacob_vs_pipeline.json` |
| SE-truth scorer vs Max's independent SE scores (Avocado, C19M22, all23) | equal to ~1e-13 relative on all nine measures (e.g. msevar 924.6037177 both) | | | `score_cache/se_unnormalized/scores/team_0/C19M22.tsv` vs `/project/def-maxwl/maxwl/epi_imputation/s04/scores/se/Avocado/C19M22.csv` |
| scorer fast path vs reference | <= 2.4e-12 every scoring | | | score job logs |

Normalized MSE (nMSE) here is mean((A - B)^2) / var(B) on the scorer grid (25 bp, blacklist bins deleted, 121,240,998 bins).
The L5 check passes Max's bound (<= 0.0002 of variance), so the t112 30M / 50M tagAligns are the reads the pipeline scored
(inferred from this match; no md5 of the Cromwell inputs exists, those folders are archived).

## Rung-vs-rung normalized MSE (`rung_compare.tsv`)

| pair | meaning | C19M22 | C40M17 | C12M02 |
|---|---|---|---|---|
| L5 vs L3 | formula vs pipeline | 6.1e-5 | 1.1e-4 | 1.4e-5 |
| L6 vs L5 | cost of 25 bp overlap counts | 0.0172 | 0.2550 | 0.0212 |
| L6 vs L3 | count route vs pipeline | 0.0171 | 0.2552 | 0.0212 |
| L3 vs L1 | our reprocessing vs Jacob's | 0.0132 | 0.1267 | 0 (bigWigs byte-identical, md5 084a93c2...) |
| L4 vs L2 | QN with our raw peaks vs truth | 0.0412 | 0.6864 | 0.1106 |
| L4o vs L2 | QN with our overlap peaks vs truth | 0.0340 | 0.1692 | 0 (exact) |
| L7 vs L4 | count route + own peaks vs pipeline + raw peaks | 0.0437 | 0.2825 | 0.0548 |

asinh-space nMSE (as in Max's notes) is in the TSV: L6 vs L5 is 0.121 / 0.265 / 0.072.

## Scores and placement

Cell = value (rank among the 25 published entrants on that experiment, all23; 1 = better than or equal to every entrant,
26 = worse than all). Entrant qn scores: our MOESM4-reproducing cache `/scratch/mforooz/onion/eic/score_cache/qnorm/scores/team_*`;
entrant se scores: scored here (job 28513030). Full rows: `scores_all23.tsv`, `placement.tsv`.

### C12M02, truth qn

| rung | mse | gwcorr | gwspear | mseprom | msegene | mseenh | msevar | mse1obs | mse1imp |
|---|---|---|---|---|---|---|---|---|---|
| L1 | 191.3 (10) | 0.9078 (1) | 0.9579 (1) | 2226 (15) | 249.3 (10) | 1567 (1) | 2.674e+05 (15) | 1.905e+04 (14) | 1.904e+04 (23) |
| L2 | 0 (1) | 1 (1) | 1 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) |
| L3 | 191.3 (10) | 0.9078 (1) | 0.9579 (1) | 2226 (15) | 249.3 (10) | 1567 (1) | 2.674e+05 (15) | 1.905e+04 (14) | 1.904e+04 (23) |
| L4 | 28.29 (1) | 0.9864 (1) | 0.9995 (1) | 228.2 (1) | 36.95 (1) | 233.8 (1) | 8721 (1) | 2430 (1) | 2470 (2) |
| L4o | 0 (1) | 1 (1) | 1 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) |
| L5 | 191.3 (10) | 0.9078 (1) | 0.9578 (1) | 2226 (15) | 249.4 (10) | 1567 (1) | 2.674e+05 (15) | 1.905e+04 (14) | 1.905e+04 (23) |
| L6 | 191.1 (10) | 0.8999 (1) | 0.8699 (1) | 2226 (15) | 249.2 (10) | 1562 (1) | 2.678e+05 (15) | 1.902e+04 (14) | 1.899e+04 (23) |
| L7 | 33.92 (1) | 0.9652 (1) | 0.903 (1) | 304.4 (1) | 42.26 (1) | 296.4 (1) | 2.847e+04 (1) | 3042 (1) | 3127 (2) |
| best entrant | 153.3 | 0.7303 | 0.609 | 1344 | 197.7 | 1599 | 2.193e+05 | 1.359e+04 | 2090 |
| median entrant | 202.8 | 0.5735 | 0.5478 | 2219 | 273.8 | 1863 | 2.608e+05 | 1.858e+04 | 1.468e+04 |

### C19M22, truth qn

| rung | mse | gwcorr | gwspear | mseprom | msegene | mseenh | msevar | mse1obs | mse1imp |
|---|---|---|---|---|---|---|---|---|---|
| L1 | 12.69 (1) | 0.9534 (1) | 0.983 (1) | 140.8 (1) | 23.05 (1) | 56.49 (1) | 3221 (1) | 1267 (1) | 1268 (2) |
| L2 | 0 (1) | 1 (1) | 1 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) |
| L3 | 12.77 (1) | 0.9414 (1) | 0.9195 (1) | 142.1 (1) | 23.17 (1) | 57.26 (1) | 3242 (1) | 1273 (1) | 1275 (2) |
| L4 | 1.921 (1) | 0.9827 (1) | 0.9322 (1) | 22.76 (1) | 3.37 (1) | 9.116 (1) | 305.9 (1) | 189.9 (1) | 190.8 (1) |
| L4o | 1.583 (1) | 0.9832 (1) | 0.9322 (1) | 18.88 (1) | 2.811 (1) | 7.113 (1) | 274.1 (1) | 156.7 (1) | 157.2 (1) |
| L5 | 12.68 (1) | 0.9409 (1) | 0.9192 (1) | 141.1 (1) | 23.01 (1) | 56.83 (1) | 3223 (1) | 1264 (1) | 1266 (2) |
| L6 | 13.84 (1) | 0.9286 (1) | 0.7498 (1) | 154.3 (3) | 25.02 (2) | 62.2 (1) | 3496 (2) | 1379 (4) | 1381 (4) |
| L7 | 3.594 (1) | 0.9623 (1) | 0.7588 (1) | 41.73 (1) | 6.334 (1) | 17.06 (1) | 657.6 (1) | 351.9 (1) | 355.4 (1) |
| best entrant | 14 | 0.8404 | 0.1474 | 147 | 24.64 | 76.26 | 3335 | 1305 | 505.1 |
| median entrant | 21.41 | 0.818 | 0.108 | 226.3 | 38.1 | 107.8 | 4362 | 1952 | 1958 |

### C40M17, truth qn

| rung | mse | gwcorr | gwspear | mseprom | msegene | mseenh | msevar | mse1obs | mse1imp |
|---|---|---|---|---|---|---|---|---|---|
| L1 | 0.0464 (1) | 0.9208 (1) | 0.9763 (1) | 0.2106 (1) | 0.05704 (1) | 0.2352 (1) | 5.701 (1) | 2.678 (1) | 3.394 (2) |
| L2 | 0 (1) | 1 (1) | 1 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) |
| L3 | 0.07443 (1) | 0.8662 (1) | 0.9175 (1) | 0.2466 (1) | 0.08675 (1) | 0.2755 (1) | 5.893 (1) | 2.91 (1) | 3.417 (2) |
| L4 | 0.2002 (1) | 0.8634 (1) | 0.9369 (1) | 0.6634 (1) | 0.2191 (1) | 0.5889 (1) | 5.962 (1) | 10.43 (20) | 9.002 (19) |
| L4o | 0.04933 (1) | 0.9205 (1) | 0.9369 (1) | 0.1372 (1) | 0.05431 (1) | 0.1304 (1) | 1.272 (1) | 1.465 (1) | 1.632 (2) |
| L5 | 0.07397 (1) | 0.8671 (1) | 0.9172 (1) | 0.2437 (1) | 0.0861 (1) | 0.2719 (1) | 5.801 (1) | 2.876 (1) | 3.375 (2) |
| L6 | 0.1168 (1) | 0.7761 (1) | 0.7631 (1) | 0.332 (1) | 0.1329 (1) | 0.361 (1) | 7.201 (1) | 4.12 (1) | 4.721 (4) |
| L7 | 0.1578 (1) | 0.7939 (1) | 0.7769 (1) | 0.4321 (1) | 0.1735 (1) | 0.4089 (1) | 4.086 (1) | 4.757 (1) | 3.959 (2) |
| best entrant | 0.2747 | 0.2896 | 0.2063 | 0.7431 | 0.3117 | 0.7798 | 13.35 | 8.791 | 0.7719 |
| median entrant | 0.2917 | 0.2476 | 0.1685 | 0.7775 | 0.329 | 0.8209 | 14.28 | 9.652 | 7.257 |

### C12M02, truth se

| rung | mse | gwcorr | gwspear | mseprom | msegene | mseenh | msevar | mse1obs | mse1imp |
|---|---|---|---|---|---|---|---|---|---|
| L1 | 0 (1) | 1 (1) | 1 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) |
| L2 | 191.3 (26) | 0.9078 (1) | 0.9579 (1) | 2226 (23) | 249.3 (25) | 1567 (26) | 2.656e+04 (26) | 1.904e+04 (26) | 1.905e+04 (26) |
| L3 | 0 (1) | 1 (1) | 1 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) |
| L4 | 328.1 (26) | 0.9454 (1) | 0.9579 (1) | 3543 (25) | 429.2 (26) | 2754 (26) | 3.959e+04 (26) | 3.215e+04 (26) | 3.215e+04 (26) |
| L4o | 191.3 (26) | 0.9078 (1) | 0.9579 (1) | 2226 (23) | 249.3 (25) | 1567 (26) | 2.656e+04 (26) | 1.904e+04 (26) | 1.905e+04 (26) |
| L5 | 8.25e-05 (1) | 1 (1) | 0.9998 (1) | 0.0005563 (1) | 0.0001063 (1) | 0.0006809 (1) | 0.004634 (1) | 0.007704 (1) | 0.007678 (1) |
| L6 | 0.1281 (1) | 0.9898 (1) | 0.9032 (1) | 0.638 (1) | 0.1606 (1) | 0.7903 (1) | 5.386 (1) | 7.93 (1) | 7.788 (1) |
| L7 | 298.1 (26) | 0.9243 (1) | 0.8445 (1) | 3191 (25) | 385.8 (26) | 2558 (26) | 3.837e+04 (26) | 2.929e+04 (26) | 2.931e+04 (26) |
| best entrant | 2.757 | 0.7496 | 0.6556 | 15.64 | 3.557 | 27.9 | 346.8 | 243.6 | 94.82 |
| median entrant | 20.89 | 0.5701 | 0.5902 | 550.4 | 35.46 | 84.65 | 1262 | 1928 | 1889 |

### C19M22, truth se

| rung | mse | gwcorr | gwspear | mseprom | msegene | mseenh | msevar | mse1obs | mse1imp |
|---|---|---|---|---|---|---|---|---|---|
| L1 | 0 (1) | 1 (1) | 1 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) |
| L2 | 12.69 (14) | 0.9534 (1) | 0.983 (1) | 140.8 (14) | 23.05 (16) | 56.49 (17) | 1381 (16) | 1268 (17) | 1267 (16) |
| L3 | 0.1723 (1) | 0.9939 (1) | 0.9318 (1) | 1.939 (1) | 0.2856 (1) | 0.894 (1) | 13.17 (1) | 15.67 (1) | 15.73 (1) |
| L4 | 15.6 (19) | 0.9599 (1) | 0.9171 (1) | 173.1 (18) | 28.16 (19) | 69.28 (18) | 1660 (19) | 1558 (21) | 1558 (19) |
| L4o | 13.48 (16) | 0.9465 (1) | 0.9171 (1) | 148.6 (16) | 24.38 (16) | 58.51 (17) | 1447 (16) | 1346 (19) | 1345 (16) |
| L5 | 0.1778 (1) | 0.9939 (1) | 0.9315 (1) | 2.003 (1) | 0.2949 (1) | 0.9258 (1) | 13.66 (1) | 16.22 (1) | 16.27 (1) |
| L6 | 0.373 (1) | 0.9857 (1) | 0.7625 (1) | 4.09 (1) | 0.603 (1) | 1.946 (1) | 29.98 (1) | 32.74 (1) | 33.01 (1) |
| L7 | 14.42 (18) | 0.9353 (1) | 0.7584 (1) | 159.4 (16) | 26.19 (18) | 64.25 (17) | 1506 (17) | 1437 (21) | 1436 (18) |
| best entrant | 3.834 | 0.8862 | 0.1635 | 39.32 | 6.516 | 23.66 | 443.2 | 363.1 | 127.5 |
| median entrant | 12.01 | 0.8348 | 0.1237 | 136.2 | 21.23 | 50.96 | 1134 | 921.7 | 890.1 |

### C40M17, truth se

| rung | mse | gwcorr | gwspear | mseprom | msegene | mseenh | msevar | mse1obs | mse1imp |
|---|---|---|---|---|---|---|---|---|---|
| L1 | 0 (1) | 1 (1) | 1 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) |
| L2 | 0.0464 (1) | 0.9208 (1) | 0.9763 (1) | 0.2106 (1) | 0.05704 (1) | 0.2352 (1) | 2.051 (18) | 3.394 (1) | 2.678 (8) |
| L3 | 0.02589 (1) | 0.9442 (1) | 0.9345 (1) | 0.03604 (1) | 0.02768 (1) | 0.03493 (1) | 0.09398 (1) | 0.242 (1) | 0.4207 (2) |
| L4 | 0.2695 (20) | 0.8196 (1) | 0.9133 (1) | 1.113 (22) | 0.3135 (20) | 1.1 (22) | 7.766 (22) | 19.81 (26) | 12.43 (21) |
| L4o | 0.08446 (1) | 0.8607 (1) | 0.9132 (1) | 0.3424 (3) | 0.1006 (1) | 0.3517 (5) | 2.783 (19) | 4.776 (1) | 3.972 (17) |
| L5 | 0.02596 (1) | 0.9442 (1) | 0.9342 (1) | 0.03641 (1) | 0.02777 (1) | 0.03525 (1) | 0.09646 (1) | 0.2465 (1) | 0.426 (2) |
| L6 | 0.06647 (1) | 0.8248 (1) | 0.7816 (1) | 0.08993 (1) | 0.0704 (1) | 0.08574 (1) | 0.2634 (1) | 1.077 (1) | 0.9174 (2) |
| L7 | 0.189 (1) | 0.7472 (1) | 0.7757 (1) | 0.673 (20) | 0.2188 (3) | 0.6821 (20) | 4.707 (21) | 9.133 (25) | 5.559 (18) |
| best entrant | 0.194 | 0.2811 | 0.2236 | 0.3126 | 0.2074 | 0.3067 | 1.419 | 5.136 | 0.3475 |
| median entrant | 0.22 | 0.2359 | 0.1782 | 0.4012 | 0.2368 | 0.4046 | 1.873 | 5.712 | 3.52 |


## Wall time and peak memory per step (`resources.tsv`, `sacct`, Nibi, 2 CPUs per job unless noted)

| step | C19M22 | C40M17 | C12M02 |
|---|---|---|---|
| starts, treatment | 0:44, 2.6 GB | 0:44, 2.5 GB | 1:10, 2.8 GB |
| starts, control | 1:37, 3.1 GB | 1:59, 3.9 GB | none |
| L2 qnorm + per-base verify | 13:05, 24.8 GB | 11:53, 24.4 GB | 10:49, 23.9 GB |
| L4 qnorm | 8:45, 20.9 GB | 6:27, 21.2 GB | 6:27, 20.3 GB |
| L4o qnorm + checks | 7:26, 20.0 GB | 7:38, 20.3 GB | 6:44, 19.9 GB |
| L5 formula | 4:55, 10.4 GB | 5:34, 10.4 GB | 2:40, 10.3 GB |
| L6 count route + peak calls + peakchecks | 1:27, 5.0 GB | 1:40, 5.4 GB | 1:25, 3.9 GB |
| L7 qnorm | 8:30, 20.8 GB | 7:18, 20.5 GB | 6:40, 20.1 GB |
| scoring (7 rungs x 2 truths + 6 comparisons) | 1:00:53, 20.5 GB | 2:30:08 * | 2:29:28 * |

\* The C40M17 and C12M02 score jobs ran at the same time and each scored both experiments (the skip-if-exists check races),
so each is about twice its own work. One scoring (one rung, one experiment, one truth) took 216-264 s (C19M22 log); the L4o job
(job 28589609, 6 scorings) took 31:33. One-off: `eicx_setup_se` (16 CPUs) binned the 51 SE truth tracks, built msevar weights
and scored the 25 entrants on the 3 experiments against the SE truth: 3:59:51, peak 64.0 GiB (at its 64G request).

## Job ids (Nibi, all COMPLETED except the first score job)

setup 28513030; C19M22 starts 28513031, 28513032, L2 28513033, L4 28513034, L5 28513035, L6 28513036, L7 28513037,
score 28513038 (FAILED in 7 s: `eic_score.py` run without the `candi` stubs; fixed in b5069b8a) and 28550707;
C40M17 starts 28555045, 28555046, L2 28555047, L4 28555048, L5 28555049, L6 28555050, L7 28555051, score 28555052;
C12M02 starts 28555064, L2 28555144, L4 28555145, L5 28555146, L6 28555147, L7 28555148, score 28555149;
L4o 28589606, 28589607, 28589608, score 28589609.

## Nibi paths (big arrays stay there; /scratch, purge-exposed)

- Rung tracks: `/scratch/mforooz/eicx/ladder/rungs/{L1..L7,L4o}/<exp>.{bigwig,npy}` (.npy = challenge format, pickled dict, raw 25 bp grid) with a `.json` report beside each.
- Read starts and 25 bp counts: `/scratch/mforooz/eicx/ladder/starts/<tagAlign stem>.{npz,json}`.
- Peaks called on L6 and on L3: `/scratch/mforooz/eicx/ladder/peaks/<exp>_{L6,L3rule}.bed`.
- Scores: `/scratch/mforooz/eicx/ladder/scores/{qn,se}/<rung>.tsv`; placements `placement_{qn,se}.tsv`; comparisons `qc/`; logs `logs/`.
- SE score cache (truth bins, msevar weights, entrant scores): `/scratch/mforooz/eicx/ladder/score_cache/se_unnormalized/`.

## What this means

1. Both deterministic transforms are exact. Max's QN rule reproduces the qn truth bit for bit (3 of 3), and the 1 bp MACS2 formula reproduces the pipeline track to nMSE <= 1.1e-4 (3 of 3).
2. For DNase-seq (C12M02) our full pipeline route equals the truth: L3 is byte-identical to Jacob's SE track, and L4o equals the qn truth exactly. The peak set that the truth's QN used is the pipeline's pseudoreplicate-overlap set, not the raw call-peak set.
3. For the two ChIP experiments our reprocessing does not reproduce Jacob's tracks (L3 vs L1 nMSE 0.013 and 0.127). The cause is not found. L5 reproduces L3, so the difference lies before the p-value step (the reads, the control or a pipeline setting), and it limits how closely any ChIP entry built from our reads can match the truth.
4. The 25 bp count route (L6) costs nMSE 0.017 (H3K4me3), 0.021 (DNase) and 0.255 (H3K27me3) against the exact formula. The broad mark loses the most. This is the floor for a count-space model scored through `signal_from_counts` with plug_in and box_d, before any model error.
5. QN dominates the qn scores. Raw rungs (L1, L3, L5, L6) have qn mse 12.7-13.8 on C19M22 and 191 on C12M02 (rank 10 on C12M02). The QN rungs (L2, L4, L4o, L7) fall to 0-3.6 and 0-34. The peak set matters: on C40M17, QN with the raw call-peak set (L4) is worse than no QN on mse1obs/mse1imp (rank 20 and 19), while the overlap set (L4o) ranks 1-2.
6. The full observed count route (L7) ranks 1 of 26 on every qn measure for C19M22, and 1 or 2 for C40M17 and C12M02. Against the se truth, every QN rung ranks 14-26 on the mse family for C19M22 and C12M02 (C40M17 is mixed, 1-26). An entry must use the normalization of the truth it is scored against.
7. These placements are an oracle bound. Every rung uses the blind experiment's own reads, so they bound the conversion cost and say nothing about a model's imputation quality.

## Blocked or not done

- Nothing in L1-L7 was blocked for these 3 experiments.
- The L7 peak caller approximates the raw MACS2 call-peak set (Jaccard 0.90 / 0.64 / 0.82 with the pipeline on L3). It does not reproduce the pseudoreplicate-overlap set that the truth's QN used, so L7 is the analogue of L4, not of L4o. A count-route analogue of L4o (peaks reproducible across two read halves) is not built.
- The cause of the ChIP L3 vs L1 gap is not investigated.

## Cost estimate for all 48 experiments (from the measured times above)

Per experiment: starts 1-3 min (controls are shared), L2 11-13 min, L4 and L4o 6-9 min each, L5 3-6 min, L6 1.5 min,
L7 7-8.5 min, scoring 8 rungs x 2 truths x about 4 min = about 64 min. That is about 1.8 h of 2-CPU jobs (about 3.6 CPU-h),
peak memory 26 GB (QN). For 48 experiments: about 175 CPU-h (48 x 3.6), plus the one-off SE setup for the other 45 experiments' entrant
scores (the 3-experiment setup took 4 h on 16 CPUs, mostly the 51-track SE binning, which is now cached). In parallel this is a few hours of wall time.
Extra work not measured here: only 7 tracks have pipeline tagAligns (`t112_cf/tagaligns/ta`). 34 more experiments have nodup BAMs but no tagAlign: each needs `bam2ta` plus the pipeline's own seeded subsample (`--subsample 30000000`, seeded by file size), and a check of L5 vs L3 before use. 7 experiments have no BAM in `EIC_REPRO/.../results` (C05M17, C05M18, C05M20, C05M29, C06M16, C28M22, C38M02): L3-L7 are blocked for them unless the reads are recovered (the Cromwell trees are on nearline tape, `nibi-nearline-archive-2026-11-17`). Run the score jobs one at a time per rung folder, or give each experiment its own rung folder, to avoid the duplicate scoring seen above.
