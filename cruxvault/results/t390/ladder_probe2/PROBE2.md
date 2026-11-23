# Observed-data ladder probe 2 (t390): peak sets and the fragment-pileup route

Written 2026-11-23 on branch `implementation/t390-eic-pvalue-entry`. This probe extends probe 1 (`../ladder_probe/LADDER_PROBE.md`).
It uses the same 3 blind experiments, the same scorer (`tools/onion_eval/eic_score.py`, nine measures, all23) and the same two truths (qn, se).
Code: `tools/eic_entry/` (`ladder_probe.py` subcommands `split`, `overlap`, `fragpile_route`, `arrdiff`; `run_probe.sh p2`; `overlap_container_check.sh`; `collect2.py`).
TSVs in this folder come from `bash runall.sh`.

## New rungs

| rung | what | how |
|---|---|---|
| L7t | L6 QN with the pipeline's overlap peaks (an oracle) | `qnorm --x L6 --mask overlap_pr_*.overlap.bfilt.narrowPeak.gz` |
| L7o | L6 QN with peaks reproducible across two read halves | (1) `split`: two random halves of the treatment reads (numpy seed 20261122; pr1 = (N+1)//2 lines, as `encode_task_spr.py:40-81`). (2) the count route (L6) and the probe-1 peak caller (`callpeaks`) on each half and on all reads. (3) `overlap`: the pipeline's naive-overlap rule and blacklist filter. (4) `qnorm` |
| L6f | the count route from a 25 bp fragment pileup | `fragpile_route`: each read extended to d along its strand (DNase-seq: centred on the 5' end), mean coverage per 25 bp bin; score = -log10 P(Poisson(lambda) > pile25), lambda from the 25 bp overlap counts / kappa |
| L7f | L6f QN with the L7o rule applied to L6f tracks | as L7o, with `fragpile_route` in place of the count route |

**Overlap rule (source).** The rule is `encode_task_overlap.py:89-91, 109-117` in `/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline/chip-seq-pipeline2/src/` (v2.2.2). The atac-seq-pipeline v2.2.3 copy is byte-identical.
The rule keeps a pooled peak when it overlaps some pr1 peak by at least half of either peak's length. The same filter is then applied against pr2.
After that, `encode_lib_blacklist_filter.py:62-66` drops peaks that touch the blacklist ENCFF356LFX (md5 393688b4...).
`chip.wdl:1989-2001` feeds this step the unfiltered pr1, pr2 and pooled peaks.

**Fragment-pileup conversion (source).** The conversion ports Max Libbrecht's sprint 10 `scripts/testbed/cac_lib.py` (in `projects/03_2026-10-20_epi_imputation/sprints/10_2026-11-11_covariate_correction/`). The functions are `pc_features` lines 277-292 (the "pf" feature), `mlog10_sf` 183-199, `box` 150-156, `lam_hat25` 170-180 and `Data.kappa` 110-118. Max applied it as "formula on the 25 bp fragment pileup" in `cac_train.py:408-411`.
Deviation: the pileup is built from 1 bp read 5' ends, so it is the exact 25 bp mean. Max used 5 bp starts.

## Checks (`checks.tsv`, `peaks.tsv`)

| check | C19M22 | C40M17 | C12M02 |
|---|---|---|---|
| L6 rerun vs probe 1 L6, raw grid | 0 of 121,241,707 bins differ | 0 differ | 0 differ |
| rescoring probe 1's L7 from a fresh folder vs `../ladder_probe` qn L7 row | max relative diff 0 (9 measures) | 0 | 0 |
| overlap port vs the pipeline's own script (in its container), our L6 halves | identical (16,752 peaks) | identical (4,223) | identical (85,439) |
| same, our L6f halves | identical (17,829) | identical (10,617) | identical (106,849) |
| same, the pipeline's own bfilt pr1 / pr2 / pooled peaks | identical (19,401) | 12,201 vs 12,202 (a) | 100,393 vs 100,583 (a) |
| pipeline script on bfilt inputs vs the pipeline's overlap file | identical (19,401) | identical (12,202) | 100,583 vs 100,584 (b) |
| port on the pipeline's peaks vs the pipeline's overlap file, base Jaccard | 1.0 | 1.0 | 0.99999 |

(a) The only difference is chrY. The port reads chr1-22 and chrX (the scorer's set); the script keeps `chr[\dXY]+`. C40M17 has 1 chrY interval (chrY:6908897-6909204) and C12M02 has 279 chrY rows. With chrY removed, the C12M02 sets have the same md5.
(b) One interval, chr12:130316251-130316700, is only in the pipeline file. The pipeline ran the rule on unfiltered peaks; only the blacklist-filtered peaks are on disk.

Container check output: Nibi `qc/<exp>_container_overlap.json`.

## Peak sets against the pipeline's overlap set (`peaks.tsv`, base Jaccard)

| peak set | C19M22 | C40M17 | C12M02 |
|---|---|---|---|
| L7o (count route, half-read rule) | 0.925 | 0.380 | 0.768 |
| L7f (fragment pileup, half-read rule) | 0.954 | 0.535 | 0.802 |
| peaks kept by the rule, L7o / L7f / pipeline | 16,752 / 17,829 / 19,401 | 4,223 / 10,617 / 12,201 | 85,439 / 106,849 / 100,393 unique intervals |

For C40M17, the L7o set covers 3.64 Mb and the pipeline overlap set covers 8.18 Mb. 89% of the L7o bases are in the pipeline set, but only 40% of the pipeline bases are in L7o. The count-route caller loses broad-mark peaks in each half.
The L6f peaks on all reads have Jaccard 0.915 / 0.682 / 0.817 with the pipeline's raw call-peak set. The probe-1 rule on L3 had 0.902 / 0.636 / 0.817.

## Rung-vs-rung normalized MSE (`rung_compare.tsv`; probe-1 pairs from `../ladder_probe/rung_compare.tsv`)

nMSE = mean((A - B)^2) / var(B) on the scorer grid (121,240,998 bins).

| pair | meaning | C19M22 | C40M17 | C12M02 |
|---|---|---|---|---|
| L6 vs L5 (probe 1) | cost of 25 bp overlap counts | 0.0172 | 0.2550 | 0.0212 |
| **L6f vs L5** | cost of a 25 bp fragment pileup | **0.0021** | **0.0616** | **3.0e-5** |
| L6f vs L3 | fragment-pileup route vs pipeline | 0.0022 | 0.0620 | 1.8e-5 |
| L7t vs L4o | count route vs pipeline track, same (overlap) peaks | 0.0438 | 0.2706 | 0.0595 |
| L7o vs L4o | count route + own half-read peaks vs pipeline + overlap peaks | 0.0445 | 0.3122 | 0.0644 |
| L7f vs L4o | fragment route + own half-read peaks vs pipeline + overlap peaks | 0.0043 | 0.0967 | 0.0076 |
| L7t vs L2 | vs the qn truth | 0.0763 | 0.4148 | 0.0595 |
| L7o vs L2 | vs the qn truth | 0.0766 | 0.3988 | 0.0644 |
| L7f vs L2 | vs the qn truth | 0.0374 | 0.2249 | 0.0076 |
| L7o vs L7 | half-read rule vs probe-1 single-pass peaks | 0.0015 | 0.1521 | 0.0924 |

For C12M02, L4o equals L2 exactly (probe 1), so the L2 and L4o columns agree.

## Scores and placement (`scores_all23.tsv`, `placement.tsv`)

Each cell gives the value, then the rank among the 25 published entrants on that experiment (all23). Rank 1 means better than or equal to every entrant; rank 26 means worse than all.
Rows L2-L7 are probe 1 (`../ladder_probe/`). The bold rows are new.

### C19M22 (H3K4me3), truth qn

| rung | mse | gwcorr | gwspear | mseprom | msegene | mseenh | msevar | mse1obs | mse1imp |
|---|---|---|---|---|---|---|---|---|---|
| L4o | 1.583 (1) | 0.9832 (1) | 0.9322 (1) | 18.88 (1) | 2.811 (1) | 7.113 (1) | 274.1 (1) | 156.7 (1) | 157.2 (1) |
| L5 | 12.68 (1) | 0.9409 (1) | 0.9192 (1) | 141.1 (1) | 23.01 (1) | 56.83 (1) | 3223 (1) | 1264 (1) | 1266 (2) |
| L6 | 13.84 (1) | 0.9286 (1) | 0.7498 (1) | 154.3 (3) | 25.02 (2) | 62.2 (1) | 3496 (2) | 1379 (4) | 1381 (4) |
| L7 | 3.594 (1) | 0.9623 (1) | 0.7588 (1) | 41.73 (1) | 6.334 (1) | 17.06 (1) | 657.6 (1) | 351.9 (1) | 355.4 (1) |
| **L7t** | 3.556 (1) | 0.9624 (1) | 0.7588 (1) | 41.55 (1) | 6.271 (1) | 16.65 (1) | 658.4 (1) | 351.5 (1) | 351.7 (1) |
| **L7o** | 3.57 (1) | 0.9616 (1) | 0.7588 (1) | 41.66 (1) | 6.289 (1) | 16.7 (1) | 665.8 (1) | 352.9 (1) | 352.9 (1) |
| **L6f** | 12.61 (1) | 0.9394 (1) | 0.8312 (1) | 140.1 (1) | 22.87 (1) | 56.49 (1) | 3202 (1) | 1256 (1) | 1257 (2) |
| **L7f** | 1.743 (1) | 0.9814 (1) | 0.8374 (1) | 20.73 (1) | 3.092 (1) | 7.9 (1) | 301.8 (1) | 172.2 (1) | 172.5 (1) |
| best entrant | 14 | 0.8404 | 0.1474 | 147 | 24.64 | 76.26 | 3335 | 1305 | 505.1 |
| median entrant | 21.41 | 0.818 | 0.108 | 226.3 | 38.1 | 107.8 | 4362 | 1952 | 1958 |

### C40M17 (H3K27me3), truth qn

| rung | mse | gwcorr | gwspear | mseprom | msegene | mseenh | msevar | mse1obs | mse1imp |
|---|---|---|---|---|---|---|---|---|---|
| L4o | 0.04933 (1) | 0.9205 (1) | 0.9369 (1) | 0.1372 (1) | 0.05431 (1) | 0.1304 (1) | 1.272 (1) | 1.465 (1) | 1.632 (2) |
| L5 | 0.07397 (1) | 0.8671 (1) | 0.9172 (1) | 0.2437 (1) | 0.0861 (1) | 0.2719 (1) | 5.801 (1) | 2.876 (1) | 3.375 (2) |
| L6 | 0.1168 (1) | 0.7761 (1) | 0.7631 (1) | 0.332 (1) | 0.1329 (1) | 0.361 (1) | 7.201 (1) | 4.12 (1) | 4.721 (4) |
| L7 | 0.1578 (1) | 0.7939 (1) | 0.7769 (1) | 0.4321 (1) | 0.1735 (1) | 0.4089 (1) | 4.086 (1) | 4.757 (1) | 3.959 (2) |
| **L7t** | 0.121 (1) | 0.8093 (1) | 0.7769 (1) | 0.2871 (1) | 0.1329 (1) | 0.2834 (1) | 2.961 (1) | 2.769 (1) | 2.546 (2) |
| **L7o** | 0.1163 (1) | 0.7893 (1) | 0.7769 (1) | 0.263 (1) | 0.1273 (1) | 0.2723 (1) | 3.351 (1) | 2.844 (1) | 2.424 (2) |
| **L6f** | 0.09081 (1) | 0.8502 (1) | 0.8451 (1) | 0.2557 (1) | 0.1036 (1) | 0.2823 (1) | 5.687 (1) | 2.774 (1) | 3.94 (2) |
| **L7f** | 0.06557 (1) | 0.8936 (1) | 0.8541 (1) | 0.1705 (1) | 0.07155 (1) | 0.159 (1) | 1.554 (1) | 1.862 (1) | 1.609 (2) |
| best entrant | 0.2747 | 0.2896 | 0.2063 | 0.7431 | 0.3117 | 0.7798 | 13.35 | 8.791 | 0.7719 |
| median entrant | 0.2917 | 0.2476 | 0.1685 | 0.7775 | 0.329 | 0.8209 | 14.28 | 9.652 | 7.257 |

### C12M02 (DNase-seq), truth qn

| rung | mse | gwcorr | gwspear | mseprom | msegene | mseenh | msevar | mse1obs | mse1imp |
|---|---|---|---|---|---|---|---|---|---|
| L4o | 0 (1) | 1 (1) | 1 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) |
| L5 | 191.3 (10) | 0.9078 (1) | 0.9578 (1) | 2226 (15) | 249.4 (10) | 1567 (1) | 2.674e+05 (15) | 1.905e+04 (14) | 1.905e+04 (23) |
| L6 | 191.1 (10) | 0.8999 (1) | 0.8699 (1) | 2226 (15) | 249.2 (10) | 1562 (1) | 2.678e+05 (15) | 1.902e+04 (14) | 1.899e+04 (23) |
| L7 | 33.92 (1) | 0.9652 (1) | 0.903 (1) | 304.4 (1) | 42.26 (1) | 296.4 (1) | 2.847e+04 (1) | 3042 (1) | 3127 (2) |
| **L7t** | 15.23 (1) | 0.9705 (1) | 0.9031 (1) | 172.3 (1) | 18.81 (1) | 137.1 (1) | 2.903e+04 (1) | 1491 (1) | 1489 (1) |
| **L7o** | 16.47 (1) | 0.968 (1) | 0.9031 (1) | 181.8 (1) | 20.61 (1) | 147.7 (1) | 3.138e+04 (1) | 1595 (1) | 1578 (1) |
| **L6f** | 191.2 (10) | 0.908 (1) | 0.9571 (1) | 2226 (15) | 249.3 (10) | 1567 (1) | 2.674e+05 (15) | 1.905e+04 (14) | 1.904e+04 (23) |
| **L7f** | 1.956 (1) | 0.9965 (1) | 0.9944 (1) | 22.39 (1) | 2.395 (1) | 12.15 (1) | 4297 (1) | 163 (1) | 168.2 (1) |
| best entrant | 153.3 | 0.7303 | 0.609 | 1344 | 197.7 | 1599 | 2.193e+05 | 1.359e+04 | 2090 |
| median entrant | 202.8 | 0.5735 | 0.5478 | 2219 | 273.8 | 1863 | 2.608e+05 | 1.858e+04 | 1.468e+04 |

### Truth se (the new rungs only; probe-1 rows are in `../ladder_probe/LADDER_PROBE.md`)

| experiment | rung | mse | gwcorr | gwspear | mseprom | msegene | mseenh | msevar | mse1obs | mse1imp |
|---|---|---|---|---|---|---|---|---|---|---|
| C19M22 | L7t | 14.27 (17) | 0.9338 (1) | 0.7584 (1) | 157.7 (16) | 25.92 (18) | 63.48 (17) | 1490 (17) | 1422 (21) | 1420 (18) |
| C19M22 | L7o | 13.75 (16) | 0.9292 (1) | 0.7584 (1) | 151.8 (16) | 24.99 (17) | 60.95 (17) | 1438 (16) | 1370 (19) | 1368 (16) |
| C19M22 | L6f | 0.212 (1) | 0.9931 (1) | 0.8448 (1) | 2.275 (1) | 0.3451 (1) | 1.074 (1) | 15.43 (1) | 18.34 (1) | 18.44 (1) |
| C19M22 | L7f | 13.48 (16) | 0.9422 (1) | 0.843 (1) | 148.7 (16) | 24.39 (16) | 58.44 (17) | 1447 (16) | 1345 (19) | 1344 (16) |
| C40M17 | L7t | 0.1428 (1) | 0.7586 (1) | 0.7758 (1) | 0.4547 (18) | 0.1643 (1) | 0.4693 (18) | 3.237 (19) | 5.593 (13) | 3.818 (17) |
| C40M17 | L7o | 0.1054 (1) | 0.7759 (1) | 0.7757 (1) | 0.2405 (1) | 0.1169 (1) | 0.2424 (1) | 1.611 (5) | 2.64 (1) | 2.243 (6) |
| C40M17 | L6f | 0.0444 (1) | 0.9243 (1) | 0.864 (1) | 0.05731 (1) | 0.04711 (1) | 0.05518 (1) | 0.1365 (1) | 0.3241 (1) | 0.7181 (2) |
| C40M17 | L7f | 0.08855 (1) | 0.8518 (1) | 0.8618 (1) | 0.3256 (2) | 0.1043 (1) | 0.3377 (3) | 2.631 (19) | 4.368 (1) | 2.772 (9) |
| C12M02 | L7t | 196.3 (26) | 0.893 (1) | 0.8444 (1) | 2223 (23) | 252.6 (25) | 1670 (26) | 2.815e+04 (26) | 1.952e+04 (26) | 1.953e+04 (26) |
| C12M02 | L7o | 164 (26) | 0.8764 (1) | 0.8445 (1) | 1899 (23) | 209.8 (25) | 1380 (26) | 2.462e+04 (26) | 1.635e+04 (26) | 1.635e+04 (26) |
| C12M02 | L6f | 0.0001072 (1) | 1 (1) | 0.9982 (1) | 0.0004628 (1) | 0.0001316 (1) | 0.0005909 (1) | 0.00357 (1) | 0.004654 (1) | 0.004648 (1) |
| C12M02 | L7f | 200.5 (26) | 0.9045 (1) | 0.965 (1) | 2335 (24) | 261.4 (25) | 1646 (26) | 2.788e+04 (26) | 1.996e+04 (26) | 1.994e+04 (26) |

## Wall time and peak memory (`resources.tsv`, `sacct`, Nibi, 2 CPUs per job)

| step | C19M22 | C40M17 | C12M02 |
|---|---|---|---|
| split + `starts` on both halves | 0:56, 4.1 GB | 3:20, 4.1 GB | 1:26, 6.6 GB |
| all reads: L6 rerun, L6f, peak calls | 1:29, 7.2 GB | 3:07, 7.3 GB | 1:20, 6.7 GB |
| halves: L6 and L6f per half, peak calls, overlap, peakchecks | 5:23, 8.2 GB | 4:07, 8.2 GB | 3:07, 7.7 GB |
| container check | 1:05, 0.1 GB | 0:13, 0.2 GB | 0:33, 0.5 GB |
| QN L7t / L7o / L7f | 5:39 / 6:33 / 6:41, 21.8 GB | 7:01 / 5:49 / 6:12, 22.0 GB | 6:32 / 5:09 / 5:42, 21.8 GB |
| scoring (9 scorings + 10 comparisons) | 31:27, 20.1 GB | 37:50, 20.6 GB | 2:00:39 *, 19.7 GB |

In total the probe used 9.0 CPU-h (the sum of elapsed time x 2 CPUs over the 24 jobs). The budget was 20 CPU-h.
\* The three score jobs ran at the same time and shared the truth cache. The C12M02 job ran about 3 times longer than the other two; the cause was not investigated.

## Job ids (Nibi)

C19M22: split 28604154, full 28604156, halves 28604157, container 28604158, L7t 28604159, L7o 28604160, L7f 28605161, score 28605164.
C40M17: 28610164, 28610165, 28610166, 28610167, 28610168, 28610169, 28610170, score 28610171.
C12M02: 28610172, 28610173, 28610174, 28610175, 28610176, 28610177, 28610178, score 28610179.

The three score jobs show FAILED. Their last step, `placement`, raised FileNotFoundError on `score_cache/qnorm/scores/team_3393457/<exp>.tsv`. The entrant qn scores live in the onion cache, `/scratch/mforooz/onion/eic/score_cache/qnorm/scores/`, not in the probe's own cache.
All 9 scorings and 10 comparisons per experiment had finished before that step (logs `logs/eicx_p2score_<exp>_<id>.out`). I reran `placement` by hand on the login node with the onion cache; it only reads TSVs.
`run_probe.sh` now passes `$ONION_CQN` to placement. No job was resubmitted.

## Nibi paths (`/scratch/mforooz/eicx/ladder`, purge-exposed)

- Rung tracks: `rungs2/<exp>/{L6,L6f,L7t,L7o,L7f}/<exp>.npy`. `L7chk` links to probe 1's L7.
- Half-read starts: `starts/<treatment>.pr{1,2}.{npz,json}`. Half tracks: `halves/<exp>_{L6,L6f}_pr{1,2}.npy`. The half tagAligns were deleted.
- Peaks: `peaks/<exp>_{L6p2,L6f,L6_pr1,L6_pr2,L6f_pr1,L6f_pr2,L7o,L7f,port_on_pipeline}.bed`.
- Container runs: `ctr/<exp>/`.
- Scores: `scores2/{qn,se}/<exp>_<rung>.tsv` and `scores2/placement_{qn,se}_<exp>.tsv`. Comparisons: `qc/`. Timings: `sacct2.tsv`.

## What this means

1. **The count step is the main loss, and a fragment pileup removes most of it.** The 25 bp fragment pileup cuts the conversion cost against the exact 1 bp formula from nMSE 0.0172 / 0.2550 / 0.0212 (overlap counts, L6) to 0.0021 / 0.0616 / 0.00003 (L6f). The reduction is 8x for H3K4me3, 4x for H3K27me3 and 700x for DNase-seq. For DNase-seq, L6f is effectively exact.
2. **With the truth's own peak set, the count route stays far from the pipeline track.** L7t vs L4o is 0.044 / 0.271 / 0.060. This is about the same as probe 1's L7 vs L4 (0.044 / 0.283 / 0.055). So with overlap counts, the error of L7 lay mostly in the counts, not in the peak set. On the qn truth, L7t improves on L7 for C40M17 and C12M02 (for example C12M02 mse 33.9 to 15.2), and only slightly for C19M22 (3.59 to 3.56).
3. **The half-read rule is a usable stand-in for the pipeline's overlap peaks.** On all three experiments, L7o scores within 8% of the oracle L7t: qn mse 3.57 vs 3.56, 0.116 vs 0.121, and 16.5 vs 15.2. This holds although its peak set agrees with the pipeline only in part (Jaccard 0.925 / 0.380 / 0.768). The broad mark loses most of its peaks, and QN is not very sensitive to that.
4. **The full entrant-side route (fragment pileup, then half-read peaks, then QN) is the closest to the truth.** L7f vs the qn truth has nMSE 0.037 / 0.225 / 0.0076. For L7o the values are 0.077 / 0.399 / 0.064. L7f ranks 1 among the 26 tracks on every qn measure, except rank 2 on mse1imp for C40M17. On C40M17 and C12M02, L7f comes close to L4o, which is the pipeline track normalized with the pipeline's peaks.
5. **An entry must match the truth's normalization.** Against the se truth, the QN rungs (L7t, L7o, L7f) rank 16-26 on the mse family for C19M22 and C12M02. The unnormalized L6f ranks 1-2 on every se measure.
6. **These rungs are oracle bounds.** Every rung uses the blind experiment's own reads. Together they show the cost of each conversion step. They say nothing about how well a model imputes.
7. **What a count-space model needs.** A model scored on the EIC qn board should output a 25 bp fragment pileup (or read starts by strand), not 25 bp overlap counts. It should also call its peaks with the half-read overlap rule on its own output. The model's halves must come from the model, for example two draws from its predicted distribution; this probe did not test that.

## Blocked or not done

- The pipeline's own pseudoreplicate split (GNU `shuf` seeded by file size, `encode_task_spr.py`) was not reproduced. The halves here are a numpy permutation with a fixed seed.
- The peak caller is still the probe-1 threshold caller, not MACS2. On the broad mark it recovers only 40% of the pipeline's overlap-set bases.
- The overlap set is built from the main chromosomes only (no chrY). It differs from the pipeline's set by chrY rows only.
- How to make the "two halves" from a model's prediction is not tested.
- The cause of the 3x longer C12M02 score job is not investigated.
