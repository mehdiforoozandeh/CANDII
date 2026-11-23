# The MOESM2 board for the t390 ladder rungs (3 blind EIC experiments)

Written 2026-11-23 on branch `implementation/t390-eic-pvalue-entry`.
MOESM2 is Additional file 2 of Schreiber et al., Genome Biology 2023: the 25 published entries scored against the
original 2019 blind truth (paired-end ENCODE processing, no reprocessing, no quantile normalization).
The PI made it the headline board for miniCANDI on 2026-11-23.
Code: `tools/onion_eval/eic_score.py` (MOESM2 record), `tools/eic_entry/run_moesm2.sh` (Nibi driver), `tools/eic_entry/moesm2.py` (`varcheck`, `collect`).
TSVs in this folder come from `bash runall.sh`.

## 1. Where the truth and the table are

| item | path | evidence |
|---|---|---|
| original blind truth, 51 bigWigs | Nibi `/scratch/maxwl/epi_imputation/data/synapse/blind_truth/C##M##.bigwig` (`eic_score.ORIGINAL_TRUTH`) | Max's sprint 04 `scripts/score/run.sh` calls this set `pe` ("original 2019 blind truth (paired-end, as scored in MOESM2)"); `ls` shows 51 files |
| binned truth and msevar weights | Nibi `/scratch/maxwl/epi_imputation/s04/cache/pe/{truth,var}` (Max's cache) | linked into our cache `/scratch/mforooz/eicx/moesm2/score_cache/pe/truth/` |
| Max's entrant scores against it | Nibi `/project/def-maxwl/maxwl/epi_imputation/s04/scores/pe/<entrant>/<exp>.csv` (25 entrants, 51 experiments; UIOWA 50) | |
| MOESM2 table | Nibi `/project/def-maxwl/mforooz/ONION/eic/published/13059_2023_2915_MOESM2_ESM.csv` (copied 2026-11-23 from Max's `/scratch/maxwl/epi_imputation/s04/code/inputs/`; nothing downloaded) | sha256 `48ad2e8d...dbff7b1`, pinned as `MOESM2_SHA256` |

Max already reproduced MOESM2 on all 51 experiments (sprint 04, `output/compare/pe_vs_MOESM2_summary.json` in his repo at
`projects/03_2026-10-20_epi_imputation/sprints/04-2026-10-28_eic/`): 25 of 25 leaderboard positions identical, D = 0, 12,740 rows joined,
median relative difference 2.7e-10 (mse) to 4.7e-7 (gwspear), largest 1.1e-3 (gwcorr). We reused his binned truth and his entrant scores.
The truth sits on Max's `/scratch`, so it can be purged; no `/project` copy was made.

## 2. The scoring rule (MOESM2 vs MOESM4)

MOESM2 uses the MOESM4 rule with the truth swapped. Both tables have the same columns, 12,740 rows, 25 teams, the same 51 experiments and
bootstraps 0-9 (read from both CSVs). The 25 bp mean binning, the blacklist deletion, the nine measures and the ten bootstrap chromosome groups
are the same. The msevar weights come from the scored truth itself. These are the per-assay variance across the 51 original tracks, with C38M02
left out (Max's `scripts/score/run.sh` stage 2, the same rule as MOESM4). Our rebuilt weights equal Max's pe weights exactly (max relative
difference 0) for M22, M17 and M02, and the members match (`checks.tsv`). Max's compare used the scored truth's own weights (`msevar_col msevar`).

## 3. Reproduction check

| check | result | file |
|---|---|---|
| our bigWig binning vs Max's binned pe truth (C19M22, C40M17, C12M02) | 0 of 69 chromosome arrays differ, bit for bit | `checks.tsv` |
| msevar weights, ours vs Max's (M22, M17, M02) | max relative difference 0 | `checks.tsv` |
| our scorer, Avocado + Average, 3 experiments x 10 bootstraps, vs MOESM2 | max relative difference 1.9e-8 (msevar) on 7 measures; mse1obs 5.7e-7; gwspear 1.5e-5 | `repro_ours.tsv` |
| same, vs Max's cached pe scores (11 chromosome sets incl. all23) | max relative difference 7.0e-13 | `repro_ours.tsv` |
| Max's cached scores, all 25 entrants, 3 experiments x 10 bootstraps, vs MOESM2 | max relative difference 6.8e-8 on 7 measures, mse1obs 6.5e-7, gwspear 1.2e-4; 0 per-measure rank differences among the 25 (750 rows per measure) | `repro_max.tsv` |
| 3-experiment leaderboard, our Avocado and Average rows in place of MOESM2's | 25 of 25 positions identical; rank statistic differs for 2 teams (CUWA 8.5 vs 9.0, BrokenNodes 14.0 vs 14.5) with no position change | `board_check.tsv` |
| fast scorer vs reference scorer | <= 3.1e-12, every scoring | `checks.tsv` |

The gwspear residual (1.5e-5 ours, 1.2e-4 Max's) is the size Max saw on MOESM4 too (his correlations up to 1.8e-3). It is from tie handling
in the Spearman ranks. It changes no rank. The ~1e-6 target holds for 8 of 9 measures (mse1obs 5.7e-7); gwspear misses it.

## 4. Rung placements against MOESM2

Rungs as in probe 1 (`../ladder_probe/LADDER_PROBE.md`): L3 = our pipeline's -log10 p bigWig; L5 = the MACS2 formula from read 5' ends;
L6 = 25 bp count route; L7 = L6 quantile normalized with its own peaks (QN, the qn-truth normalization).
Cell = value (rank among the 25 entrants, chromosome set all23, from Max's cached all23 scores; 1 = better than or equal to every entrant,
26 = worse than all). Full rows: `placement.tsv`, `scores_all23.tsv`.

### C19M22 (H3K4me3)

| rung | mse | gwcorr | gwspear | mseprom | msegene | mseenh | msevar | mse1obs | mse1imp |
|---|---|---|---|---|---|---|---|---|---|
| L3 | 889.5 (19) | 0.8215 (18) | 0.3537 (1) | 1.075e+04 (19) | 1662 (19) | 2251 (12) | 4.869e+05 (22) | 8.893e+04 (19) | 8.89e+04 (21) |
| L5 | 888.6 (19) | 0.8212 (18) | 0.353 (1) | 1.074e+04 (19) | 1661 (19) | 2248 (12) | 4.866e+05 (22) | 8.884e+04 (19) | 8.88e+04 (21) |
| L6 | 893.9 (19) | 0.8185 (18) | 0.3622 (1) | 1.08e+04 (20) | 1670 (19) | 2264 (12) | 4.89e+05 (22) | 8.938e+04 (19) | 8.934e+04 (21) |
| L7 | 741.7 (6) | 0.8372 (17) | 0.3571 (1) | 9044 (8) | 1390 (7) | 1754 (2) | 4.256e+05 (11) | 7.416e+04 (7) | 7.406e+04 (8) |
| best entrant | 676.6 | 0.8822 | 0.241 | 8130 | 1260 | 1747 | 3.74e+05 | 6.76e+04 | 1.149e+04 |
| median entrant | 846.5 | 0.8476 | 0.1798 | 1.017e+04 | 1578 | 2287 | 4.421e+05 | 8.462e+04 | 8.212e+04 |

### C40M17 (H3K27me3)

| rung | mse | gwcorr | gwspear | mseprom | msegene | mseenh | msevar | mse1obs | mse1imp |
|---|---|---|---|---|---|---|---|---|---|
| L3 | 0.5179 (1) | 0.4264 (1) | 0.4097 (1) | 1.601 (1) | 0.5804 (1) | 0.7394 (1) | 425.4 (1) | 25.07 (1) | 11.84 (4) |
| L5 | 0.5175 (1) | 0.4272 (1) | 0.4092 (1) | 1.598 (1) | 0.5798 (1) | 0.737 (1) | 425.1 (1) | 25.02 (1) | 11.91 (4) |
| L6 | 0.5061 (1) | 0.4282 (1) | 0.4221 (1) | 1.613 (1) | 0.5685 (1) | 0.7466 (1) | 426.8 (1) | 26.39 (1) | 13.7 (6) |
| L7 | 0.5987 (15) | 0.4288 (1) | 0.4135 (1) | 1.856 (8) | 0.6658 (15) | 0.9839 (6) | 392.3 (1) | 26.75 (1) | 10.64 (3) |
| best entrant | 0.5586 | 0.2963 | 0.3458 | 1.808 | 0.6355 | 0.9559 | 439.6 | 28.88 | 1.452 |
| median entrant | 0.5765 | 0.2538 | 0.2795 | 1.875 | 0.657 | 1.052 | 452.4 | 31.49 | 14.01 |

### C12M02 (DNase-seq)

| rung | mse | gwcorr | gwspear | mseprom | msegene | mseenh | msevar | mse1obs | mse1imp |
|---|---|---|---|---|---|---|---|---|---|
| L3 | 553.7 (21) | 0.5824 (20) | 0.6068 (1) | 9993 (22) | 696.1 (21) | 3431 (20) | 2.181e+07 (20) | 5.531e+04 (21) | 5.525e+04 (25) |
| L5 | 553.7 (21) | 0.5825 (20) | 0.6065 (1) | 9994 (22) | 696.2 (21) | 3431 (20) | 2.181e+07 (20) | 5.532e+04 (21) | 5.526e+04 (25) |
| L6 | 552.5 (21) | 0.5851 (20) | 0.593 (1) | 9979 (22) | 694.8 (21) | 3421 (18) | 2.181e+07 (20) | 5.52e+04 (21) | 5.514e+04 (25) |
| L7 | 404.4 (7) | 0.6132 (19) | 0.5515 (1) | 6196 (3) | 514.3 (7) | 2864 (4) | 2.074e+07 (18) | 3.877e+04 (6) | 3.971e+04 (9) |
| best entrant | 322.2 | 0.8216 | 0.5181 | 5426 | 401.4 | 2614 | 3.281e+06 | 3.218e+04 | 7577 |
| median entrant | 451.8 | 0.7152 | 0.443 | 8253 | 600 | 3285 | 1.656e+07 | 4.495e+04 | 4.269e+04 |

### Combined rank within each experiment (`placement_combined.tsv`)

This is the organizers' per-experiment statistic: the sum over the nine measures of log(rank/27), re-ranked, with the rung added alone to
MOESM2's 25 published rows. Cell = mean over bootstraps 0-9 (min-max), out of 26.

| rung | C19M22 | C40M17 | C12M02 |
|---|---|---|---|
| L3 | 15 (15-15) | 1 (1-1) | 17.1 (17-18) |
| L5 | 15 (15-15) | 1 (1-1) | 17.1 (17-18) |
| L6 | 15 (15-15) | 1 (1-1) | 17.1 (17-18) |
| L7 | 5.9 (5-6) | 1.4 (1-5) | 4.9 (4-5) |

## 5. What this means

1. The scorer reproduces MOESM2 on the 3 probe experiments. Values match to <= 1.9e-8 on 7 measures and 5.7e-7 on mse1obs. gwspear matches to 1.5e-5, which is the known tie-handling residual. Ranks match exactly.
2. Against the original PE truth, the observed-read rungs are not an oracle. Our reads are single-end reprocessed, and the truth is the original paired-end processing. L3, L5 and L6 rank 15 of 26 (C19M22) and 17 of 26 (C12M02) on the combined statistic, with mse ranks 19-21. Against the qn truth they ranked 1 (probe 1). This is the size of the covariate shift that MOESM2 tests. The C12M02 L3 track is byte-identical to Jacob's SE truth (probe 1), so the SE-vs-PE processing gap alone puts the observed DNase data at rank 21 of 26 on mse.
3. For H3K27me3 (C40M17), every raw rung ranks 1 on 8 of 9 measures. The broad-mark entrants are weak there (best entrant gwcorr 0.296).
4. Quantile normalization (L7) helps on MOESM2 too: combined rank 5.9 (C19M22) and 4.9 (C12M02), against 15 and 17.1 raw. On C40M17 it costs mse (rank 15 vs 1). This is measured on 3 experiments only, and L7's QN reference is the qn truth's, so this is an observation, not a rule.
5. The 25 bp count route costs nothing visible here. L6 ranks equal L3 and L5 to within 1-2 places on every measure.
6. The rungs use the blind experiment's own reads, so these numbers bound conversion and processing cost. They say nothing about a model's imputation quality.

## 6. Time, memory, jobs (`resources.tsv`, Nibi, 2 CPUs each)

| job | what | elapsed | peak RSS |
|---|---|---|---|
| 28627126 `eicx_m2_verify` | check-bins x 3, msevar weights for 8 assays, Avocado + Average on 3 experiments, board check, varcheck | 27:57 | 47.97 GB (at its 48G request) |
| 28627127 `eicx_m2_rungs` | L3, L5, L6, L7 on 3 experiments (12 scorings, 153-246 s each) | 39:12 | 21.36 GB |

Total 2.2 CPU-h ((27:57 + 39:12) x 2 CPUs). The verify job peaked at its memory request (the msevar build). A rerun of the weights should ask for 64G.
Outputs: `/scratch/mforooz/eicx/moesm2/{scores,verify,qc,logs}`, cache `/scratch/mforooz/eicx/moesm2/score_cache/pe/` (its `var/` is ours, its `truth/` links to Max's).

## 7. Blocked or not done

- Nothing blocked. The 48-experiment batch was not run (PI hold).
- The 25 entrants were not rescored with our code. We used Max's cached pe scores, which match MOESM2 (`repro_max.tsv`) and match our code on Avocado and Average to 7e-13 (`repro_ours.tsv`).
- The original truth and Max's binned cache live on Max's `/scratch` (purge-exposed). No durable copy exists under `/project/def-maxwl/mforooz`.
