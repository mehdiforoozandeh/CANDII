# Validation boards side by side: raw, slope calibration (two fits), read-depth cap

Same 30 validation experiments on all four boards (`SIDE_BY_SIDE.tsv`, built by `side_by_side.py`). Prediction: miniCANDI seed 0, scored on the mean. Truth: original 2019 processing (MOESM6 field, 10 round-1 entries). Unscored experiments (15) count as the worst, so positions compare an entry with 30 of 45 experiments.

nMSE = squared error divided by the variance of the truth; 1 = predicting the mean. Combined rank = the organizers' per-experiment rank among the 11 (eight measures, msevar excluded).

| board | method | position of 11 | mean combined rank | median nMSE | nMSE > 1 | rank 1-3 | rank 10-11 |
|---|---|---|---|---|---|---|---|
| raw | predicted mean as is | 5 | 5.31 | 1.55 | 20/30 | 11 | 5 |
| slope, in-sample | mu' = max(m + b(mu - m), 0); b per assay from all study items of the assay, the track's own experiment included. Preview of the blind-entry gain, not a validation number | 5 | 4.56 | 0.998 | 14/30 | 14 | 3 |
| slope, leave one experiment out | same, b from the other experiments of the assay only; 8 of 30 tracks have none and keep b = 1 | 5 | 4.97 | 1.09 | 18/30 | 12 | 4 |
| read-depth cap | ChIP tracks with depth_reads > 20 M: mu and depth_reads times 20 M / depth_reads; cap fitted on 196 training ChIP tracks; DNase unchanged | **4** | **3.70** | **0.917** | **10/30** | **17** | **2** |

Zero clip: the slope boards clip mu' at 0, because b > 1 can make it negative (9 in-sample and 6 leave-one-out tracks first failed the conversion). The raw and depth-cap boards need no clip.

Against raw, per experiment (change in combined rank above 0.05): in-sample better 10, worse 7, mean -0.75; leave-one-out better 6, worse 4, mean -0.35; depth cap better 15, worse 3, mean -1.61.

Reading: the depth cap is the only fix that moves the position, and it is fitted without validation truth. The fair slope fit (leave one experiment out) gains little, because the per-experiment slope varies (b 0.30-1.92) and is not predicted by the assay.

Sources: Fir `/scratch/mforooz/eicx/val/{rcp_combo_w2x_unet_s0_raw,rcp_s0_slope_insample,rcp_s0_slope_loeo,rcp_s0_depthcap}/` (board_validation.tsv, SUMMARY.md copied here). The raw folder now holds 34 scored experiments; this table uses the 30 common ones.
