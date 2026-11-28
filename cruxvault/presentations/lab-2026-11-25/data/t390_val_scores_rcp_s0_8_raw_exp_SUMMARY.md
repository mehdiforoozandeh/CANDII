# EIC validation scores

Folder `/scratch/mforooz/eicx/val/rcp_combo_w2x_unet_s0_raw_exp`. Truth: `/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/validation_data` (original 2019 processing). Cell = mean over bootstraps 0-9 (rank among MOESM6's 10 round-1 entries on that experiment; 1 = better than or equal to every entry). nmse = mse(all23) / var(truth). Combined = the organizers' per-experiment rank of the entry added alone to MOESM6's field, mean over bootstraps (min-max), over the eight measures other than msevar (its weights are not reproduced).

No leaderboard position: 8 of 45 validation experiments scored (the organizers' statistic counts a missing experiment as the worst).

| experiment | nmse | mse | gwcorr | gwspear | mseprom | msegene | mseenh | msevar | mse1obs | mse1imp | combined |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C02M22 | 11.02 | 9.506 (6) | 0.7504 (3) | 0.3201 (1) | 115.9 (6) | 16.62 (6) | 46.09 (6) | 2299 (-) | 766.9 (6) | 939.9 (6) | 4.3 (4-5) of 11 |
| C03M02 | 0.5988 | 16.16 (2) | 0.8201 (3) | 0.458 (2) | 249.9 (2) | 19.95 (2) | 98.84 (2) | 3.042e+04 (-) | 1581 (2) | 1490 (3) | 2 (2-2) of 11 |
| C10M17 | 0.9337 | 0.225 (1) | 0.3963 (1) | 0.3841 (1) | 0.4385 (4) | 0.2334 (4) | 0.4656 (4) | 19.27 (-) | 6.435 (2) | 4.703 (4) | 1 (1-1) of 11 |
| C16M17 | 1.285 | 0.3541 (6) | 0.3796 (2) | 0.3817 (1) | 1.149 (7) | 0.425 (7) | 1.366 (7) | 4.35 (-) | 9.019 (2) | 14.55 (7) | 4 (4-4) of 11 |
| C20M22 | 0.1788 | 3.201 (1) | 0.9339 (2) | 0.264 (1) | 34.48 (1) | 5.508 (1) | 18.19 (1) | 421.1 (-) | 290.1 (1) | 292.4 (1) | 1 (1-1) of 11 |
| C24M17 | 0.931 | 4.768 (5) | 0.2881 (6) | 0.3435 (3) | 9.166 (6) | 5.858 (5) | 14.67 (6) | 975.9 (-) | 316.7 (5) | 85.21 (8) | 5 (5-5) of 11 |
| C24M25 | 0.4865 | 11.48 (1) | 0.7264 (1) | 0.08387 (3) | 71.46 (1) | 20.19 (1) | 55.41 (1) | 5788 (-) | 762.2 (1) | 753 (3) | 1 (1-1) of 11 |
| C45M22 | 0.2207 | 3.239 (1) | 0.8839 (2) | 0.4394 (1) | 35.11 (1) | 5.56 (1) | 16.07 (1) | 895.7 (-) | 287.5 (1) | 295.3 (1) | 1 (1-1) of 11 |

Lambda sources: local_background, observed_control.
