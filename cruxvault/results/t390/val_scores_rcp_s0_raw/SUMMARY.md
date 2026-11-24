# EIC validation scores

Folder `/scratch/mforooz/eicx/val/rcp_combo_w2x_unet_s0_raw`. Truth: `/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/validation_data` (original 2019 processing). Cell = mean over bootstraps 0-9 (rank among MOESM6's 10 round-1 entries on that experiment; 1 = better than or equal to every entry). nmse = mse(all23) / var(truth). Combined = the organizers' per-experiment rank of the entry added alone to MOESM6's field, mean over bootstraps (min-max), over the eight measures other than msevar (its weights are not reproduced).

No leaderboard position: 7 of 45 validation experiments scored (the organizers' statistic counts a missing experiment as the worst).

| experiment | nmse | mse | gwcorr | gwspear | mseprom | msegene | mseenh | msevar | mse1obs | mse1imp | combined |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C02M22 | 8.676 | 7.486 (6) | 0.7527 (3) | 0.3167 (1) | 91.72 (6) | 13.11 (6) | 35.66 (6) | 1844 (-) | 611.3 (6) | 741.2 (6) | 4.3 (4-5) of 11 |
| C03M02 | 0.643 | 17.35 (2) | 0.8312 (3) | 0.4039 (5) | 268.2 (2) | 21.39 (2) | 105 (2) | 3.195e+04 (-) | 1708 (2) | 1603 (3) | 2.1 (2-3) of 11 |
| C10M17 | 0.9205 | 0.2218 (1) | 0.3666 (2) | 0.3712 (1) | 0.3597 (4) | 0.2194 (1) | 0.3966 (4) | 19.7 (-) | 7.605 (3) | 3.009 (4) | 1 (1-1) of 11 |
| C16M17 | 1.087 | 0.2996 (5) | 0.3462 (2) | 0.4657 (1) | 0.7955 (6) | 0.3469 (5) | 0.9906 (6) | 3.277 (-) | 10.35 (2) | 9.277 (6) | 2 (2-2) of 11 |
| C20M22 | 0.1504 | 2.692 (1) | 0.9336 (2) | 0.2582 (1) | 27.94 (1) | 4.581 (1) | 15.43 (1) | 381.1 (-) | 245 (1) | 240.1 (1) | 1 (1-1) of 11 |
| C24M17 | 0.9676 | 4.955 (5) | 0.2593 (7) | 0.3513 (3) | 8.956 (6) | 6.023 (5) | 14.68 (6) | 1002 (-) | 335.2 (5) | 84.11 (8) | 5 (5-5) of 11 |
| C45M22 | 0.2193 | 3.218 (1) | 0.8847 (2) | 0.4282 (1) | 34.94 (1) | 5.505 (1) | 15.14 (1) | 1037 (-) | 293.9 (1) | 292.2 (1) | 1 (1-1) of 11 |

Lambda sources: local_background, observed_control.
