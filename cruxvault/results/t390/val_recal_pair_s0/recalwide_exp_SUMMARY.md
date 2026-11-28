# EIC validation scores

Folder `/scratch/mforooz/eicx/val/rcp_combo_w2x_unet_s0_recalwide_exp`. Truth: `/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE/validation_data` (original 2019 processing). Cell = mean over bootstraps 0-9 (rank among MOESM6's 10 round-1 entries on that experiment; 1 = better than or equal to every entry). nmse = mse(all23) / var(truth). Combined = the organizers' per-experiment rank of the entry added alone to MOESM6's field, mean over bootstraps (min-max), over the eight measures other than msevar (its weights are not reproduced).

Leaderboard position among the 10 round-1 entries plus this one: 10 of 11 (rank_stat 10). 8 of 45 validation experiments scored; the 37 unscored count as the worst under the organizers' rule: C04M16 C09M20 C12M16 C12M32 C13M20 C17M04 C17M19 C17M29 C17M32 C18M21 C18M25 C23M03 C23M07 C23M26 C23M34 C25M21 C25M26 C27M03 C27M13 C27M24 C27M26 C29M29 C31M25 C32M08 C32M12 C32M20 C34M02 C34M32 C36M18 C37M29 C46M10 C46M18 C46M21 C46M35 C47M18 C48M16 C50M02.

| experiment | nmse | mse | gwcorr | gwspear | mseprom | msegene | mseenh | msevar | mse1obs | mse1imp | combined |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C02M22 | 18.62 | 16.07 (6) | 0.7443 (3) | 0.2978 (1) | 194.7 (6) | 28.06 (6) | 79.29 (6) | 3837 (-) | 1287 (6) | 1588 (6) | 4.5 (4-6) of 11 |
| C03M02 | 0.6349 | 17.13 (2) | 0.8236 (3) | 0.4676 (1) | 265.7 (2) | 21.12 (2) | 103.8 (2) | 3.178e+04 (-) | 1685 (2) | 1586 (3) | 2 (2-2) of 11 |
| C10M17 | 1.026 | 0.2473 (4) | 0.3889 (1) | 0.3896 (1) | 0.5548 (5) | 0.262 (4) | 0.5775 (5) | 18.99 (-) | 6.335 (2) | 6.486 (5) | 2 (2-2) of 11 |
| C16M17 | 1.584 | 0.4367 (7) | 0.3739 (2) | 0.3589 (1) | 1.594 (7) | 0.5345 (7) | 1.876 (7) | 5.926 (-) | 9.301 (2) | 21 (7) | 4.1 (4-5) of 11 |
| C20M22 | 0.3626 | 6.492 (1) | 0.9321 (2) | 0.2653 (1) | 73.78 (1) | 11.32 (1) | 35.94 (2) | 891.5 (-) | 592.1 (1) | 619.1 (1) | 1 (1-1) of 11 |
| C24M17 | 0.9355 | 4.79 (5) | 0.2983 (6) | 0.3356 (6) | 10.41 (10) | 5.97 (5) | 15.85 (10) | 943.7 (-) | 299.5 (1) | 102.2 (9) | 5 (5-5) of 11 |
| C24M25 | 0.482 | 11.37 (1) | 0.7276 (1) | 0.08799 (3) | 71.14 (1) | 20.03 (1) | 54.79 (1) | 5864 (-) | 770.4 (1) | 750.5 (3) | 1 (1-1) of 11 |
| C45M22 | 0.3038 | 4.457 (2) | 0.8809 (2) | 0.4318 (1) | 48.96 (2) | 7.708 (2) | 25.13 (2) | 710.9 (-) | 379.1 (2) | 416.3 (2) | 2 (2-2) of 11 |

Lambda sources: local_background, observed_control.
