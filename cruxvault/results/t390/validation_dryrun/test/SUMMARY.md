# EIC validation scores

Folder `/scratch/mforooz/eicx/val/test`. Truth: `/scratch/maxwl/epi_imputation/data/synapse/validation_data` (original 2019 processing). Cell = mean over bootstraps 0-9 (rank among MOESM6's 10 round-1 entries on that experiment; 1 = better than or equal to every entry). nmse = mse(all23) / var(truth). Combined = the organizers' per-experiment rank of the entry added alone to MOESM6's field, mean over bootstraps (min-max), over the eight measures other than msevar (its weights are not reproduced).

No leaderboard position: 3 of 45 validation experiments scored (the organizers' statistic counts a missing experiment as the worst).

| experiment | nmse | mse | gwcorr | gwspear | mseprom | msegene | mseenh | msevar | mse1obs | mse1imp | combined |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C03M02 | 0.341 | 9.202 (1) | 0.9249 (1) | 0.8969 (1) | 162.9 (2) | 11.16 (1) | 45.77 (1) | 2.322e+04 (-) | 916.4 (1) | 903.4 (1) | 1 (1-1) of 11 |
| C12M32 | 0.4866 | 0.4451 (1) | 0.88 (1) | 0.688 (1) | 0.8991 (1) | 0.7429 (1) | 1.209 (1) | 22.3 (-) | 20.12 (1) | 27.52 (10) | 1 (1-1) of 11 |
| C17M04 | 0.7981 | 0.2337 (2) | 0.7485 (1) | 0.5587 (1) | 0.4885 (2) | 0.2749 (2) | 0.7785 (2) | nan (-) | 4.402 (1) | 8.756 (6) | 2 (2-2) of 11 |

Lambda sources: local_background, observed_control.
