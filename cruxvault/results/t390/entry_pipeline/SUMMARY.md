# EIC entry scores

Folder `/scratch/mforooz/eicx/entry/test`. Cell = value (rank among the published entrants on that experiment, chromosome set all23; 1 = better than or equal to every entrant). Combined = the organizers' per-experiment rank of the entry added alone to the published field, mean over bootstraps 0-9 (min-max).

## MOESM2 (raw -log10 p vs the original 2019 truth; headline)

| experiment | mse | gwcorr | gwspear | mseprom | msegene | mseenh | msevar | mse1obs | mse1imp | combined |
|---|---|---|---|---|---|---|---|---|---|---|
| C12M02 | 552.5 (21) | 0.5851 (20) | 0.593 (1) | 9979 (22) | 694.8 (21) | 3421 (18) | 2.181e+07 (20) | 5.52e+04 (21) | 5.514e+04 (25) | 17.1 (17-18) of 26 |
| C19M22 | 893.9 (19) | 0.8185 (18) | 0.3622 (1) | 1.08e+04 (20) | 1670 (19) | 2264 (12) | 4.89e+05 (22) | 8.938e+04 (19) | 8.934e+04 (21) | 15 (15-15) of 26 |
| C40M17 | 0.5061 (1) | 0.4282 (1) | 0.4221 (1) | 1.613 (1) | 0.5685 (1) | 0.7466 (1) | 426.8 (1) | 26.39 (1) | 13.7 (6) | 1 (1-1) of 26 |

## MOESM4 (QN with own peaks vs the qnorm truth)

| experiment | mse | gwcorr | gwspear | mseprom | msegene | mseenh | msevar | mse1obs | mse1imp | combined |
|---|---|---|---|---|---|---|---|---|---|---|
| C12M02 | 33.92 (1) | 0.9652 (1) | 0.903 (1) | 304.4 (1) | 42.26 (1) | 296.4 (1) | 2.847e+04 (1) | 3042 (1) | 3127 (2) | 1 (1-1) of 26 |
| C19M22 | 3.594 (1) | 0.9623 (1) | 0.7588 (1) | 41.73 (1) | 6.334 (1) | 17.06 (1) | 657.6 (1) | 351.9 (1) | 355.4 (1) | 1 (1-1) of 26 |
| C40M17 | 0.1578 (1) | 0.7939 (1) | 0.7769 (1) | 0.4321 (1) | 0.1735 (1) | 0.4089 (1) | 4.086 (1) | 4.757 (1) | 3.959 (2) | 1 (1-1) of 26 |

Lambda sources: local_background, observed_control. QN peaks: rung L7 threshold caller on the entry itself (the half-read overlap rule needs reads and is not available for a mean prediction).
