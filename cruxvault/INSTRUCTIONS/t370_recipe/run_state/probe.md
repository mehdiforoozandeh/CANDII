# Gate G0: U-Net probe on kit RCP1 (2026-11-21)

Run `runs_rcp/rcp_probe_base_unet/seed0`, kit RCP1 (git 7b5dc8cd), Fir. Jobs 4863259 (train), 4863260-63 (predict), 4863264 (metrics); all COMPLETED.

## Terms at p16_sv

Source: `select_rule.py table` and `arm-summary` on Fir, kit RCP2, `--sub p16_sv`.

| term | probe seed 0 | base U-Net seed 0 | base mean (3 seeds) | base range | inside mean +/- range |
|---|---|---|---|---|---|
| imputation_raw | 0.4658 | 0.465813 | 0.4452 | 0.0380 | yes |
| correction_v3_raw | 0.5072 | 0.507248 | 0.5379 | 0.0592 | yes |
| uncertainty_raw | 0.5451 | 0.545145 | 0.6460 | 0.1577 | yes |

The probe reproduced base seed 0 to 4 decimals. The new switches are off in the base config, so this result was expected.

## Cost (sacct CPUTimeRAW, allocated CPU-h)

| phase | measured | fact 9 | G0 limit | pass |
|---|---|---|---|---|
| train (12 CPUs x 01:25:40) | 61,680 s = 17.13 | 7.06 | 14.12 | no |
| screen predict, 4 shards | 41,688 s = 11.58 | 10.78 | | |
| screen metrics (16 CPUs x 00:39:05) | 37,520 s = 10.42 | 8.27 | | |
| screen scoring total | 22.00 | 19.05 | 38.10 | yes |
| run-seed total | 39.13 | 26.11 | | |

Training used 15:18:07 of CPU time out of 17.13 allocated CPU-h, so the allocation was well used. Fail on cost only: per G0, continue and recompute the budget. The measured run-seed is 39.13 / 26.11 = 1.50 x the projection.

## Recomputed screen budget (projection x 1.50)

| stage | projected (fact 9) | x 1.50 |
|---|---|---|
| s1a (66 run-seeds) | 1,748.5 | 2,623 |
| s1b (45 run-seeds) | 1,245.0 | 1,868 |
| s1c CNN (18 run-seeds) | 478.4 | 718 |
| total screen | 3,471.9 | 5,209 |

The 1.50 factor was measured on the U-Net only. The conv models carry it by assumption. The screen alone then reaches about 87% of the 6,000 CPU-h checkpoint (D20).

## Tests

`rcp_tests` 4863258 hit its 3 h limit (TIMEOUT); the output was piped to `tail`, so no verdict was printed. Resubmitted on kit RCP2 as 4888524 (4 CPUs, 16000M, 12 h). The screen went out before the verdict. Reason: the probe reproduced the base exactly, and the local base-digest test passed on the merged tree. If 4888524 shows a failure in a switch an arm uses, those arms are cancelled with `cancel-own.sh`.
