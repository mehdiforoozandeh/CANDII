# C13b: 16-region headline scoring, collected 2026-11-25 (UTC)

Source: `select_rule.py arm-summary --runs /project/def-maxwl/mforooz/t392/runs_both --sub p16_sv`, run on Fir from kit RS7. `runs_both` holds two symlinks: the read-start arm and `/project/def-maxwl/mforooz/ONION/runs_rcp/rcp_halves_mix_avg_pearson_unet`. Full output: `arm_summary_p16_sv.tsv`.

| model | imputation_raw | correction_v3_raw | uncertainty_raw (calibration) |
|---|---|---|---|
| read-count reference (rcp_halves_mix_avg_pearson_unet) | 0.4996 (range 0.0349) | 0.8156 (range 0.0105) | 0.6852 (range 0.0924) |
| read-start model (rs_halves_mix_avg_pearson_unet) | 0.4568 (range 0.0070) | 0.7826 (range 0.0362) | 0.6064 (range 0.0679) |

Mean of 3 seeds; range = max - min over seeds.

## Stop rules
- (a) The reference row reproduces 0.4996, 0.8156, 0.6852 to 4 decimals. Pass.
- (b) Open-edge share, shards 1-3 (shard 0 holds the copy-leak guardrail windows): n_open_edges / n_windows = 0.25000 for every seed (s0 70,528 / 282,112; s1 121,280 / 485,120; s2 120,256 / 481,024). Share of bins within 8 bins of an open edge = 0.25 x 8 / 2,048 = 0.195%, below the 0.25% limit. Pass.
- (c) Spend: 574 CPU-h before the RS7 scoring (plan Log) + 247.6 CPU-h for RS7 scoring jobs 5874564-5875188 by sacct CPUTimeRAW at about 03:30 UTC (deviance s1, s2 still running) = 821.6 CPU-h of 900. `check.py spend` returns a lower bound only (55 rows lack cputime_raw_s).

## Sanity against read-start truth (R12; not a scorer metric)
`sanity.tsv`: mean NB deviance per bin 0.527-0.543, Pearson r of log1p mean against log1p truth 0.445-0.457, on 63 targets over 16 regions.
