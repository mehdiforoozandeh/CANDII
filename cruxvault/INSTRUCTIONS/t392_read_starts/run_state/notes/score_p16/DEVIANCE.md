# C14b: 16-region deviance metrics (PR 65 scorer), collected 2026-11-25 (UTC)

Source: `scores.json["deviance"][<key>]["macro"]` of `<arm>/seed{0,1,2}/p16_dev`. Read-start: `/project/def-maxwl/mforooz/t392/runs/rs_halves_mix_avg_pearson_unet` (jobs 5875186/87/88, kit RS7, reusing the kept p16_sv prediction stores). Reference: `/project/def-maxwl/mforooz/ONION/runs_rcp/rcp_halves_mix_avg_pearson_unet` (existing p16_dev, job 5381562 and siblings; not rerun). Full table: `deviance_p16.tsv`.

| model | imputation skill | imputation R2 | correction gap closed | calibration ratio | NB log-score gain vs marginal |
|---|---|---|---|---|---|
| read-count reference | 0.5426 (range 0.0653) | 0.3423 (0.0175) | 0.9508 (0.0023) | 1.0584 (0.0398) | 0.3747 (0.0105) |
| read-start model | 0.3746 (range 0.0870) | 0.2886 (0.0324) | 0.9452 (0.0044) | 1.3081 (0.0965) | 0.3064 (0.0335) |

Mean of 3 seeds; range = max - min over seeds. Calibration ratio 1 is ideal; above 1 means the model is overconfident. Imputation deviance floor: 0.5162 (reference) vs 0.5254 (read-start). The cause of the floor difference is not checked.

Spend at close of scoring: 836.2 CPU-h of 900 (574 before RS7 + 262.2 RS7 scoring by sacct CPUTimeRAW).
