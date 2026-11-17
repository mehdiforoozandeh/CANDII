---
id: f90
type: finding
schema: 3
title: Each claim is a difference of means larger than the larger seed range;…
parent: q48
confidence: moderate
why: Each claim is a difference of means larger than the larger seed range; single-model effects are not generalized.
evidence: results/ablations_closeout/ABLATIONS_READ.md
created: "2026-11-16T22:22:45"
updated: "2026-11-16T22:22:45"
---

# f90 — Each claim is a difference of means larger than the larger seed range;…

Parent:: [[q48_which_minicandi_design_choices_move_impu]]

## Answer

Of 16 ablations (3 seeds, 80,000 windows, 16-region screening set), most arms stay within the seed range on the composite. Beyond the range: training views and true covariates are needed for the correction-only term on all three models; native two-step routing lowers crop-pair correction on both conv models; local-ratio and quantile count transforms lower imputation on all three; a fixed input depth trades U-Net imputation (-0.028) for correction (+0.168). Calibration ranges of 0.11 to 0.18 hide smaller calibration effects.
