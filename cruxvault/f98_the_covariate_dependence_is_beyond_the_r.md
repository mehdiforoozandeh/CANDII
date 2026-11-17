---
id: f98
type: finding
schema: 3
title: The covariate dependence is beyond the range on all three models and agrees…
parent: q23
confidence: moderate
why: The covariate dependence is beyond the range on all three models and agrees between the scrambled-training arm and the foreign-covariate scoring.
evidence: results/ablations_closeout/ABLATIONS_READ.md
created: "2026-11-16T22:22:47"
updated: "2026-11-16T22:22:47"
---

# f98 — The covariate dependence is beyond the range on all three models and agrees…

Parent:: [[q23_is_the_model_wired_as_designed_does_scra]]

## Answer

On all three models at 80,000 windows: scrambled covariates lower the correction-only term beyond the seed range (0.690 to 0.526, 0.659 to 0.462, 0.318 to 0.106) and leave imputation within it; adaLN-zero injection equals FiLM; dropping masked assays equals a learned mask token; concatenation injection is within the range or better; covariate tokens lower U-Net imputation to 0.383. Foreign covariates on the output side at scoring lower gap closed by 0.13 to 0.24 with depth kept, and foreign covariates on the input side change little.
