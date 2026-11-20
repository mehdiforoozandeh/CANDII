---
id: f79
type: finding
schema: 3
title: At rung 4 under the oracle scale, scrambling the covariates raised gap…
parent: h40
direction: mixed
confidence: weak
null: none
why: At rung 4 under the oracle scale, scrambling the covariates raised gap closed on all headline pairs and lowered it on the pairs with no duplicate-keeping side, so the correction part of the expectation holds only on the clean pairs and below 2 seed spreads; weak because it is a screening score and the clean subset has 18 pairs in 6 assays.
vs_expect: "Expected: scrambled covariates lower the correction term by more than 2 seed spreads and leave imputation within 2. Rung 4, 3 paired seeds, composite version 2: gap closed (oracle) on all 102 headline pairs +0.101 (larger seed spread 0.052), i.e. the scrambled arm corrects BETTER; on the 18 pairs with no duplicate-keeping side -0.099 (0.081), lower as expected but below 2 seed spreads (0.162). Correction CRPS -0.017 (0.011). Specific variance -0.029 (0.013), more than 2 seed spreads (0.027), against the expectation. Composite +0.017 (0.030). Rung 1 under the same scale: gap closed -0.021 (0.029), no-nodedup pairs -0.048 (0.046), specific variance -0.019 (0.003). The two gap-closed columns are macros over different assay sets (11 and 6), so they are not a strict decomposition."
verifiables: "Check 1 (correction term worse by more than 2 seed spreads): not met on either rung; at rung 4 the all-pairs term moves the other way. Check 2 (imputation within 2 seed spreads): not met at rungs 1 and 4."
evidence: results/metrics_screening/scores_rung4_arms.md, results/metrics_screening/scores_rung1_arms.md
created: "2026-11-15T00:23:58"
updated: "2026-11-15T00:23:58"
---

# f79 — At rung 4 under the oracle scale, scrambling the covariates raised gap…

Parent:: [[h40_scrambled_covariates_lowers_the_correcti]]
