---
id: f105
type: finding
schema: 3
title: On all 256 regions scrambled covariates lower the correction-only term on…
parent: h40
direction: supports
confidence: moderate
why: On all 256 regions scrambled covariates lower the correction-only term on all three models (twice the seed range on the two conv models, 1.9 times on the U-Net) and leave imputation within the range; the dna_off part was not rescored, and the U-Net near miss lowers the confidence one step.
vs_expect: "correction-only term: conv autoencoder -0.195 (bar 0.046), conv + attention -0.230 (bar 0.081), U-Net -0.242 (bar 0.129); imputation within on all three; pre-written rule 2.1 rates the use of covariates for same-view correction settled, strong"
evidence: tasks/t349_onion_first_read_of_the_256_region_.md
created: "2026-11-18T09:48:31"
updated: "2026-11-18T09:48:31"
---

# f105 — On all 256 regions scrambled covariates lower the correction-only term on…

Parent:: [[h40_scrambled_covariates_lowers_the_correcti]]
