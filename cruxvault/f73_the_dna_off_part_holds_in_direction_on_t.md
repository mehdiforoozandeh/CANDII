---
id: f73
type: finding
schema: 3
title: The DNA-off part holds in direction on the composite's imputation term,…
parent: h40
direction: supports
confidence: weak
null: none
why: The DNA-off part holds in direction on the composite's imputation term, consistently in both seeds, but the effect is about one seed spread, rests on 2 seeds (seed 2 has no score), and count CRPS moves the other way.
vs_expect: "Expected: dna_off lowers imputation. Rung 1, paired seeds 0 and 1: dna_off minus base on specific variance above noise -0.0033 (spread of the difference 0.0007; per seed -0.0036, -0.0030), against base seed spread 0.0029, i.e. lower, by about one seed spread. Count CRPS improved by 0.0030 with the DNA off, and the composite rose by 0.020 and the pooled PIT deviation fell by 0.0052, so the DNA input at rung 1 shows no benefit on any other imputation or calibration metric. Screening score."
verifiables: "Check 3 (dna_off minus base on the imputation term is worse): met, by about one seed spread, 2 seeds."
evidence: results/metrics_screening/scores_rung1_arms.md
created: "2026-11-14T16:39:15"
updated: "2026-11-14T16:39:15"
---

# f73 — The DNA-off part holds in direction on the composite's imputation term,…

Parent:: [[h40_scrambled_covariates_lowers_the_correcti]]
