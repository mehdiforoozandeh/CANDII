---
id: f72
type: finding
schema: 3
title: The scrambled-covariates part misses both of its thresholds at rungs 1 and…
parent: h40
direction: mixed
confidence: weak
null: none
why: The scrambled-covariates part misses both of its thresholds at rungs 1 and 4, while the oracle-scale correction ordering points the expected way on the pairs with no duplicate-keeping side; weak because it is a screening score on 16 pair regions, the composite term used (version 1) was clipped to 1 for every run, and no difference clears 2 seed spreads.
vs_expect: "Expected: scrambled covariates lower the correction term by more than 2 seed spreads and leave imputation within 2. Correction CRPS (paired, 3 seeds): rung 4 -0.017 (spread 0.011), rung 1 +0.005 (0.011), so scrambling does not hurt correction CRPS (at rung 4 it helps). Oracle scale at rung 1 (ruling 47): base 0.484, 0.477, 0.461 against scrambled 0.444, 0.443, 0.472 on all 102 headline pairs (base ahead in 2 of 3 seeds; means 0.474 and 0.453, seed spreads 0.023 and 0.029); on the 18 pairs with no duplicate-keeping side base 0.591, 0.610, 0.579 against 0.535, 0.527, 0.573 (ahead in 3 of 3), a mean gap of about 0.05 against 2 seed spreads of about 0.09. Imputation (specific variance above noise): scrambling costs 0.029 (0.017) at rung 4 and 0.019 (0.006) at rung 1, i.e. more than 2 seed spreads (rung 1 seed spreads 0.0029 and 0.0031), against the expectation that imputation stays within 2. What would settle it: the scrambled arms of rungs 1 and 4 scored on all 256 pair regions with the oracle-floor correction term (composite version 2), read on the pairs with no duplicate-keeping side, or on fragment-disjoint halves."
verifiables: "Check 1 (correction term worse by more than 2 seed spreads): not met on correction CRPS; on the oracle scale the direction is as expected but below 2 seed spreads. Check 2 (imputation within 2 seed spreads): not met; scrambling lowers imputation beyond 2 seed spreads at rungs 1 and 4."
evidence: results/metrics_screening/scores_rung1_arms.md, results/metrics_screening/scores_ladder.md, results/oracle_floor/ORACLE_FLOOR.md
created: "2026-11-14T16:39:15"
updated: "2026-11-14T16:39:15"
---

# f72 — The scrambled-covariates part misses both of its thresholds at rungs 1 and…

Parent:: [[h40_scrambled_covariates_lowers_the_correcti]]
