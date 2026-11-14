---
id: f69
type: finding
schema: 3
title: The imputation margin is many seed spreads wide and the correction term…
parent: h39
direction: supports
confidence: moderate
null: none
why: The imputation margin is many seed spreads wide and the correction term closes part of the gap on the oracle scale of ruling 47; moderate because it is a screening score (pair tests on 16 regions) and the position average has one seed.
vs_expect: "Imputation term (specific variance recovered above noise): rung 1 base 0.3431 (seed spread 0.0029, 3 seeds) against the position average 0.1190, a margin of 0.224, about 77 seed spreads (expected more than 2). Correction: on the superseded half-read scale of composite version 1 rung 1 scored 5.34 (0.41), i.e. beyond the floor, not between depth-only and the floor; on the oracle scale (PI ruling 47) it scores 0.474 [0.408, 0.528] (seeds 0.484, 0.477, 0.461), i.e. it closes part of the gap. Correction CRPS 0.4971 (0.0034) against depth-only 0.6014 and the position average 0.5563. Calibration: pooled PIT deviation 0.0231 against 0.0499."
verifiables: "Check 1 (imputation term beats the position average by more than 2 seed spreads): met. Check 2 (correction between depth-only and the floor, above depth-only): met on the oracle scale (0.474 between 0 and 1); on the superseded half-read scale the model lies beyond the floor (5.34)."
evidence: results/metrics_screening/scores_ladder.md, results/metrics_screening/first_scores.md, results/oracle_floor/ORACLE_FLOOR.md
created: "2026-11-14T16:38:40"
updated: "2026-11-14T16:38:40"
---

# f69 — The imputation margin is many seed spreads wide and the correction term…

Parent:: [[h39_rung_1_base_beats_the_position_average_o]]
