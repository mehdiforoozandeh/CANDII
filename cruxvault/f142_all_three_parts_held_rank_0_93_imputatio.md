---
id: f142
type: finding
schema: 3
title: All three parts held (rank 0.93 imputation, 0.78 correction; between-arm…
parent: h48
direction: supports
confidence: moderate
null: none
why: All three parts held (rank 0.93 imputation, 0.78 correction; between-arm share above the current term for every goal; C1-C5 pass), but only 10 arms x 2 seeds and the correction rank clears 0.7 by a modest margin.
vs_expect: "imputation Spearman 0.93 and correction 0.78 (>= 0.7); between-arm share 0.945 vs 0.921, 0.993 vs 0.950, 0.965 vs 0.947; sim C1-C5 pass. Calibration ranks disagree (0.19), outside the expectation: ratio and PIT measure spread vs shape."
evidence: results/t371/compare/rcp/rank_agreement.tsv, results/t371/compare/rcp/seed_stability.tsv, results/t371/sim/checks.json
created: "2026-11-23T00:48:51"
updated: "2026-11-23T00:48:51"
---

# f142 — All three parts held (rank 0.93 imputation, 0.78 correction; between-arm…

Parent:: [[h48_on_the_same_predictions_each_deviance_te]]
