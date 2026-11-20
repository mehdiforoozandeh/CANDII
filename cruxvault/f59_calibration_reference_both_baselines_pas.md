---
id: f59
type: finding
schema: 3
title: "Calibration reference: both baselines passed the pseudoreplicate lower…"
parent: q19
confidence: moderate
evidence: results/t307/BASELINES.md, results/t307/calibration_baselines.tsv, results/t307/baselines_headline.tsv
created: "2026-11-14T06:08:37"
updated: "2026-11-14T06:08:37"
---

# f59 — Calibration reference: both baselines passed the pseudoreplicate lower…

Parent:: [[q19_what_do_the_training_free_baselines_reac]]

## Answer

Calibration reference: both baselines passed the pseudoreplicate lower variance bound (predicted over replicate variance >= 1) in all 9 scorable assays (position average 1.21 to 16.97, macro 4.78 [3.43, 6.67]; marginal NB 1.40 to 3.99, macro 2.54 [1.88, 3.32]). Pooled non-randomized PIT deviation: position average 0.050 [0.037, 0.085] against marginal NB 0.060 [0.053, 0.127]; worst stratum 0.074 against 0.185. The upper variance bound (other_experiment rung) is not populated: eic_v1 has no second experiment of a validation cell. Caveats: dataset eic_v1; DNase-seq not scorable (1 validation track with halves, the rule needs 3); H3K9ac flagged (no signal-space score, kept in counts space; imputation macros also given without it); signal-space (signal_arcsinh) metrics undefined for every baseline (covariates lack frag_len); no model involved.
