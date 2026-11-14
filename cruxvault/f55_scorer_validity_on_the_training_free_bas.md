---
id: f55
type: finding
schema: 3
title: "Scorer validity on the training-free baselines: the scorer (version 1,…"
parent: q27
confidence: moderate
evidence: results/t307/BASELINES.md, results/t307/validity_checks.tsv
created: "2026-11-14T06:08:26"
updated: "2026-11-14T06:08:26"
---

# f55 — Scorer validity on the training-free baselines: the scorer (version 1,…

Parent:: [[q27_are_the_onion_run_s_estimators_scorers_a]]

## Answer

Scorer validity on the training-free baselines: the scorer (version 1, legacy path) gave the exact reference values on both tiers (256 regions and the 16-region screening set). Depth-only scaling gave gap closed 0 and the half-against-half floor gave 1 on all 17 values (macro, 11 assays, 3 knobs, both interval ends; max abs deviation 0.0); the marginal NB uncertainty term was 0; subset pairs gave 0 and 1. A second computation from the per-pair records reproduced the correction CRPS of depth-only scaling (0.5723844358), the floor (0.5326421965) and the position average (0.5234223086) exactly, and the pooled PIT deviation per assay from suffstats.npz with largest difference 0. Caveats: dataset eic_v1; DNase-seq not scorable (1 validation track with halves, the rule needs 3); H3K9ac flagged (no signal-space score, kept in counts space; imputation macros also given without it); signal-space (signal_arcsinh) metrics undefined for every baseline (covariates lack frag_len); no model involved. Moderate, not strong: the checks are exact, but this question covers the whole scoring and training pipeline and more work on it is planned.
