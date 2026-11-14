---
id: f58
type: finding
schema: 3
title: "Imputation reference: the position average recovered 0.119 [-0.156, 0.194]…"
parent: q19
confidence: moderate
evidence: results/t307/BASELINES.md, results/t307/baselines_headline.tsv, results/t307/noise_floor_by_assay.tsv
created: "2026-11-14T06:08:37"
updated: "2026-11-14T06:08:37"
---

# f58 — Imputation reference: the position average recovered 0.119 [-0.156, 0.194]…

Parent:: [[q19_what_do_the_training_free_baselines_reac]]

## Answer

Imputation reference: the position average recovered 0.119 [-0.156, 0.194] of the cell-type-specific variance above the noise floor in counts_log1p (0.119 without H3K9ac); its count CRPS was 0.795 = 0.775 oracle_scaled + 0.019 scale_error. Marginal NB's count CRPS skill against the position average was -0.216 [-0.309, -0.129] (count CRPS 1.011 = 0.845 + 0.166). In the 9 scorable assays the noise floor was 58% to 85% of the specific variance at 25 bp. Caveats: dataset eic_v1; DNase-seq not scorable (1 validation track with halves, the rule needs 3); H3K9ac flagged (no signal-space score, kept in counts space; imputation macros also given without it); signal-space (signal_arcsinh) metrics undefined for every baseline (covariates lack frag_len); no model involved.
