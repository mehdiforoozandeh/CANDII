---
id: f57
type: finding
schema: 3
title: Covariate correction (headline metric 4 of the evaluation protocol, gap…
parent: q19
confidence: moderate
evidence: results/t307/BASELINES.md, results/t307/baselines_headline.tsv, results/t307/correction_headroom.tsv
created: "2026-11-14T06:08:37"
updated: "2026-11-14T06:08:37"
---

# f57 — Covariate correction (headline metric 4 of the evaluation protocol, gap…

Parent:: [[q19_what_do_the_training_free_baselines_reac]]

## Answer

Covariate correction (headline metric 4 of the evaluation protocol, gap closed against depth-only scaling on never-trained pairs): the position average, which ignores the source track, went below the half-against-half floor and closed 2.86 [1.85, 6.59] of the depth-only-to-floor gap (macro correction CRPS 0.523 = 0.513 oracle_scaled + 0.010 scale_error, against depth-only scaling 0.572 = 0.561 + 0.011 and the floor 0.533 = 0.523 + 0.010; 11 assays). On the 102 never-trained non-subset pairs depth-only scaling is 0.042 [0.019, 0.081] CRPS per bin above the floor (pair means 0.561 against 0.519). The position average beat the floor on 7 of the 10 knobs with at least 3 cells. So the floor is not a lower bound on correction CRPS: its mean is one pseudoreplicate half and carries that half's sampling noise, and the gap-closed scale of metric 4 can exceed 1 for a smooth predictor. Which reference should define the 1 of that scale is open and with the PI. Caveats: dataset eic_v1; DNase-seq not scorable (1 validation track with halves, the rule needs 3); H3K9ac flagged (no signal-space score, kept in counts space; imputation macros also given without it); signal-space (signal_arcsinh) metrics undefined for every baseline (covariates lack frag_len); no model involved.
