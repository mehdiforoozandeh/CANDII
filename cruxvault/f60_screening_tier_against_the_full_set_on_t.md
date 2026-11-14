---
id: f60
type: finding
schema: 3
title: Screening tier against the full set, on the four training-free baselines…
parent: q15
confidence: weak
evidence: results/t307/BASELINES.md, results/t307/baselines_headline.tsv
created: "2026-11-14T06:08:47"
updated: "2026-11-14T06:08:47"
---

# f60 — Screening tier against the full set, on the four training-free baselines…

Parent:: [[q15_are_256_evaluation_regions_enough]]

## Answer

Screening tier against the full set, on the four training-free baselines only: with pair jobs on 16 of the 256 regions, the pair records were byte-identical to the full records restricted to those regions; imputation and uncertainty metrics were identical (difference 0); every correction CRPS was 0.029 to 0.036 higher, a shift common to all four predictors and inside the full intervals; the largest difference was the position average's gap closed (3.35 against 2.86 [1.85, 6.59]). Two values fell outside the full interval: the depth-law error of the floor (0.171 against 0.099 [0.072, 0.135]) and the subset-pair gap closed of marginal NB (-6.35 against -8.72 [-174.8, -6.59]); neither enters model selection. The position average's composite was 0.939 on 16 regions against 0.817 on 256. Weak: baselines only; it does not replace the planned rung-4 check, which compares models. Caveats: dataset eic_v1; DNase-seq not scorable; H3K9ac flagged; signal-space metrics undefined for baselines; baselines only, no model.
