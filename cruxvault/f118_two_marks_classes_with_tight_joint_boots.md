---
id: f118
type: finding
schema: 3
title: Two marks classes with tight joint-bootstrap intervals, but the values are…
parent: q18
confidence: moderate
why: Two marks classes with tight joint-bootstrap intervals, but the values are per assay class on log1p counts at a common depth, the read-sampling layer may carry a duplicate-removal bias, and DNase-seq has no usable value.
evidence: results/closeout_E4/ladder_two_sided_2026-11-18/COMPARE.md, results/closeout_E4/ladder_two_sided_2026-11-18/two_sided/ladder.tsv
created: "2026-11-18T16:22:08"
updated: "2026-11-18T16:22:08"
---

# f118 — Two marks classes with tight joint-bootstrap intervals, but the values are…

Parent:: [[q18_how_large_is_each_layer_of_the_variance_]]

## Answer

Two-sided replicate ladder (PI decision 5 of 2026-11-17; joint cluster bootstrap over cells; log1p counts at each assay's common depth; all strata). Increments [95% interval]: broad histone marks: read sampling 0.0463 [0.0413, 0.0512] (262 pairs, 27 cells), biological replicate 0.00053 [0.00025, 0.00082] (76, 17), cell type 0.00273 [0.0015, 0.00396] (355, 16); narrow histone marks: read sampling 0.0262 [0.0228, 0.0296] (391, 37), replicate 0.00058 [0.00029, 0.00087] (119, 24), cell type 0.00355 [0.00198, 0.00511] (445, 23). As shares of the cell-type pair value on the same cells and assays, all bands: broad sampling 0.951, replicate 0.013, cell type 0.063; narrow 0.875, 0.018, 0.120. By band: fine band broad 0.957 / 0.024 / 0.039, narrow 0.947 / 0.019 / 0.032; broad band broad 0.035 / 0.200 / 0.772, narrow 0.030 / 0.120 / 0.838; domain band broad 0.453 / 0.129 / 0.454, narrow 0.180 / 0.106 / 0.751; peak band the replicate increment is negative (broad -0.099, narrow -0.033 of the pair value). Sampling dominates at 25 bp and fine scale; cell type dominates at broad and domain scale. DNase-seq values are not interpretable (every DNase-seq pair is at 15,201 reads), and no other-experiment rung is estimable two-sided.
