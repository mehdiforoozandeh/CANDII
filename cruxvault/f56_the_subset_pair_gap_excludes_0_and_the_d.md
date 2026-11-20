---
id: f56
type: finding
schema: 3
title: The subset-pair gap excludes 0 and the depth knobs are not the largest…
parent: h38
direction: against
confidence: moderate
null: none
why: The subset-pair gap excludes 0 and the depth knobs are not the largest pooled, so both verifiables fail; moderate because the run-type part is untestable (1 cell) and the depth part holds on depth-increasing pairs.
vs_expect: "Expected the largest gaps on the depth and run-type knobs and near 0 on pure subset pairs. Pure subset (depth-thinning) pairs: 0.035 [0.021, 0.055], not near 0; it mixes -0.011 [-0.016, -0.007] on depth-decreasing and 0.081 [0.049, 0.125] on depth-increasing pairs, so the sign of the gap follows the direction of the depth change. Pooled, crop+mapq (0.116 [0.064, 0.170]) and depth+nodedup (0.107 [0.061, 0.164]) lead with overlapping intervals; split by direction, the depth-increasing pairs of the three compound depth knobs have the largest gaps (0.182 to 0.234) and every depth-decreasing set is negative. Every knob with se (run type) has 2 to 8 pairs from 1 cell, so the run-type part is untested. Caveats: dataset eic_v1; DNase-seq not scorable (1 validation track with halves, the rule needs 3); H3K9ac flagged (no signal-space score, kept in counts space; imputation macros also given without it); signal-space (signal_arcsinh) metrics undefined for every baseline (covariates lack frag_len); no model involved."
verifiables: "Depth and run-type knobs have the two largest gaps: fail (pooled), run type untestable. Subset-pair gap within its interval of 0: fail (0.035 [0.021, 0.055])."
evidence: results/t307/BASELINES.md, results/t307/correction_headroom.tsv
created: "2026-11-14T06:08:26"
updated: "2026-11-14T06:08:26"
---

# f56 — The subset-pair gap excludes 0 and the depth knobs are not the largest…

Parent:: [[h38_the_gap_between_depth_only_scaling_and_t]]
