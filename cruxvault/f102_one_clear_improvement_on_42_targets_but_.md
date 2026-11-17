---
id: f102
type: finding
schema: 3
title: One clear improvement on 42 targets, but the variant was fitted on the same…
parent: q57
confidence: moderate
why: One clear improvement on 42 targets, but the variant was fitted on the same targets and edge bins are still biased
evidence: results/closeout_E2/FORMULA_EXTENSION.md
created: "2026-11-17T05:00:29"
updated: "2026-11-17T05:00:29"
---

# f102 — One clear improvement on 42 targets, but the variant was fitted on the same…

Parent:: [[q57_how_closely_can_a_formula_on_25_bp_binne]]

## Answer

Closer than the in-bin form, but not exactly. On observed counts of 41 ChIP-seq validation targets, spreading each count over a box of the fragment length raised the median Pearson with the pipeline track from 0.831 to 0.921 and peak-tile F1 from 0.573 to 0.782 (precision 0.421 to 0.853, recall 0.957 to 0.758); DNase-seq Pearson rose from 0.943 to 0.994 (1 target). With the pipeline's own fragment length the Pearson was 0.944. An exact match is impossible, because binning loses strand and position inside the bin. The box variant was chosen among 4 on these targets.
