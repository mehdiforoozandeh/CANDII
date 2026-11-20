---
id: f123
type: finding
schema: 3
title: Read from stored per-pair records with an independent reimplementation that…
parent: q32
confidence: moderate
why: Read from stored per-pair records with an independent reimplementation that agreed to 1e-10, but the split rests on 102 pairs and the arm-level effect of denoising cannot be separated from correction with the stored files.
evidence: results/correction_metric_review/REVIEW.md, results/gap_closed_floor/GAP_FLOOR.md
created: "2026-11-18T16:23:00"
updated: "2026-11-18T16:23:00"
---

# f123 — Read from stored per-pair records with an independent reimplementation that…

Parent:: [[q32_which_reference_should_define_the_1_of_t]]

## Answer

What the oracle reference removes and what it keeps. Gap closed = (CRPS of depth-only scaling d - CRPS of the method) / (d - O), with O = half the mean absolute difference of the target's two read halves. O removes the target's own read-sampling noise from the top of the scale, but the depth-only forecast is built from the source half, so the denominator d - O still holds the source half's sampling noise: it splits into the covariate cost of a raw copy, d - f = 0.0420, and the sampling noise of using one half as the mean, f - O = 0.1835, over the 102 headline pairs (f = the target's own other half), so the covariate part is 18.6% of the denominator (15.1% as a macro over assays) and about 81 to 85% of the scale measures denoising of the source half. On the 18 crop and crop + single-end pairs with no duplicate-keeping side, the covariate share is 6.5% and 13.4%. PI ruling of 2026-11-16 (composite version 3): the composite's correction term is gap closed on these 18 crop pairs, and a correction-only term (same-view score as the reference) is reported beside it from the final round on.
