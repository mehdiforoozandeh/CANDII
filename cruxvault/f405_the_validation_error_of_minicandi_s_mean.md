---
id: f405
type: finding
schema: 3
title: The validation error of miniCANDI's mean predictions is a scale error, not a…
parent: q90
confidence: moderate
evidence: results/t390/val_scale_diag/SCALE_DIAG.md
created: "2026-11-24T19:46:57"
updated: "2026-11-24T19:46:57"
---

# f405 — The validation error of miniCANDI's mean predictions is a scale error, not a…

Parent:: [[q90_can_a_count_space_model_like_minicandi_b]]

## Answer

The validation error of miniCANDI's mean predictions is a scale error, not a depth or lambda error. On chr1, chr9 and chr21, nMSE is above 1 on 19 of 30 tracks and the scale term is a median 87% of it; an oracle affine map puts all 30 below 1. Predicted totals are within 0.77-1.33x of the observed; lambda is shared with the observed reads. The predicted mean is over-dispersed in counts on 16 of 30 (slope ratio above 1.3), and the convex p-value transform enlarges it (log-log power 1.66). On 9 tracks the observed reads themselves convert to nMSE above 1, so the store reads differ from the organizers'.
