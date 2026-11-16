---
id: f85
type: finding
schema: 3
title: recomputed from stored per-pair records for 78 seeds, matching the scorer to…
parent: q46
confidence: moderate
why: recomputed from stored per-pair records for 78 seeds, matching the scorer to 1e-10; the copy mechanism was not tested directly and the conv autoencoder still falls with more windows
evidence: results/correction_metric_review/REVIEW.md
created: "2026-11-16T01:14:00"
updated: "2026-11-16T01:14:00"
---

# f85 — recomputed from stored per-pair records for 78 seeds, matching the scorer to…

Parent:: [[q46_why_does_the_held_out_nb_loss_rank_the_u]]

## Answer

The composite's gap-closed term is at fault. 84 of its 102 pairs are no-MAPQ-filter + no-duplicate-removal pairs, which carry 0.81 of the macro weight; all three odd behaviors (scrambled covariates raise it, added views lower it, more training lowers it) sit in those pairs. 81 to 85% of the term's denominator is denoising of the source half, not covariate cost. On the 18 crop pairs alone the scrambled control scores below the U-Net base (0.420 vs 0.519), the drop from added views disappears (0.505 to 0.553 in all view arms), and the U-Net rises with 160k windows (0.507 to 0.552). Likely mechanism, not proven: training on full views that share reads teaches a near-identity copy, which the disjoint-half scoring penalizes.
