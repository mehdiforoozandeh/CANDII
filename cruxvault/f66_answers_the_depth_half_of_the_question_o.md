---
id: f66
type: finding
schema: 3
title: Answers the depth half of the question on real data by thinning and by an…
parent: q31
confidence: moderate
why: Answers the depth half of the question on real data by thinning and by an exact depth-free scale; moderate because the higher depths are extrapolated, the variance-stabilized scale was not computed, and the eta estimator is peak-weighted.
evidence: results/eta_space/LAYERS_ETA.md, results/eta_space/depth_invariance.tsv, results/eta_space/THEORY.md
created: "2026-11-14T16:37:49"
updated: "2026-11-14T16:37:49"
---

# f66 — Answers the depth half of the question on real data by thinning and by an…

Parent:: [[q31_how_do_the_variance_layer_shares_change_]]

## Answer

Partly answered by the eta-space work. On log1p counts the layer shares are a property of depth: thinning from the common (pseudoreplicate-half) depth to 1/2 and 1/4 lowered the log1p cell-type share of the 10 target assays by 8% to 63% (H3K27ac 0.199, 0.170, 0.139), and in simulation the log1p sampling share fell from 0.979 to 0.057 as the mean count rose from 0.005 to 100 per bin with the latent layers fixed. On a depth-free scale (squared CV of the rate, i.e. eta space; sampling removed by moments, no log) the replicate and cell-type layers moved by at most 10% over the same thinnings (H3K27ac cell type 5.31, 5.31, 5.37), so the shares above sampling do not change with depth. At full depth the sampling share is therefore smaller than the 76% to 97% measured at half depth, and the depth at which it equals the layers above it is 2.8 M to 99 M reads (1.0 to 23 times the common depth). A variance-stabilizing transform such as the Anscombe root was not computed; the theory note expects it to keep the depth dependence of log1p. Not measured: shares at depths above the full read set (no deeper libraries in eic_v1).
