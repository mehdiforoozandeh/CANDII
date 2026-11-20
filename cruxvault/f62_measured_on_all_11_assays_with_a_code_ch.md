---
id: f62
type: finding
schema: 3
title: Measured on all 11 assays with a code check that reproduces the log1p layers…
parent: q33
confidence: moderate
why: Measured on all 11 assays with a code check that reproduces the log1p layers exactly; moderate because the depths are nested thinnings of one read set and the estimator is peak-weighted.
evidence: results/eta_space/LAYERS_ETA.md, results/eta_space/depth_invariance.tsv, results/eta_space/consistency.tsv
created: "2026-11-14T16:37:14"
updated: "2026-11-14T16:37:14"
---

# f62 — Measured on all 11 assays with a code check that reproduces the log1p layers…

Parent:: [[q33_can_the_negative_binomial_be_written_wit]]

## Answer

Real data (eic_v1, the 3,908 layer pairs, Nibi job 27341631): thinning every pair from the assay's common depth to 1/2 and 1/4 of it moved the eta-space (squared-CV) replicate and cell-type layers by at most 10% (largest: H3K4me1 replicate layer 0.514, 0.525, 0.565; cell-type layer at most 8%, H3K9ac 2.06, 2.16, 2.23), while the log1p cell-type share of the 10 target assays fell by 8% (H3K9me3 0.098 to 0.091) to 63% (H3K4me2 0.148, 0.096, 0.054) from the common depth to 1/4 of it. H3K27ac: log1p cell-type share 0.199, 0.170, 0.139; eta cell-type layer 5.31, 5.31, 5.37 (CV^2). The eta sampling term doubled with each halving (DNase-seq 7.1, 14.1, 28.2), as m/D predicts. So the eta-space layers are depth-invariant on real data and the log1p shares are not. Caveats: the three depths reuse the same reads (nested thinning), so they are not independent estimates; the moments are weighted toward peak bins; the fine band is biased (shares above 1, colored sampling noise between adjacent 25 bp bins, cause not found); the full-track consistency check (0.994 to 1.010) confirms the halves and depths but is not an independent test of the Poisson assumption.
