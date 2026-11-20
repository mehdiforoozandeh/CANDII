---
id: f64
type: finding
schema: 3
title: Measured on the same 3,908 pairs as f51 to f53 with intervals; moderate…
parent: q18
confidence: moderate
why: Measured on the same 3,908 pairs as f51 to f53 with intervals; moderate because the eta estimator is peak-weighted, untested against an independent Poisson check, and unresolved for the broad repressive marks.
evidence: results/eta_space/LAYERS_ETA.md, results/eta_space/eta_layers_by_assay.tsv, results/eta_space/eta_by_band_stratum_decile.tsv
created: "2026-11-14T16:37:27"
updated: "2026-11-14T16:37:27"
---

# f64 — Measured on the same 3,908 pairs as f51 to f53 with intervals; moderate…

Parent:: [[q18_how_large_is_each_layer_of_the_variance_]]

## Answer

In eta space (squared CV of the rate, sampling removed by the cross moment of the pseudoreplicate halves), the layer sizes per assay at the common depth are: cell-type layer (mixed with donor and experiment) 0.66 of the sampling-free variance for DNase-seq, 0.62 H3K27ac, 0.70 H3K4me1, 0.44 H3K4me3, 0.27 to 0.43 for H3K79me2, H3K4me2, H3K36me3, H3K9ac; replicate layer 0.08 to 0.25 of it for the narrow marks and DNase-seq and 0.47 [0.05, 0.88] (H3K27me3) and 0.59 [-0.38, 1.56] (H3K9me3) for the broad repressive marks. This qualifies three log1p findings without replacing them: f51 (sampling largest in every assay) still holds at eic_v1 depth, but only because every assay sits at or below the depth where sampling equals the layers above it, so the 0.76 to 0.97 sampling share is a property of the depth; f52 (replicate layer about 0.01 to 0.03 of three layers) understates the replicate layer, which looked small because sampling filled the denominator (as a share of the two layers above sampling: DNase-seq 0.12 and H3K27ac 0.24 in eta space against 0.04 and 0.07 on log1p); f53 (the ChIP-seq control has no cell-type layer; promoters carry the largest cell-type share) is reversed on both points: the control has a cell-type layer of 0.38 [0.23, 0.54] of its sampling-free variance (703 pairs; positive in every stratum and rate decile), and promoters have the lowest eta cell-type share of the four strata (DNase-seq 0.24 [0.16, 0.33], H3K27ac 0.38, H3K4me3 0.25, against 0.51, 0.34, 0.38 on log1p). Caveats: peak-weighted moments; the fine band is biased; no independent test of the Poisson assumption; the H3K27me3 and H3K9me3 intervals are about plus or minus 1 wide and their pooled cell-type layer is negative (positive within rate deciles).
