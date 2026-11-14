---
id: f63
type: finding
schema: 3
title: Read from eta_layers_by_assay.tsv with 95% cluster-bootstrap intervals over…
parent: q33
confidence: moderate
why: Read from eta_layers_by_assay.tsv with 95% cluster-bootstrap intervals over cells; moderate because the estimate is peak-weighted and rests on 8 to 19 isogenic pairs per assay.
evidence: results/eta_space/LAYERS_ETA.md, results/eta_space/eta_layers_by_assay.tsv
created: "2026-11-14T16:37:14"
updated: "2026-11-14T16:37:14"
---

# f63 — Read from eta_layers_by_assay.tsv with 95% cluster-bootstrap intervals over…

Parent:: [[q33_can_the_negative_binomial_be_written_wit]]

## Answer

The depth D* at which read sampling noise equals the summed replicate and cell-type layers above it (eta space, common-depth pairs, all regions): 2.8 M reads (H3K4me3, [1.9, 4.0]) to 99 M reads (H3K9me3, [45, 216]) for the 10 target assays, i.e. 1.0 to 23 times their common depth (DNase-seq 6.4 M, 0.96 x; H3K27ac 5.3 M, 2.2 x; H3K36me3 46 M, 12 x; H3K27me3 60 M, 13 x); the ChIP-seq control 297 M [221, 398], 104 x. Every assay is at or below its D*, so 'sampling dominates at 25 bp' is a property of the depth of eic_v1, not of the biology. D* is one depth-free number per assay and replaces depth-dependent shares as the headline form. Caveats: peak-weighted moments; the broad marks' intervals are wide.
