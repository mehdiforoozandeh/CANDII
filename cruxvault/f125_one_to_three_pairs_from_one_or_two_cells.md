---
id: f125
type: finding
schema: 3
title: One to three pairs from one or two cells per assay, unclipped values above 1…
parent: q24
confidence: weak
why: One to three pairs from one or two cells per assay, unclipped values above 1 on two assays, and only one of the layers computable.
evidence: results/closeout_E3/tables/tables.md
created: "2026-11-18T16:23:18"
updated: "2026-11-18T16:23:18"
---

# f125 — One to three pairs from one or two cells per assay, unclipped values above 1…

Parent:: [[q24_how_much_of_each_variance_layer_does_the]]

## Answer

Only the cell-type layer can be read on this evaluation set: the isogenic and anisogenic layers are 0 by construction (both members get the same inputs), and the sampling and other-experiment layers have no pair with two predictions. Explained fraction of the cell-type layer (part of the true difference between two cell types that the difference of the two predictions captures; 3 seeds): on the 3-pair assays other than H3K4me2, the conv, U-Net and axial models capture 0.00 to 0.48 (conv autoencoder base: H3K27ac 0.41, H3K27me3 0.11, H3K36me3 0.41, H3K4me1 0.18, H3K79me2 0.28, H3K9me3 0.14, H4K20me1 0.29), Perceiver IO at most 0.15; both baselines give 0 by definition. H3K4me2 (1.66 to 2.30) and H2AFZ (4.81 to 8.29, 1 pair) are above 1, i.e. the predictions differ more than the truths. Twelve of 22 assays have no row.
