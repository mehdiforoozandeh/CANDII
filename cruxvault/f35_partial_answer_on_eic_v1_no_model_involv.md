---
id: f35
type: finding
schema: 3
title: "Partial answer on eic_v1 (no model involved): only two layers resolve, a…"
parent: q18
confidence: weak
evidence: results/t305/LAYERS_REAL.md
created: "2026-11-14T02:17:47"
updated: "2026-11-14T02:17:47"
---

# f35 — Partial answer on eic_v1 (no model involved): only two layers resolve, a…

Parent:: [[q18_how_large_is_each_layer_of_the_variance_]]

## Answer

Partial answer on eic_v1 (no model involved): only two layers resolve, a replicate layer (isogenic pair value; holds sampling, sequencing, library and biological-replicate variance together; 219 pairs, 34 assays) and a cell-type layer (other_cell_type increment; holds donor and experiment too; 2,911 pairs, 31 assays). At 25 bp over all regions the cell-type share is at most 0.29 (DNase-seq 0.0293 against a replicate layer of 0.0732; H3K4me3 0.24; H3K9me3 0.12; chipseq-control 0.007), so the replicate layer is 71% to 99% of the two layers; at the broad band the cell-type share is 0.63 to 0.91 for every assay, and in promoters it is largest (DNase-seq 0.61, H3K4me3 0.49). The sampling, sequencing and technical layers are not separated: EIC gives no sequencing or technical pairs, and the pseudoreplicate pairs were dropped by evalset.py:581 (a != b); a layers rerun with the training and validation pseudoreplicate pairs is in progress. Absolute layers are valid at each assay's common depth only (1,510,281 to 37,024,952 reads).
