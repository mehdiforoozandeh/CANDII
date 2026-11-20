---
id: f51
type: finding
schema: 3
title: Read from the rerun with the pseudoreplicate pairs added (Nibi job…
parent: q18
confidence: moderate
why: Read from the rerun with the pseudoreplicate pairs added (Nibi job 27265045); moderate because the shares are at half depth on log1p counts and a full-depth or variance-stabilized check is still open. It supersedes the first run's two-layer answer on this question (f35), whose replicate layer held sampling inside it; the first-run note is now results/t305/LAYERS_REAL_first_run.md.
evidence: results/t305/LAYERS_REAL.md, results/t305/layers_by_assay.tsv, results/t305/layers_by_stratum.tsv
created: "2026-11-14T03:35:47"
updated: "2026-11-14T03:35:47"
---

# f51 — Read from the rerun with the pseudoreplicate pairs added (Nibi job…

Parent:: [[q18_how_large_is_each_layer_of_the_variance_]]

## Answer

Sampling layer (pseudoreplicate pair value, two read halves of one library; 778 pairs, 36 assays): on eic_v1 at 25 bp over all regions it is the largest of the three layers in every assay, 0.7602 of the three layers for DNase-seq (0.048328 [0.044995, 0.051661], 33 pairs) up to 0.9748 for the ChIP-seq control (0.035031 [0.032219, 0.037842], 88 pairs); the broad marks H3K27me3, H3K36me3, H3K9me3 are 0.8783, 0.8939, 0.8828, and H3K27ac and H3K4me3 0.7875 and 0.7905 (layers_by_assay.tsv, share_of_layers). At the fine band it is 0.85 to 0.95; at the broad band 0.01 to 0.13. Caveats: values are 0.5 E[(Y-Y')^2] of log1p counts in 25 bp bins at the assay's common depth, which is now a pseudoreplicate half (about half the first run's depth, e.g. H3K4me3 2,323,010 against 4,647,553 reads); on log1p counts the sampling share shrinks as depth grows, so at full depth it is smaller than reported; the cell-type layer it is compared against is mixed with donor and experiment; ATAC-seq has no isogenic rung (4 pseudoreplicate pairs, few pairs).
