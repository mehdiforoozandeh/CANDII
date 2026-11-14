---
id: f53
type: finding
schema: 3
title: Read from layers_by_assay.tsv and layers_by_stratum.tsv of the rerun (job…
parent: q18
confidence: moderate
why: "Read from layers_by_assay.tsv and layers_by_stratum.tsv of the rerun (job 27265045); moderate because the layer mixes donor and experiment and is at half depth. It supersedes the first run's cell-type shares (f35: at most 0.29 at 25 bp, 0.63 to 0.91 at the broad band) and its ATAC-seq value 0.282, which was a total variance with no rung below it, not a cell-type layer; the 4 anisogenic pairs of cell C09 were dropped, so the 4 affected marks now have finite intervals."
evidence: results/t305/LAYERS_REAL.md, results/t305/layers_by_assay.tsv, results/t305/layers_by_stratum.tsv
created: "2026-11-14T03:35:47"
updated: "2026-11-14T03:35:47"
---

# f53 — Read from layers_by_assay.tsv and layers_by_stratum.tsv of the rerun (job…

Parent:: [[q18_how_large_is_each_layer_of_the_variance_]]

## Answer

Cell-type layer (other_cell_type minus isogenic; mixed with donor and experiment, because the anisogenic and other_experiment rungs are empty, so it is not a pure cell-type layer; 2,911 pairs, 31 assays): at 25 bp over all regions it is 0.2305 of the three layers for DNase-seq (0.014650 [0.011748, 0.017552]), 0.1986 for H3K27ac, 0.1952 for H3K4me3, 0.1403 for H3K4me1, and 0.0994, 0.0981, 0.0983 for H3K27me3, H3K36me3, H3K9me3; the ChIP-seq control has none (-0.0000674 [-0.00344, 0.00331]). The interval excludes 0 for 8 assays (DNase-seq, H3K27ac, H3K27me3, H3K36me3, H3K4me1, H3K4me3, H3K9ac, H3K9me3). It is the largest layer at the broad band (0.60 to 0.91 for the 11 assays with all three layers and at least 8 isogenic pairs; 0.67 to 0.91 for the narrow marks and DNase-seq), at the domain band for the narrow marks and DNase-seq (0.44 to 0.81), and in promoters for DNase-seq (0.51; H3K4me3 0.38, H3K27ac 0.34). ATAC-seq has no isogenic pair: its other_cell_type increment over sampling (0.0072 [-0.050, 0.064], 6 pairs, 3 cells) holds every layer above sampling. Caveats: log1p counts at half depth; the sampling share, and so the cell-type share's denominator, changes with depth.
