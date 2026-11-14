---
id: f52
type: finding
schema: 3
title: Read from layers_by_assay.tsv of the rerun (job 27265045); moderate because…
parent: q18
confidence: moderate
why: Read from layers_by_assay.tsv of the rerun (job 27265045); moderate because the layer pools three sources and is measured at half depth. It supersedes the first run's replicate layer (f35), which was the isogenic pair value with sampling inside it (71% to 99% of two layers).
evidence: results/t305/LAYERS_REAL.md, results/t305/layers_by_assay.tsv
created: "2026-11-14T03:35:47"
updated: "2026-11-14T03:35:47"
---

# f52 — Read from layers_by_assay.tsv of the rerun (job 27265045); moderate because…

Parent:: [[q18_how_large_is_each_layer_of_the_variance_]]

## Answer

Replicate layer above sampling (isogenic minus pseudoreplicate; holds sequencing, library and biological-replicate variance together, because those rungs are empty in EIC; 219 isogenic pairs, 34 assays): at 25 bp over all regions it is 0.001446 (H4K20me1) to 0.06036 (H4K5ac) of the three layers, about 0.01 to 0.03 for most marks (DNase-seq 0.009357, H3K27ac 0.01388, H3K4me3 0.01428, H3K27me3 0.02223). Its 95% cluster-bootstrap interval excludes 0 for 8 assays (H2AFZ, H3K27ac, H3K27me3, H3K4me1, H3K4me2, H3K4me3, H3K79me2, H3K9ac); e.g. H3K4me3 0.000400 [0.000149, 0.000650] against a sampling layer of 0.022163. So at 25 bp two pseudoreplicate halves of one experiment differ almost as much as two biological replicates. Caveats: log1p counts at half depth; 17 assays rest on fewer than 5 isogenic pairs or 3 cells (flag few pairs); only 8 target assays and the control have at least 10 isogenic pairs.
