---
id: f54
type: finding
schema: 3
title: The broad-mark part holds by a clear margin and the narrow-mark part fails…
parent: h37
direction: mixed
confidence: moderate
null: none
why: The broad-mark part holds by a clear margin and the narrow-mark part fails at 25 bp over all regions, so the claim as stated is half right; moderate because the shares are on log1p counts at half depth (the sampling share shrinks with depth) and the cell-type layer is mixed with donor and experiment. It supersedes f34, whose broad-mark part was untestable because the first run had no sampling layer (first-run note now results/t305/LAYERS_REAL_first_run.md).
vs_expect: "Part 2 (sampling layer largest at 25 bp for broad marks): holds; sampling is 0.8783, 0.8939, 0.8828 of the three layers for H3K27me3, H3K36me3, H3K9me3 (e.g. H3K27me3 0.047036 [0.041044, 0.053027] against a cell-type layer of 0.005325 and a replicate layer of 0.001190). Part 1 (cell-type layer largest for narrow marks and accessibility): fails at 25 bp over all regions; sampling is the largest layer there as well (0.7602 DNase-seq to 0.8479 H3K4me1) and the cell-type share is at most 0.2305 (DNase-seq). Part 1 holds at the broad band (cell-type share 0.67 to 0.91 for the narrow marks and DNase-seq) and in DNase-seq promoters (0.51)."
verifiables: "Check 1 (cell-type layer exceeds every other layer beyond its interval, narrow marks and accessibility): not met at 25 bp over all regions; for every such assay the sampling interval lies wholly above the cell-type interval. Check 2 (sampling layer exceeds every other layer beyond its interval, broad marks at 25 bp): met for H3K27me3, H3K36me3 and H3K9me3."
evidence: results/t305/LAYERS_REAL.md, results/t305/layers_by_assay.tsv, results/t305/layers_by_stratum.tsv
created: "2026-11-14T03:35:47"
updated: "2026-11-14T03:35:47"
---

# f54 — The broad-mark part holds by a clear margin and the narrow-mark part fails…

Parent:: [[h37_on_real_data_the_cell_type_layer_is_the_]]
