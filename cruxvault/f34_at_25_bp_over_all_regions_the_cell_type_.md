---
id: f34
type: finding
schema: 3
title: At 25 bp over all regions the cell-type layer is never the largest (share at…
parent: h37
direction: mixed
confidence: weak
null: none
why: At 25 bp over all regions the cell-type layer is never the largest (share at most 0.29, DNase-seq), but it dominates at the broad and domain scales and in DNase-seq promoters, and the sampling layer is not separated because the pseudoreplicate pairs were dropped by a selection defect, so neither part can be judged as stated.
vs_expect: "Part 1 (cell-type layer largest for narrow marks and accessibility): fails at 25 bp over all regions (replicate layer 2.5 to 30 times larger in the 28 target assays with a positive cell-type layer); holds at the broad and domain scales (cell-type share 0.50 to 0.91 for the narrow marks and DNase-seq) and in promoters for DNase-seq (0.61). Part 2 (sampling layer largest at 25 bp for broad marks): untestable as stated, because only two layers resolve on eic_v1 (isogenic replicate layer, 219 pairs; cell-type layer, 2,911 pairs); its weaker form holds (replicate layer, which contains sampling, is 84% to 90% of the two layers at the fine band for H3K27me3, H3K36me3, H3K9me3). A rerun with the training and validation pseudoreplicate pairs is in progress (I1a resumed), so this finding is interim."
verifiables: "Check 1 (cell-type layer exceeds every other layer beyond its interval, narrow marks and accessibility): not met at 25 bp over all regions; the cell-type interval excludes 0 for only 6 of 30 assays. Check 2 (sampling layer, broad marks): not evaluable, the pseudoreplicate rung is empty."
evidence: results/t305/LAYERS_REAL.md
created: "2026-11-14T02:17:47"
updated: "2026-11-14T02:17:47"
---

# f34 — At 25 bp over all regions the cell-type layer is never the largest (share at…

Parent:: [[h37_on_real_data_the_cell_type_layer_is_the_]]
