---
id: f65
type: finding
schema: 3
title: "The eta-space rerun keeps the split verdict of f54 and qualifies it: part 2…"
parent: h37
direction: mixed
confidence: moderate
null: none
why: "The eta-space rerun keeps the split verdict of f54 and qualifies it: part 2 holds at every depth up to 12 to 23 times eic_v1's, and part 1 fails with sampling included at eic_v1 depth but holds above sampling for four of the narrow-mark and accessibility assays; moderate because the eta estimator is peak-weighted and untested against an independent Poisson check."
vs_expect: "Qualifies f54, which stays on record. Part 2 (sampling largest at 25 bp for broad marks): holds, and is a property of depth: the depth at which sampling equals the layers above it is 46 M (H3K36me3), 60 M (H3K27me3) and 99 M reads (H3K9me3), 12 to 23 times the common depth. Part 1 (cell type largest for narrow marks and accessibility): with sampling counted it still fails at eic_v1 depth (D* is 0.96 to 5.1 times the common depth for these assays, so sampling is at least about equal); above sampling, the cell-type layer exceeds the replicate layer beyond both intervals for DNase-seq (0.66 [0.23, 1.08] against 0.09 [-0.03, 0.20] of the sampling-free variance), H3K27ac (0.62 against 0.19), H3K4me1 (0.70 against 0.15) and H3K4me3 (0.44 against 0.08), and not for H3K4me2, H3K9ac and H3K79me2, whose intervals overlap. The f54 support from DNase-seq promoters (log1p cell-type share 0.51) is reversed in eta space (0.24, the lowest stratum)."
verifiables: "Check 1 (cell type exceeds every other layer beyond its interval, narrow marks and accessibility): not met with sampling counted; met above sampling for DNase-seq, H3K27ac, H3K4me1, H3K4me3. Check 2 (sampling exceeds every other layer, broad marks at 25 bp): met at eic_v1 depth (D* 12 to 23 times the common depth)."
evidence: results/eta_space/LAYERS_ETA.md, results/eta_space/eta_layers_by_assay.tsv, results/eta_space/eta_by_band_stratum_decile.tsv
created: "2026-11-14T16:37:38"
updated: "2026-11-14T16:37:38"
---

# f65 — The eta-space rerun keeps the split verdict of f54 and qualifies it: part 2…

Parent:: [[h37_on_real_data_the_cell_type_layer_is_the_]]
