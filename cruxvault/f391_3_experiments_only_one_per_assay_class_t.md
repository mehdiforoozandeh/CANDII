---
id: f391
type: finding
schema: 3
title: 3 experiments only, one per assay class; the model-side test needs blind…
parent: q90
confidence: moderate
why: 3 experiments only, one per assay class; the model-side test needs blind predictions that do not exist yet
evidence: results/t390/ladder_probe/LADDER_PROBE.md
created: "2026-11-22T23:15:15"
updated: "2026-11-22T23:15:15"
---

# f391 — 3 experiments only, one per assay class; the model-side test needs blind…

Parent:: [[q90_can_a_count_space_model_like_minicandi_b]]

## Answer

Partly. The organizers' side is exact: quantile normalization and the 1 bp MACS2 formula reproduce the truth chain (3 of 3). Two costs remain for a count-space entry. First, 25 bp overlap counts cost normalized MSE 0.017 (H3K4me3), 0.021 (DNase) and 0.255 (H3K27me3) against the exact formula. Second, the truth's normalization used the pseudoreplicate-overlap peak set; with that set our DNase route equals the truth exactly, while a raw peak set gave nMSE 0.04 to 0.69. Our ChIP reprocessing differs from Jacob's before the p-value step (nMSE 0.013 and 0.127), cause unknown. On observed reads the full count route ranks 1 or 2 of 26 on every MOESM4 measure, which is an oracle bound on conversion cost only.
