---
id: f393
type: finding
schema: 3
title: 3 experiments; reproduction exact, placements are observed-read rungs, not…
parent: q90
confidence: moderate
why: 3 experiments; reproduction exact, placements are observed-read rungs, not model predictions
evidence: results/t390/moesm2/MOESM2.md
created: "2026-11-23T02:05:18"
updated: "2026-11-23T02:05:18"
---

# f393 — 3 experiments; reproduction exact, placements are observed-read rungs, not…

Parent:: [[q90_can_a_count_space_model_like_minicandi_b]]

## Answer

The MOESM2 (original paired-end truth) board is scorable exactly: the challenge scorer with the original truth reproduces all 25 MOESM2 positions on the 3 probe experiments (Max's cache: 0 rank differences; ours: <=1.9e-8 on 7 measures, gwspear 1.5e-5). Observed single-end reads are no oracle there: the pipeline track ranks 15, 1 and 17 of 26 (combined rank, H3K4me3, H3K27me3, DNase), and the DNase track that equals Jacob's single-end truth ranks 21 of 26 on mse. That is the size of the processing shift MOESM2 tests; a miniCANDI entry must predict at the paired-end measurement covariates.
