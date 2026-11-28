---
id: f395
type: finding
schema: 3
title: 3 experiments, observed reads; how to make two halves from a model's…
parent: q90
confidence: moderate
why: 3 experiments, observed reads; how to make two halves from a model's prediction is untested
evidence: results/t390/ladder_probe2/PROBE2.md
created: "2026-11-23T02:12:04"
updated: "2026-11-23T02:12:04"
---

# f395 — 3 experiments, observed reads; how to make two halves from a model's…

Parent:: [[q90_can_a_count_space_model_like_minicandi_b]]

## Answer

The 25 bp overlap counts are the main conversion loss, and a 25 bp fragment pileup removes most of it: against the exact 1 bp formula nMSE falls from 0.017/0.255/0.021 to 0.0021/0.062/0.00003 (H3K4me3/H3K27me3/DNase). With half-read peaks the normalized fragment route reaches qn mse 1.74/0.066/1.96 vs 3.57/0.116/16.5 for counts, and ranks 1 of 26 on almost every MOESM4 measure on observed reads. A count-space entrant should output a 25 bp fragment pileup (or stranded read starts) and take its peaks by the half-read overlap rule.
