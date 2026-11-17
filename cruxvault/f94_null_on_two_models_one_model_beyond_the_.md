---
id: f94
type: finding
schema: 3
title: Null on two models, one model beyond the range, two confounded arms, one arm…
parent: q52
confidence: weak
why: Null on two models, one model beyond the range, two confounded arms, one arm missing.
evidence: results/ablations_closeout/ABLATIONS_READ.md
created: "2026-11-16T22:22:46"
updated: "2026-11-16T22:22:46"
---

# f94 — Null on two models, one model beyond the range, two confounded arms, one arm…

Parent:: [[q52_do_the_dna_source_and_the_way_dna_featur]]

## Answer

Nucleotide-transformer embeddings score within the seed range of the own DNA tower on all three models. DNA off and a 5 Mb shifted DNA lower imputation by about 0.02, beyond the range on conv + attention only. The fusion point does not matter on the conv autoencoder or the U-Net; late joins lower conv + attention imputation. The Flashzoi arm was not run. The U-Net DNA-off arm and the U-Net early-fusion arm differ from the base in width or parameter count.
