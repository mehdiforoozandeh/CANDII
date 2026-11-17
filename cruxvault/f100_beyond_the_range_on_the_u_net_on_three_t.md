---
id: f100
type: finding
schema: 3
title: Beyond the range on the U-Net on three terms; the mechanism is not checked.
parent: q36
confidence: moderate
why: Beyond the range on the U-Net on three terms; the mechanism is not checked.
evidence: results/ablations_closeout/ABLATIONS_READ.md
created: "2026-11-16T22:22:47"
updated: "2026-11-16T22:22:47"
---

# f100 — Beyond the range on the U-Net on three terms; the mechanism is not checked.

Parent:: [[q36_does_hiding_depth_from_the_encoders_a_ha]]

## Answer

A fixed input depth of 2^23 on the U-Net raises crop-pair gap closed from 0.507 to 0.675 and the composite from 0.495 to 0.518, and lowers imputation from 0.432 to 0.404; all three beyond the seed range. On conv + attention the gap closed rises to 0.722. The conv autoencoder arm has 2 seeds.
