---
id: f132
type: finding
schema: 3
title: 3 seeds per model on 256 regions and the signs agree with the 16-region read…
parent: q50
confidence: moderate
why: 3 seeds per model on 256 regions and the signs agree with the 16-region read in all 15 cells; one model carries the effect.
evidence: results/rescoring_p256/read_20261119T0110Z/arms.tsv
created: "2026-11-18T18:09:18"
updated: "2026-11-18T18:09:18"
---

# f132 — 3 seeds per model on 256 regions and the signs agree with the 16-region read…

Parent:: [[q50_does_the_leave_one_out_average_of_the_tr]]

## Answer

On all 256 regions the leave-one-out average input helps imputation only on the U-Net (+0.035, seed range 0.023). On the two conv models every goal term stays within the seed range; only the correction-only term rises (conv autoencoder +0.098, range 0.034; conv + attention +0.083, range 0.048). The input stays in the U-Net pool and stays off on the conv models.
