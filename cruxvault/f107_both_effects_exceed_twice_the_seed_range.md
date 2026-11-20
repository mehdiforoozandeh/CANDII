---
id: f107
type: finding
schema: 3
title: Both effects exceed twice the seed range on 256 regions, but the U-Net arm…
parent: q54
confidence: moderate
why: Both effects exceed twice the seed range on 256 regions, but the U-Net arm is unscored and the choice for conv + attention waits for the mix-and-match round.
evidence: tasks/t349_onion_first_read_of_the_256_region_.md
created: "2026-11-18T09:48:32"
updated: "2026-11-18T09:48:32"
---

# f107 — Both effects exceed twice the seed range on 256 regions, but the U-Net arm…

Parent:: [[q54_does_another_loss_poisson_mean_nb_crps_e]]

## Answer

Imputation loss weight 4 costs crop-pair correction on the conv autoencoder (-0.050, bar 0.021, on 256 regions) and stays a candidate on the conv + attention only (imputation +0.039, bar 0.017; crop-pair correction within). The U-Net arm was not scored yet; it cannot change this verdict (rule 2.6). The weights stay 1 / 1 / 1 until the mix-and-match round.
