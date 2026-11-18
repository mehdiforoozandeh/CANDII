---
id: f106
type: finding
schema: 3
title: Twice the seed range on all three models with 3 seeds each on all 256…
parent: q44
confidence: strong
why: Twice the seed range on all three models with 3 seeds each on all 256 regions, by a rule written before the scores.
evidence: tasks/t349_onion_first_read_of_the_256_region_.md
created: "2026-11-18T09:48:32"
updated: "2026-11-18T09:48:32"
---

# f106 — Twice the seed range on all three models with 3 seeds each on all 256…

Parent:: [[q44_does_training_the_u_net_on_the_counterfa]]

## Answer

Yes for correction, no for the composite: on all 256 regions, training on the base view only lowers the correction-only term by more than twice the seed range on all three models (conv autoencoder -0.341, bar 0.067; conv + attention -0.315, bar 0.104; U-Net -0.262, bar 0.129), while the composite stays within the range (+0.007, +0.008, +0.020). All counterfactual views stay in training (rule 2.2).
