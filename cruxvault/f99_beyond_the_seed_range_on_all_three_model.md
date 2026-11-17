---
id: f99
type: finding
schema: 3
title: Beyond the seed range on all three models.
parent: q44
confidence: moderate
why: Beyond the seed range on all three models.
evidence: results/ablations_closeout/ABLATIONS_READ.md
created: "2026-11-16T22:22:47"
updated: "2026-11-16T22:22:47"
---

# f99 — Beyond the seed range on all three models.

Parent:: [[q44_does_training_the_u_net_on_the_counterfa]]

## Answer

On the two conv models, as on the U-Net, the composite terms do not need the counterfactual views, and the correction-only term collapses without them (conv autoencoder 0.690 to 0.402, conv + attention 0.659 to 0.390). Training on alignment-level views only lowers imputation on both conv models (0.340 and 0.323).
