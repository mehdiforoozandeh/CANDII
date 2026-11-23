---
id: f142
type: finding
schema: 3
title: Pre-set gate G5 rule 1 holds on the U-Net by a clear margin over seed range;…
parent: q76
confidence: moderate
why: Pre-set gate G5 rule 1 holds on the U-Net by a clear margin over seed range; one data version, 8 held-out cells, pre-pairs arms.
evidence: results/t370/control/control_read.md
created: "2026-11-22T14:07:02"
updated: "2026-11-22T14:07:02"
---

# f142 — Pre-set gate G5 rule 1 holds on the U-Net by a clear margin over seed range;…

Parent:: [[q76_can_minicandi_predict_the_chip_seq_contr]]

## Answer

Partly yes for the U-Net: on 8 cells whose control was never seen in training, the predicted control beats the training-cell average (CRPS skill 0.131, seed range 0.017) and its signal p-value tracks the observed one better (Pearson 0.948 vs 0.932, range 0.010). The conv models do not pass. Measured before read-disjoint pairs; rerun on the recipe is training.
