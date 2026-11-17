---
id: f96
type: finding
schema: 3
title: A consistent null across three models; one single-model gain.
parent: q54
confidence: moderate
why: A consistent null across three models; one single-model gain.
evidence: results/ablations_closeout/ABLATIONS_READ.md
created: "2026-11-16T22:22:46"
updated: "2026-11-16T22:22:46"
---

# f96 — A consistent null across three models; one single-model gain.

Parent:: [[q54_does_another_loss_poisson_mean_nb_crps_e]]

## Answer

No loss raises the composite beyond the seed range on two or more models. Imputation weight 4 raises conv + attention imputation from 0.387 to 0.426 (beyond the range) with composite 0.559.
