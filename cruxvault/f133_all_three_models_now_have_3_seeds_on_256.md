---
id: f133
type: finding
schema: 3
title: All three models now have 3 seeds on 256 regions; the choice for conv +…
parent: q54
confidence: moderate
why: All three models now have 3 seeds on 256 regions; the choice for conv + attention still waits for the mix-and-match round.
evidence: results/rescoring_p256/read_20261119T0110Z/arms.tsv
created: "2026-11-18T18:09:18"
updated: "2026-11-18T18:09:18"
---

# f133 — All three models now have 3 seeds on 256 regions; the choice for conv +…

Parent:: [[q54_does_another_loss_poisson_mean_nb_crps_e]]

## Answer

On all 256 regions imputation weight 4 moves no goal term of the U-Net beyond the seed range (imputation +0.007, range 0.025; crop-pair correction +0.021, range 0.068). It raises imputation on conv + attention (+0.039, range 0.017) and costs the conv autoencoder crop-pair correction (-0.050, range 0.021). Weight 4 stays a candidate for conv + attention only.
