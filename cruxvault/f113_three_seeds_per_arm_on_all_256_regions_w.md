---
id: f113
type: finding
schema: 3
title: Three seeds per arm on all 256 regions with a rule written before the…
parent: q36
confidence: moderate
why: Three seeds per arm on all 256 regions with a rule written before the scores, but the scoring thins the inputs as in training, so the result does not cover deployment at native depth.
evidence: results/rescoring_p256/read_20261118T1942Z/RESCORING_READ.md, results/rescoring_p256/read_20261118T1942Z/arms.tsv
created: "2026-11-18T16:21:20"
updated: "2026-11-18T16:21:20"
---

# f113 — Three seeds per arm on all 256 regions with a rule written before the…

Parent:: [[q36_does_hiding_depth_from_the_encoders_a_ha]]

## Answer

On all 256 regions (read of 2026-11-18 19:42 UTC), a fixed input depth keeps the U-Net trade-off: correction on crop pairs +0.166 (0.504 to 0.670; seed range 0.066, more than twice the range) and imputation -0.028 (0.432 to 0.404; seed range 0.023, beyond the range), composite +0.023 (range 0.010, 3 seeds each). On conv + attention the crop-pair correction rises by 0.039 (range 0.026, beyond) and imputation stays within (-0.011, range 0.012). On the conv autoencoder (2 seeds) every term stays within the range. By the pre-written rule 2.4 the switch stays a candidate for the mix-and-match round on the U-Net and on conv + attention, and enters no recipe from these scores, because the inputs were thinned at scoring as in training, which is not the deployment case; the native-depth score of the mix-and-match round decides deployability.
