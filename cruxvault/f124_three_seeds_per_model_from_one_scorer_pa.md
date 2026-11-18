---
id: f124
type: finding
schema: 3
title: Three seeds per model from one scorer pass on the 16-region screening tier,…
parent: q22
confidence: moderate
why: Three seeds per model from one scorer pass on the 16-region screening tier, with several metrics undefined (all signal-space macros, budget terms in the accessibility-only regime).
evidence: results/closeout_E3/tables/tables.md
created: "2026-11-18T16:23:10"
updated: "2026-11-18T16:23:10"
---

# f124 — Three seeds per model from one scorer pass on the 16-region screening tier,…

Parent:: [[q22_does_the_smallest_model_beat_the_baselin]]

## Answer

The full set of computed metrics (16 pair regions, 256 regions for imputation, 3 seeds, equal windows; leave-one-assay-out regime, 43 items, 9 scorable assays) shows where the models stand against the baselines. Error budget: the read-noise share of the target variance is 0.567 for every model, so 0.433 is the most any model can explain; the conv autoencoder explains 0.257 (seed range 0.005), conv + attention 0.269, the U-Net 0.255, the axial transformer 0.250, Perceiver IO 0.102, the position average 0.221. Depth-law error on never-trained pairs (mean absolute log2 error): 0.106 (U-Net, lr 3e-3) to 0.196 (Perceiver IO) against 0.316 for the position average; the best depth-law model is not the one with the best gap closed (U-Net lr 3e-3 0.106 against conv autoencoder 0.177, but crop-pair gap closed 0.534 against 0.718). In the accessibility-only regime (20 items, 8 assays) no model beats the position average on count CRPS beyond its seed range (skill -0.051 to 0.033; Perceiver IO -0.073), though r_spec stays positive (0.11 to 0.26) and the pooled PIT deviation is lower (0.029 to 0.046 against 0.056). The predicted-over-replicate variance ratio is above 1 for every model (1.71 to 3.94 per seed mean), driven by single targets.
