---
id: f84
type: finding
schema: 3
title: one seed per trial, the pointwise MLP not run, and the composite of the Fir…
parent: q39
confidence: weak
why: one seed per trial, the pointwise MLP not run, and the composite of the Fir runs has no same-cluster base yet
evidence: results/ladder_tuning/ROUND2_READ.md
exploratory: true
created: "2026-11-15T21:45:45"
updated: "2026-11-15T21:45:45"
---

# f84 — one seed per trial, the pointwise MLP not run, and the composite of the Fir…

Parent:: [[q39_do_the_context_rungs_2_to_6_convolution_]]

## Answer

First tuning round (one seed per trial, held-out NB loss): three architectures gained past the margin (conv autoencoder with 3 levels -0.015, conv + attention with attention at 128 positions -0.009, U-Net with lr 3e-3 -0.007); the axial transformer and Perceiver IO did not. Perceiver IO stays last under every trial. Held-out loss ranks U-Net first and the conv models third and fourth, the reverse of the composite. Twice the windows lowered held-out loss in all three architectures tested, so none had converged at 80,000 windows.
