---
id: f45
type: finding
schema: 3
title: Class-based strata barely change what is sampled at the model's scale; a…
parent: q21
confidence: strong
evidence: results/t292/STRATA_REAL.md
created: "2026-11-14T02:19:05"
updated: "2026-11-14T02:19:05"
---

# f45 — Class-based strata barely change what is sampled at the model's scale; a…

Parent:: [[q21_does_stratified_selection_of_training_re]]

## Answer

Class-based strata barely change what is sampled at the model's scale; a continuous tilt does. On eic_v1 (51,394 sampling tiles of 51.2 kb; 267 training tracks in 51 cells) 23 tiles hold no training peak (genome share 0.0004), so the class lists are near-uniform: effective sample size per draw 0.94 (peak frequency), 0.86 (cCRE-like, built from training peaks; promoter-like tiles are 69.6% of the genome share) and 0.905 (cross-cell variance); only kcenter departs far from uniform (0.057). Lists with a continuous score and a solved tilt reach the target effective sample size 0.50 and 0.25 (cCRE density alpha 0.624 and 0.899; variance alpha 2.002). The cCRE-like density and the cross-cell variance are anti-correlated across tiles (Spearman rho of q -0.511), so the two tilts favor different regions. This is a property of the data, not a result on model accuracy.
